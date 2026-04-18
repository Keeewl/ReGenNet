# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import pickle
import numpy as np
import torch
from utils.parser_util import cgenerate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from utils.online_window import sliding_window_sample
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
import shutil
from scipy.ndimage import gaussian_filter1d
from data_loaders.tensors import ccollate
import time


def _load_interaction_order(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_actor_is_p1(order_dict, dataset_key):
    if not order_dict or not dataset_key:
        return -1
    label = order_dict.get(dataset_key, None)
    if label is None:
        return -1
    return 1 if int(label) == 1 else 0


def _to_obj_array(values):
    return np.array(values, dtype=object)


def _build_meta_payload(meta_list):
    if not meta_list:
        return {}
    payload = {}
    payload["meta_version"] = np.array("v1")
    payload["sample_idx"] = np.arange(len(meta_list), dtype=np.int64)
    payload["rep_i"] = np.array([m.get("rep_i", -1) for m in meta_list], dtype=np.int64)
    payload["action_i"] = np.array([m.get("action_i", -1) for m in meta_list], dtype=np.int64)
    payload["action_id"] = np.array([m.get("action_id", -1) for m in meta_list], dtype=np.int64)
    payload["data_index"] = np.array([m.get("data_index", -1) for m in meta_list], dtype=np.int64)
    payload["length"] = np.array([m.get("length", -1) for m in meta_list], dtype=np.int64)
    payload["raw_nframes"] = np.array([m.get("raw_nframes", -1) for m in meta_list], dtype=np.int64)
    payload["start_frame"] = np.array([m.get("start_frame", -1) for m in meta_list], dtype=np.int64)
    payload["end_frame"] = np.array([m.get("end_frame", -1) for m in meta_list], dtype=np.int64)
    payload["num_frames"] = np.array([m.get("num_frames", -1) for m in meta_list], dtype=np.int64)
    payload["motion_length"] = np.array([m.get("motion_length", -1) for m in meta_list], dtype=np.int64)
    payload["sampling_step"] = np.array([m.get("sampling_step", -1) for m in meta_list], dtype=np.int64)
    payload["actor_is_p1"] = np.array([m.get("actor_is_p1", -1) for m in meta_list], dtype=np.int64)
    payload["downsample"] = np.array([m.get("downsample", -1) for m in meta_list], dtype=np.int64)

    payload["dataset_key"] = _to_obj_array([m.get("dataset_key", "") for m in meta_list])
    payload["action_name"] = _to_obj_array([m.get("action_name", "") for m in meta_list])
    payload["actor_reactor_mapping"] = _to_obj_array([m.get("actor_reactor_mapping", "") for m in meta_list])
    payload["sampling"] = _to_obj_array([m.get("sampling", "") for m in meta_list])
    payload["split"] = _to_obj_array([m.get("split", "") for m in meta_list])
    payload["dataset_name"] = _to_obj_array([m.get("dataset_name", "") for m in meta_list])
    payload["data_path"] = _to_obj_array([m.get("data_path", "") for m in meta_list])

    frame_ix_list = [m.get("frame_ix") for m in meta_list]
    valid_ix = [ix for ix in frame_ix_list if ix is not None]
    if valid_ix:
        max_len = max(len(ix) for ix in valid_ix)
        frame_ix = np.full((len(meta_list), max_len), -1, dtype=np.int64)
        frame_ix_len = np.zeros(len(meta_list), dtype=np.int64)
        for i, ix in enumerate(frame_ix_list):
            if ix is None:
                continue
            arr = np.asarray(ix, dtype=np.int64)
            frame_ix_len[i] = len(arr)
            frame_ix[i, : len(arr)] = arr
        payload["frame_ix"] = frame_ix
        payload["frame_ix_len"] = frame_ix_len
    else:
        payload["frame_ix"] = np.empty((len(meta_list), 0), dtype=np.int64)
        payload["frame_ix_len"] = np.zeros(len(meta_list), dtype=np.int64)

    return payload


def main():
    args = cgenerate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 150 if args.dataset in ['chi3d', 'interx'] else 60
    n_frames = min(max_frames, int(args.motion_length))
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist()
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames, args.num_person, args.data_path, args.pose_rep)
    total_num_samples = args.num_samples * args.num_repetitions
    order_dict = _load_interaction_order(args.interaction_order)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    model_window_size = None
    if getattr(model, 'arch', None) == 'mlp':
        try:
            model_window_size = int(model.mlp.motion_mlp.mlps[0].fc0.in_channels)
        except Exception:
            model_window_size = None

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            # t2m
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            # a2m
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text, 
                            inp=data.dataset._get_item_cmotion_index(one_action, mode='random')['inp'].cuda()) for
                            arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = ccollate(collate_args) # 'mask', 'lengths', 'tokens', 'action', 'action_text'

    all_outputs = []
    all_cmotions = []
    all_motions = []
    all_lengths = []
    all_text = []
    all_map = []
    all_meta = []

    if args.reaction_mode == 'online' and args.online_strategy == 'sliding_window':
        if model_window_size is not None and args.window_size > model_window_size:
            raise ValueError('window_size must be <= model MLP sequence length')

    time_all = 0.0
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        action = data.dataset.action_name_to_action(action_text)
        collate_args_with_inp = []
        rep_map = []
        for action_i, (arg, one_action, one_action_text) in enumerate(zip(collate_args, action, action_text)):
            cmotion_item = data.dataset._get_item_cmotion_index(
                one_action, mode='appointed', data_index=rep_i
            )
            collate_args_with_inp.append(
                dict(
                    arg,
                    action=one_action,
                    action_text=one_action_text,
                    inp=cmotion_item['inp'].cuda(),
                )
            )
            if 'data_index' in cmotion_item and 'data_key' in cmotion_item:
                rep_map.append(
                    (
                        rep_i,
                        action_i,
                        one_action_text,
                        int(one_action),
                        cmotion_item['data_index'],
                        cmotion_item['data_key'],
                    )
                )
            frame_ix = cmotion_item.get("frame_ix", None)
            if frame_ix is not None:
                frame_ix = np.asarray(frame_ix, dtype=np.int64)
            start_frame = int(frame_ix[0]) if frame_ix is not None and len(frame_ix) > 0 else -1
            end_frame = int(frame_ix[-1]) if frame_ix is not None and len(frame_ix) > 0 else -1
            actor_is_p1 = _resolve_actor_is_p1(order_dict, cmotion_item.get("data_key", ""))
            if actor_is_p1 == 1:
                mapping_str = "actor=P1,reactor=P2"
            elif actor_is_p1 == 0:
                mapping_str = "actor=P2,reactor=P1"
            else:
                mapping_str = "unknown"
            downsample = 4 if args.dataset == "interx" else 1
            all_meta.append(
                {
                    "rep_i": int(rep_i),
                    "action_i": int(action_i),
                    "action_id": int(one_action),
                    "action_name": one_action_text,
                    "data_index": int(cmotion_item.get("data_index", -1)),
                    "dataset_key": cmotion_item.get("data_key", ""),
                    "actor_reactor_mapping": mapping_str,
                    "length": int(cmotion_item.get("sampled_num_frames", n_frames)),
                    "raw_nframes": int(cmotion_item.get("raw_nframes", -1)),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frame_ix": frame_ix,
                    "sampling": cmotion_item.get("sampling", ""),
                    "sampling_step": int(cmotion_item.get("sampling_step", -1)),
                    "num_frames": int(n_frames),
                    "motion_length": int(args.motion_length),
                    "split": getattr(data.dataset, "split", ""),
                    "dataset_name": args.dataset,
                    "data_path": args.data_path,
                    "actor_is_p1": int(actor_is_p1),
                    "downsample": int(downsample),
                }
            )
        collate_args = collate_args_with_inp
        if rep_map:
            all_map.extend(rep_map)
        _, model_kwargs = ccollate(collate_args) # 'mask', 'lengths', 'tokens', 'action', 'action_text'

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        
        t_start = time.time()
        if args.reaction_mode == 'online' and args.online_strategy == 'sliding_window':
            sample, _ = sliding_window_sample(
                model,
                diffusion,
                model_kwargs,
                window_size=args.window_size,
                window_stride=args.window_stride,
                window_emit=args.window_emit,
                pad_mode=args.window_pad_mode,
                overlap_handling=args.window_overlap_handling,
                sample_fn=sample_fn,
                model_window_size=model_window_size,
            )
        elif args.reaction_mode == 'online' and args.online_strategy == 'autoregressive':
            cmotion_bak = model_kwargs['y']['cmotion']
            B, V, C, T = cmotion_bak.shape
            cmotion = torch.zeros_like(cmotion_bak)
            output = torch.zeros((B, V, C, T), device=cmotion_bak.device)
            for frame_idx in range(T):
                cmotion[:, :, :, frame_idx] = cmotion_bak[:, :, :, frame_idx]
                model_kwargs['y']['cmotion'] = cmotion
                sample = sample_fn(
                    model,
                    (args.batch_size, model.njoints, model.nfeats, n_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )
                output[:, :, :, frame_idx] = sample[:, :, :, frame_idx]
            sample = output
        else:
            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
        t_end = time.time()
        if rep_i >= 1:
            time_all += (t_end - t_start)*1000
        print(sample.shape)
        print('Generating time consumption: %s ms' % ((t_end - t_start)*1000))

        sample_gf = gaussian_filter1d(sample.cpu().numpy(), sigma=1, axis=-1)
        sample = torch.from_numpy(sample_gf).to(sample.device)
        all_outputs.append(sample_gf)
        all_cmotions.append(model_kwargs['y']['cmotion'].cpu().numpy())

        rot2xyz_pose_rep = 'xyz' if model.data_rep == 'xyz' else model.data_rep
        rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                               jointstype=args.body_model, vertstrans=True, num_person=1, betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)

        text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
        all_text += model_kwargs['y'][text_key]

        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

    if args.num_repetitions != 1:
        print('Average Time Consumption: %s ms' % (time_all / (args.num_repetitions-1)))

    all_motions = np.concatenate(all_motions, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_cmotions = np.concatenate(all_cmotions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_outputs = all_outputs[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_cmotions = all_cmotions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_meta = all_meta[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'output': all_outputs, 'cmotion': all_cmotions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    if all_map:
        map_path = os.path.join(out_path, 'map.txt')
        with open(map_path, 'w') as fw:
            fw.write('output_index\trep_i\taction_i\taction_name\taction_id\tdata_index\tdataset_key\n')
            for output_index, (rep_i, action_i, action_name, action_id, data_index, data_key) in enumerate(all_map[:total_num_samples]):
                fw.write(f"{output_index}\t{rep_i}\t{action_i}\t{action_name}\t{action_id}\t{data_index}\t{data_key}\n")
        print(f"saving map file to [{map_path}]")

    meta_path = os.path.join(out_path, 'results_meta.npz')
    meta_payload = _build_meta_payload(all_meta)
    if meta_payload:
        np.savez_compressed(meta_path, **meta_payload)
        print(f"saving metadata file to [{meta_path}]")

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


def load_dataset(args, max_frames, n_frames, num_person, data_path, pose_rep):
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              num_person=num_person,
                              data_path=data_path,
                              pose_rep=pose_rep,
                              setting=args.setting,
                              split='test',
                              hml_mode='text_only')
    data.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
