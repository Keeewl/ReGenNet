import copy
import torch
from tqdm import tqdm
import functools
import numpy as np
import os
from utils.fixseed import fixseed

from eval.a2m.stgcn.evaluate import Evaluation as STGCNEvaluation
from eval.metrics_contact import contact_distance
from torch.utils.data import DataLoader
from data_loaders.tensors import collate, ccollate

from .tools import format_metrics
import utils.rotation_conversions as geometry
from utils import dist_util
from utils.online_window import sliding_window_sample

def _load_interx_action_names(data_path):
    candidates = []
    if data_path:
        abs_path = os.path.abspath(data_path)
        dataset_dir = os.path.dirname(os.path.dirname(abs_path))
        candidates.append(os.path.join(dataset_dir, "annots", "action_setting.txt"))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates.append(os.path.join(repo_root, "dataset", "interx", "annots", "action_setting.txt"))

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    return []

def convert_x_to_rot6d(x, pose_rep):
    # convert rotation to rot6d
    if pose_rep == "rotvec":
        x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
    elif pose_rep == "rotmat":
        x = x.reshape(*x.shape[:-1], 3, 3)
        x = geometry.matrix_to_rotation_6d(x)
    elif pose_rep == "rotquat":
        x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
    elif pose_rep == "rot6d":
        x = x
    else:
        raise NotImplementedError("No geometry for this one.")
    return x



def split_actor_reactor_xyz(output_xyz):
    """
    output_xyz: [B, J, C, T] with C in {3, 6}
    returns actor_xyz/reactor_xyz: [B, J, 3, T]
    """
    if output_xyz.shape[2] == 6:
        actor_xyz = output_xyz[:, :, :3, :]
        reactor_xyz = output_xyz[:, :, 3:, :]
        return actor_xyz, reactor_xyz
    if output_xyz.shape[2] == 3 and output_xyz.shape[1] % 2 == 0:
        half = output_xyz.shape[1] // 2
        actor_xyz = output_xyz[:, :half, :, :]
        reactor_xyz = output_xyz[:, half:, :, :]
        return actor_xyz, reactor_xyz
    raise ValueError(f"Unexpected output_xyz shape: {tuple(output_xyz.shape)}")


def evaluate_cd_gen(gen_loader, gt_loader, tau_contact=0.1):
    """
    Count-weighted CD for Stage1 generation.
    """
    sum_cd = None
    sum_count = None

    for gen_batch, gt_batch in zip(gen_loader, gt_loader):
        gen_xyz = gen_batch["output_xyz"]
        gt_xyz = gt_batch["output_xyz"]
        lengths = gt_batch["lengths"]

        actor_gt, reactor_gt = split_actor_reactor_xyz(gt_xyz)
        _, reactor_gen = split_actor_reactor_xyz(gen_xyz)

        metrics = contact_distance(
            actor_gt,
            reactor_gen,
            reactor_gt,
            list(range(actor_gt.shape[1])),
            list(range(reactor_gt.shape[1])),
            tau_contact=tau_contact,
            lengths=lengths,
            active_mask=None,
        )

        if sum_cd is None:
            sum_cd = torch.zeros((), device=metrics["cd"].device)
            sum_count = torch.zeros((), device=metrics["cd"].device)

        sum_cd += metrics["cd"] * metrics["count"]
        sum_count += metrics["count"]

    if sum_cd is None:
        return 0.0
    denom = sum_count.clamp(min=1.0)
    return (sum_cd / denom).item()


def evaluate_cd_gt(gt_loader, tau_contact=0.1):
    sum_cd = None
    sum_count = None

    for gt_batch in gt_loader:
        gt_xyz = gt_batch["output_xyz"]
        lengths = gt_batch["lengths"]

        actor_gt, reactor_gt = split_actor_reactor_xyz(gt_xyz)

        metrics = contact_distance(
            actor_gt,
            reactor_gt,
            reactor_gt,
            list(range(actor_gt.shape[1])),
            list(range(reactor_gt.shape[1])),
            tau_contact=tau_contact,
            lengths=lengths,
            active_mask=None,
        )

        if sum_cd is None:
            sum_cd = torch.zeros((), device=metrics["cd"].device)
            sum_count = torch.zeros((), device=metrics["cd"].device)

        sum_cd += metrics["cd"] * metrics["count"]
        sum_count += metrics["count"]

    if sum_cd is None:
        return 0.0
    denom = sum_count.clamp(min=1.0)
    return (sum_cd / denom).item()


class NewDataloader:
    def __init__(self, mode, model, diffusion, dataiterator, device, dataset, num_samples, num_person, body_model, setting, reaction_mode="offline", online_strategy="sliding_window", window_size=30, window_stride=10, window_emit="stride", window_pad_mode="edge", window_overlap_handling="latest", auto_regressive=False, use_ddim=False):
        assert mode in ["gen", "gt"]

        self.batches = []
        sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
        model_window_size = None
        if getattr(model, 'arch', None) == 'mlp':
            try:
                model_window_size = int(model.mlp.motion_mlp.mlps[0].fc0.in_channels)
            except Exception:
                model_window_size = None

        with torch.no_grad():
            for motions, model_kwargs in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                motions = motions.to(device)
                if num_samples != -1 and len(self.batches) * dataiterator.batch_size > num_samples:
                    continue  # do not break because it confuses the multiple loaders
                batch = dict()
                if mode == "gen":
                    for _k in model_kwargs['y'].keys():
                        if type(model_kwargs['y'][_k]) == torch.Tensor:
                            model_kwargs['y'][_k] = model_kwargs['y'][_k].to(device)
                    if reaction_mode == "online" and online_strategy == "sliding_window":
                        sample, _ = sliding_window_sample(
                            model,
                            diffusion,
                            model_kwargs,
                            window_size=window_size,
                            window_stride=window_stride,
                            window_emit=window_emit,
                            pad_mode=window_pad_mode,
                            overlap_handling=window_overlap_handling,
                            sample_fn=sample_fn,
                            model_window_size=model_window_size,
                        )
                        if setting == 'cmdm' or 'cmotion' in model_kwargs['y']:
                            batch['output'] = torch.cat((model_kwargs['y']['cmotion'], sample), axis=2)
                        else:
                            batch['output'] = sample
                    elif auto_regressive or (reaction_mode == "online" and online_strategy == "autoregressive"):
                        cmotion_bak = model_kwargs['y']['cmotion']
                        B, V, C, T = cmotion_bak.shape
                        cmotion = torch.zeros_like(model_kwargs['y']['cmotion']).to(device)
                        if setting == 'cmdm' or 'cmotion' in model_kwargs['y']:
                            output = torch.zeros((B, V, C*2, T)).to(device)
                        else:
                            output = torch.zeros((B, V, C, T)).to(device)
                        for frame_idx in range(cmotion.shape[-1]):
                            cmotion[:,:,:,frame_idx] = cmotion_bak[:,:,:,frame_idx]
                            model_kwargs['y']['cmotion'] = cmotion
                            sample = sample_fn(model, motions.shape, clip_denoised=False, model_kwargs=model_kwargs)
                            if setting == 'cmdm' or 'cmotion' in model_kwargs['y']:
                                tmp = torch.cat((model_kwargs['y']['cmotion'], sample), axis=2)
                            else:
                                tmp = sample
                            output[:,:,:,frame_idx] = tmp[:,:,:,frame_idx]
                        batch['output'] = output
                    else:
                        sample = sample_fn(model, motions.shape, clip_denoised=False, model_kwargs=model_kwargs)
                        if setting == 'cmdm' or 'cmotion' in model_kwargs['y']:
                            batch['output'] = torch.cat((model_kwargs['y']['cmotion'], sample), axis=2)
                        else:
                            batch['output'] = sample
                    batch['text'] = model_kwargs['y']['action_text']
                elif mode == "gt":
                    batch['output'] = motions

                max_n_frames = model_kwargs['y']['lengths'].max()
                mask = model_kwargs['y']['mask'].reshape(dataiterator.batch_size, max_n_frames).bool()

                batch["output_xyz"] = model.rot2xyz(x=batch["output"], mask=mask, pose_rep='rot6d', glob=True,
                                                    translation=True, jointstype=body_model, vertstrans=True, betas=None,
                                                    beta=0, glob_rot=None, get_rotations_back=False, num_person=num_person)
                ### Modification by Derek, 2023.03.23, for multi-person, root translations matter
                # if model.translation:
                #     # the stgcn model expects rotations only
                #     batch["output"] = batch["output"][:, :-1]

                batch["lengths"] = model_kwargs['y']['lengths'].to(device)
                # using torch.long so lengths/action will be used as indices
                batch["y"] = model_kwargs['y']['action'].squeeze().long().cpu()  # using torch.long so lengths/action will be used as indices
                self.batches.append(batch)

            num_samples_last_batch = num_samples % dataiterator.batch_size
            if num_samples_last_batch > 0:
                for k, v in self.batches[-1].items():
                    self.batches[-1][k] = v[:num_samples_last_batch]
            # if mode == 'gen':
            #     split = dataiterator.dataset.split
            #     outputs = []
            #     cmotions = []
            #     texts = []
            #     for idx in range(len(self.batches)):
            #         outputs.append(self.batches[idx]['output'][:,:,6:12,:].cpu())
            #         cmotions.append(self.batches[idx]['output'][:,:,0:6,:].cpu())
            #         texts.append(self.batches[idx]['text'])
            #     outputs = np.concatenate(outputs, axis=0)
            #     cmotions = np.concatenate(cmotions, axis=0)
            #     texts = np.concatenate(texts, axis=0)
            #     if not os.path.exists('vis_data'):
            #         os.makedirs('./vis_data')
            #     filename = dataiterator.dataset.data_path.split('/')[-1]
            #     np.save(os.path.join('./vis_data', '{}_split_{}_{}.npy'.format(dataset, split, filename)), {'cmotion': cmotions, 'output': outputs, 'text': texts})


    def __iter__(self):
        return iter(self.batches)


def evaluate(args, model, diffusion, data, rec_model_path, setting, acc_only, auto_regressive=False):
    torch.multiprocessing.set_sharing_strategy('file_system')

    bs = args.batch_size
    if args.dataset == 'chi3d':
        args.num_classes = 8
        args.nfeats = 6
    elif args.dataset == 'interx':
        action_names = _load_interx_action_names(args.data_path)
        if not action_names:
            raise ValueError("InterX action_setting.txt not found or empty.")
        args.num_classes = len(action_names)
        args.nfeats = 6
    else:
        raise NotImplementedError("This dataset is not supported.")
    args.model_path = rec_model_path
    
    device = dist_util.dev()

    recogparameters = args.__dict__.copy()
    recogparameters["pose_rep"] = args.pose_rep
    recogparameters["nfeats"] = args.nfeats * 2
    recogparameters["model_path"] = args.model_path
    recogparameters["num_person"] = 2 # for cmdm, also 2

    stgcnevaluation = STGCNEvaluation(args.dataset, args.body_model, recogparameters, device)

    stgcn_metrics = {}

    data_types = ['train', 'test']
    datasetGT = {'train': [data], 'test': [copy.deepcopy(data)]}

    for key in data_types:
        datasetGT[key][0].split = key

    compute_gt_gt = False #False
    if compute_gt_gt:
        for key in data_types:
            datasetGT[key].append(copy.deepcopy(datasetGT[key][0]))

    model.eval()

    allseeds = list(range(args.num_seeds))

    for index, seed in enumerate(allseeds):
        print(f"Evaluation number: {index + 1}/{args.num_seeds}")
        fixseed(seed)
        for key in data_types:
            for data in datasetGT[key]:
                data.reset_shuffle()
                data.shuffle()

        dataiterator = {key: [DataLoader(data, batch_size=bs, shuffle=False, num_workers=8, drop_last=True, collate_fn=collate)
                            for data in datasetGT[key]]
                        for key in data_types}
        dataiterator_con = {key: [DataLoader(data, batch_size=bs, shuffle=False, num_workers=8, drop_last=True, collate_fn=ccollate)
                            for data in datasetGT[key]]
                        for key in data_types}

        new_data_loader = functools.partial(NewDataloader, model=model, diffusion=diffusion, device=device,
                                            dataset=args.dataset, num_samples=args.num_samples, num_person=2, body_model=args.body_model, setting=setting,
                                            reaction_mode=args.reaction_mode, online_strategy=args.online_strategy, window_size=args.window_size, window_stride=args.window_stride,
                                            window_emit=args.window_emit, window_pad_mode=args.window_pad_mode, window_overlap_handling=args.window_overlap_handling,
                                            auto_regressive=auto_regressive, use_ddim=args.use_ddim)
        gtLoaders = {key: new_data_loader(mode="gt", dataiterator=dataiterator[key][0])
                     for key in ["train", "test"]}

        if compute_gt_gt:
            gtLoaders2 = {key: new_data_loader(mode="gt", dataiterator=dataiterator[key][0])
                          for key in ["train", "test"]}

        if setting in ['cmdm', 'cnet_v1', 'cnet_v2', 'cnet_v3', 'cnet_v4', 'cnet_v5', 'cnet_v5_actor_bodyhand', 'cnet_v5_actor_globalonly']:
            genLoaders = {key: new_data_loader(mode="gen", dataiterator=dataiterator_con[key][0])
                        for key in ["train", "test"]}
        elif setting == 'mdm':
            genLoaders = {key: new_data_loader(mode="gen", dataiterator=dataiterator[key][0])
                        for key in ["train", "test"]}
        else:
            raise NotImplementedError(f"Unsupported evaluation setting: {setting}")

        loaders = {"gen": genLoaders,
                   "gt": gtLoaders}

        if compute_gt_gt:
            loaders["gt2"] = gtLoaders2

        if not acc_only:
            stgcn_metrics[seed] = stgcnevaluation.evaluate(model, loaders, setting)
        else:
            stgcn_metrics[seed] = stgcnevaluation.evaluate_acc(model, loaders, setting)
        for split in data_types:
            cd_value = evaluate_cd_gen(genLoaders[split], gtLoaders[split], tau_contact=0.1)
            stgcn_metrics[seed][f"cd_gen_{split}"] = cd_value
            cd_gt_value = evaluate_cd_gt(gtLoaders[split], tau_contact=0.1)
            stgcn_metrics[seed][f"cd_gt_{split}"] = cd_gt_value
        del loaders

    metrics = {"feats": {key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds] for key in stgcn_metrics[allseeds[0]]}}

    return metrics
