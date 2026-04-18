import argparse
import json
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from data_loaders.get_data import get_dataset_loader
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.cnet.cnet_v5 import CNetV5
from utils.fixseed import fixseed


RESTORED_CACHE_FIELDS = (
    "dataset_key",
    "actor_is_p1",
    "reactor_is_p2",
    "processed_frame_ix",
    "raw_frame_ix",
    "processed_nframes",
    "raw_nframes",
    "processed_fps",
    "raw_fps",
    "downsample",
    "actor_betas",
    "reactor_betas",
    "actor_gender_id",
    "reactor_gender_id",
    "body_model_type",
    "num_betas",
    "ground_offset_y_actor",
    "ground_offset_y_reactor",
    "pair_base_trans",
    "loader_base_trans",
    "actor_raw_trans_clip",
    "reactor_raw_trans_clip",
    "actor_raw_root_orient_clip",
    "reactor_raw_root_orient_clip",
)


def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = args.diffusion_steps
    scale_beta = 1.0
    timestep_respacing = args.timestep_respacing
    if getattr(args, "use_ddim", False) and not timestep_respacing:
        timestep_respacing = "ddim5"
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            gd.ModelVarType.FIXED_LARGE
            if not args.sigma_small
            else gd.ModelVarType.FIXED_SMALL
        )
        if not learn_sigma
        else gd.ModelVarType.LEARNED_RANGE,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_orient=args.lambda_orient,
        lambda_body=args.lambda_body,
        lambda_transl=args.lambda_transl,
        data_rep=args.pose_rep,
        num_person=1,
        body_model=args.body_model,
        vel_threshold=args.vel_threshold,
    )


def load_model_args(model_path):
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.exists(args_path):
        return {}
    with open(args_path, "r") as f:
        return json.load(f)


def merge_args(base, override):
    for key, val in override.items():
        if val is not None:
            base[key] = val
    return base


def build_model(args, num_actions):
    if args.body_model == "smpl":
        njoints = 25
    else:
        njoints = 56
    nfeats = 6 if args.pose_rep == "rot6d" else 3
    model = CNetV5(
        modeltype="",
        njoints=njoints,
        nfeats=nfeats,
        num_actions=num_actions,
        translation=True,
        pose_rep=args.pose_rep,
        glob=True,
        glob_rot=True,
        num_frames=args.num_frames,
        latent_dim=args.latent_dim,
        ff_size=1024,
        num_layers=args.layers,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        data_rep=args.pose_rep,
        cond_mode="no_cond" if args.unconstrained else "action",
        cond_mask_prob=args.cond_mask_prob,
        arch=args.arch,
        cm_mode=args.cm_mode,
        body_model=args.body_model,
        wo_pos_emb=args.wo_pos_emb,
        clip_version="ViT-B/32",
        dataset=args.dataset,
    )
    return model


def load_model_weights(model, model_path):
    state_dict = torch.load(model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:3]} ...")
    if missing:
        print(f"[warn] missing keys: {missing[:3]} ...")


def to_device(cond, device):
    if "y" not in cond:
        return cond
    cond = {"y": {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond["y"].items()}}
    return cond


def _stack_cache_field(chunks, key):
    values = chunks.get(key, [])
    if not values:
        return None
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(values, axis=0)
    if isinstance(first, list):
        return np.array(sum(values, []), dtype=object)
    return np.asarray(values)


def _pad_array_list(values, pad_value):
    arrays = [np.asarray(v) for v in values]
    max_shape = []
    ndim = max(arr.ndim for arr in arrays)
    for dim in range(ndim):
        max_shape.append(max(arr.shape[dim] if dim < arr.ndim else 1 for arr in arrays))
    out = np.full((len(arrays),) + tuple(max_shape), pad_value, dtype=arrays[0].dtype)
    for i, arr in enumerate(arrays):
        slices = (i,) + tuple(slice(0, s) for s in arr.shape)
        out[slices] = arr
    return out


def _write_h5_dataset(h5_file, key, value):
    if value is None:
        return
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O"}:
        flat = arr.reshape(-1).tolist()
        flat = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in flat]
        dt = h5py.string_dtype(encoding="utf-8")
        arr = np.asarray(flat, dtype=dt).reshape(arr.shape)
        h5_file.create_dataset(key, data=arr, dtype=dt)
    else:
        h5_file.create_dataset(key, data=arr)


def save_cache(output_path, actor_motion, reactor_gt, reactor_coarse, lengths, sample_indices, extra_fields=None):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "actor_motion": actor_motion,
        "reactor_gt": reactor_gt,
        "reactor_coarse": reactor_coarse,
        "lengths": lengths,
        "sample_indices": sample_indices,
    }
    if extra_fields:
        payload.update({k: v for k, v in extra_fields.items() if v is not None})
    if output_path.endswith(".h5"):
        with h5py.File(output_path, "w") as f:
            for key, value in payload.items():
                _write_h5_dataset(f, key, value)
        return
    np.savez_compressed(output_path, **payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--dataset", default="", type=str)
    parser.add_argument("--num_frames", default=-1, type=int)
    parser.add_argument("--batch_size", default=-1, type=int)
    parser.add_argument("--num_person", default=-1, type=int)
    parser.add_argument("--pose_rep", default="", type=str)
    parser.add_argument("--body_model", default="", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--max_batches", default=-1, type=int)
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--restoration_meta_path", default="", type=str)
    parser.add_argument("--raw_motions_root", default="", type=str)
    parser.add_argument("--interaction_order_path", default="", type=str)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--timestep_respacing", default="", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args_cli = parser.parse_args()

    model_args = load_model_args(args_cli.model_path)
    overrides = {
        "data_path": args_cli.data_path or None,
        "dataset": args_cli.dataset or None,
        "num_frames": args_cli.num_frames if args_cli.num_frames > 0 else None,
        "batch_size": args_cli.batch_size if args_cli.batch_size > 0 else None,
        "num_person": args_cli.num_person if args_cli.num_person > 0 else None,
        "pose_rep": args_cli.pose_rep or None,
        "body_model": args_cli.body_model or None,
        "timestep_respacing": args_cli.timestep_respacing or None,
    }
    merged = merge_args(model_args, overrides)
    merged.setdefault("setting", "cnet_v5")
    merged.setdefault("use_ddim", False)
    if args_cli.use_ddim:
        merged["use_ddim"] = True
    merged.setdefault("timestep_respacing", "")

    args = argparse.Namespace(**merged)
    args.output_path = args_cli.output_path
    args.split = args_cli.split
    args.max_batches = args_cli.max_batches
    args.num_samples = args_cli.num_samples
    args.restoration_meta_path = args_cli.restoration_meta_path
    args.raw_motions_root = args_cli.raw_motions_root
    args.interaction_order_path = args_cli.interaction_order_path
    args.seed = args_cli.seed
    args.device = args_cli.device

    fixseed(args.seed)

    print("Loading dataset...")
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_person=args.num_person,
        data_path=args.data_path,
        pose_rep=args.pose_rep,
        body_model=args.body_model,
        setting=args.setting,
        split=args.split,
        restoration_meta_path=args.restoration_meta_path,
        raw_motions_root=args.raw_motions_root,
        interaction_order_path=args.interaction_order_path,
    )

    num_actions = getattr(data.dataset, "num_actions", 1)
    model = build_model(args, num_actions=num_actions)
    diffusion = create_gaussian_diffusion(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    load_model_weights(model, args_cli.model_path)

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    actor_list = []
    gt_list = []
    coarse_list = []
    lengths_list = []
    indices_list = []
    extra_chunks = {key: [] for key in RESTORED_CACHE_FIELDS}

    total = 0
    total_batches = len(data)
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)
    with torch.no_grad():
        pbar = tqdm(
            enumerate(data),
            total=total_batches,
            desc="Generate coarse cache",
        )
        for batch_idx, (motion, cond) in pbar:
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if args.num_samples > 0 and total >= args.num_samples:
                break

            motion = motion.to(device)
            cond = to_device(cond, device)

            sample = sample_fn(
                model,
                motion.shape,
                clip_denoised=False,
                model_kwargs=cond,
                progress=False,
                noise=None,
            )

            actor = cond["y"]["cmotion"]
            lengths = cond["y"]["lengths"]
            bs = motion.shape[0]
            keep = bs
            if args.num_samples > 0 and total + bs > args.num_samples:
                keep = args.num_samples - total

            actor_list.append(actor[:keep].cpu().numpy())
            gt_list.append(motion[:keep].cpu().numpy())
            coarse_list.append(sample[:keep].cpu().numpy())
            lengths_list.append(lengths[:keep].cpu().numpy())
            indices_list.append(np.arange(total, total + keep, dtype=np.int64))
            for key in RESTORED_CACHE_FIELDS:
                if key not in cond["y"]:
                    continue
                value = cond["y"][key]
                if torch.is_tensor(value):
                    value = value[:keep].detach().cpu().numpy()
                elif isinstance(value, list):
                    value = value[:keep]
                    if value and isinstance(value[0], np.ndarray):
                        shapes = [tuple(np.asarray(v).shape) for v in value]
                        if len(set(shapes)) == 1:
                            value = np.stack(value, axis=0)
                else:
                    value = np.asarray(value)[:keep]
                extra_chunks[key].append(value)
            total += keep
            pbar.set_postfix(samples=total)

    actor_motion = np.concatenate(actor_list, axis=0).astype(np.float32)
    reactor_gt = np.concatenate(gt_list, axis=0).astype(np.float32)
    reactor_coarse = np.concatenate(coarse_list, axis=0).astype(np.float32)
    lengths = np.concatenate(lengths_list, axis=0).astype(np.int64)
    sample_indices = np.concatenate(indices_list, axis=0).astype(np.int64)
    extra_fields = {}
    for key in RESTORED_CACHE_FIELDS:
        chunks = extra_chunks[key]
        if not chunks:
            continue
        first = chunks[0]
        if isinstance(first, list):
            flat = sum([list(x) for x in chunks], [])
            if flat and isinstance(flat[0], np.ndarray):
                pad_value = -1 if "frame_ix" in key else 0.0
                value = _pad_array_list(flat, pad_value=pad_value)
            else:
                value = np.array(flat, dtype=object)
        else:
            value = _stack_cache_field(extra_chunks, key)
        if value is not None:
            extra_fields[key] = value

    print(f"Saving cache to {args.output_path} (samples={len(sample_indices)})")
    save_cache(
        args.output_path,
        actor_motion=actor_motion,
        reactor_gt=reactor_gt,
        reactor_coarse=reactor_coarse,
        lengths=lengths,
        sample_indices=sample_indices,
        extra_fields=extra_fields,
    )


if __name__ == "__main__":
    main()
