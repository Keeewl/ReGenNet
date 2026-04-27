"""Build Stage2-lite reaction_data from a frozen Stage1 model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_loaders.get_data import get_collate_fn, get_dataset
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.cfg_sampler import ClassifierFreeSampleModel
from refine.data.restored_space import (
    REQUIRED_RESTORATION_METADATA_FIELDS,
    RESTORED_PAIR_SPACE,
    extract_restoration_metadata,
    restore_pair_batch,
)
from refine.data.schema import OPTIONAL_REACTION_DATA_FIELDS
from utils.fixseed import fixseed


def _load_model_args(model_path):
    args_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "args.json")
    if not os.path.exists(args_path):
        return {}
    with open(args_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_override(base, key, value):
    if value is not None:
        base[key] = value


def _resolve_data_path(dataset, split, explicit_path):
    if explicit_path:
        return explicit_path
    candidates = [
        os.path.join("dataset", dataset, "regen", f"{split}.h5"),
        os.path.join("dataset", dataset, "motions", f"{split}.h5"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return explicit_path or ""


def _build_args(args_cli):
    model_args = _load_model_args(args_cli.model_path)
    merged = dict(model_args)
    _maybe_override(merged, "dataset", args_cli.dataset)
    resolved_data_path = _resolve_data_path(
        args_cli.dataset or model_args.get("dataset", ""),
        args_cli.split,
        args_cli.data_path,
    )
    if resolved_data_path:
        merged["data_path"] = resolved_data_path
    _maybe_override(merged, "num_frames", args_cli.num_frames)
    _maybe_override(merged, "batch_size", args_cli.batch_size)
    _maybe_override(merged, "num_person", args_cli.num_person)
    _maybe_override(merged, "pose_rep", args_cli.pose_rep)
    _maybe_override(merged, "body_model", args_cli.body_model)
    _maybe_override(merged, "setting", args_cli.setting)
    _maybe_override(merged, "arch", args_cli.arch)
    _maybe_override(merged, "latent_dim", args_cli.latent_dim)
    _maybe_override(merged, "layers", args_cli.layers)
    _maybe_override(merged, "timestep_respacing", args_cli.timestep_respacing)
    if args_cli.guidance_param is not None:
        merged["guidance_param"] = args_cli.guidance_param
    if args_cli.use_ddim:
        merged["use_ddim"] = True

    merged.setdefault("dataset", "interx")
    merged.setdefault("setting", "cnet_v5")
    merged.setdefault("arch", "online")
    merged.setdefault("cm_mode", "concat")
    merged.setdefault("emb_trans_dec", False)
    merged.setdefault("wo_pos_emb", False)
    merged.setdefault("num_frames", 150)
    merged.setdefault("batch_size", 8)
    merged.setdefault("num_person", 2)
    merged.setdefault("pose_rep", "rot6d")
    merged.setdefault("body_model", "smplx")
    merged.setdefault("latent_dim", 512)
    merged.setdefault("layers", 8)
    merged.setdefault("guidance_param", 1.0)
    merged.setdefault("use_ddim", False)
    merged.setdefault("timestep_respacing", "")
    merged.setdefault("split", args_cli.split)
    merged.setdefault("cond_mask_prob", 0.0)
    merged.setdefault("unconstrained", True)
    merged.setdefault("noise_schedule", "cosine")
    merged.setdefault("diffusion_steps", 1000)
    merged.setdefault("sigma_small", True)
    merged.setdefault("lambda_vel", 0.0)
    merged.setdefault("lambda_rcxyz", 0.0)
    merged.setdefault("lambda_fc", 0.0)
    merged.setdefault("lambda_orient", 1.0)
    merged.setdefault("lambda_body", 1.0)
    merged.setdefault("lambda_transl", 1.0)
    merged.setdefault("vel_threshold", 0.01)
    return argparse.Namespace(**merged)


def _ensure_clip_stub():
    try:
        import clip as _clip  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    def _missing(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "The optional 'clip' package is not installed. "
            "This Stage1 checkpoint can only be sampled here if it does not "
            "instantiate CLIP-conditioned text modules."
        )

    clip_stub = types.ModuleType("clip")
    clip_stub.load = _missing
    clip_stub.tokenize = _missing
    clip_stub.model = types.SimpleNamespace(convert_weights=lambda *_args, **_kwargs: None)
    sys.modules["clip"] = clip_stub


def _ensure_einops_stub():
    try:
        import einops as _einops  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    einops_stub = types.ModuleType("einops")
    einops_stub.rearrange = lambda x, *_args, **_kwargs: x
    sys.modules["einops"] = einops_stub


def _ensure_timm_stub():
    try:
        import timm as _timm  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import torch.nn as nn

    layers_stub = types.ModuleType("timm.models.layers")

    class _DropPath(nn.Identity):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

    layers_stub.DropPath = _DropPath
    models_stub = types.ModuleType("timm.models")
    models_stub.layers = layers_stub
    timm_stub = types.ModuleType("timm")
    timm_stub.models = models_stub
    sys.modules["timm"] = timm_stub
    sys.modules["timm.models"] = models_stub
    sys.modules["timm.models.layers"] = layers_stub


def _load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys[:8]}")
    if not all(key.startswith("clip_model.") for key in missing_keys):
        raise RuntimeError(f"Unexpected missing checkpoint keys: {missing_keys[:8]}")


def _get_model_ctor(setting):
    if setting == "cmdm":
        from model.cmdm import CMDM

        return CMDM
    if setting == "cnet_v1":
        from model.cnet.cnet_v1 import CNetV1

        return CNetV1
    if setting == "cnet_v2":
        from model.cnet.cnet_v2 import CNetV2

        return CNetV2
    if setting == "cnet_v3":
        from model.cnet.cnet_v3 import CNetV3

        return CNetV3
    if setting == "cnet_v4":
        from model.cnet.cnet_v4 import CNetV4

        return CNetV4
    if setting == "cnet_v5":
        from model.cnet.cnet_v5 import CNetV5

        return CNetV5
    if setting == "cnet_v5_actor_bodyhand":
        from model.ablation.cnet_v5_actor_bodyhand import CNetV5ActorBodyHand

        return CNetV5ActorBodyHand
    raise ValueError(f"Unsupported Stage1 setting for reaction_data build: {setting}")


def _create_gaussian_diffusion(args):
    predict_xstart = True
    steps = int(getattr(args, "diffusion_steps", 1000))
    scale_beta = 1.0
    timestep_respacing = getattr(args, "timestep_respacing", "")
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
        num_person=args.num_person,
        body_model=args.body_model,
        vel_threshold=args.vel_threshold,
    )


def _get_stage1_model_kwargs(args, dataset):
    if args.body_model == "smpl":
        njoints = 25
    elif args.body_model == "smplx":
        njoints = 56
    else:
        raise ValueError(f"Unsupported body_model for Stage1 sampling: {args.body_model}")

    if args.pose_rep == "rot6d":
        nfeats = 6
    elif args.pose_rep == "xyz":
        nfeats = 3
    else:
        raise ValueError(f"Unsupported pose_rep for Stage1 sampling: {args.pose_rep}")

    cond_mode = "no_cond" if getattr(args, "unconstrained", False) else "action"
    num_actions = getattr(dataset, "num_actions", 1)
    num_person = getattr(dataset, "num_person", args.num_person)

    return {
        "modeltype": "",
        "njoints": njoints,
        "nfeats": nfeats,
        "num_actions": num_actions,
        "num_person": num_person,
        "num_frames": args.num_frames,
        "translation": True,
        "pose_rep": args.pose_rep,
        "glob": True,
        "glob_rot": True,
        "latent_dim": args.latent_dim,
        "ff_size": 1024,
        "num_layers": args.layers,
        "num_heads": 4,
        "dropout": 0.1,
        "activation": "gelu",
        "data_rep": args.pose_rep,
        "cond_mode": cond_mode,
        "cond_mask_prob": args.cond_mask_prob,
        "action_emb": "tensor",
        "arch": args.arch,
        "cm_mode": args.cm_mode,
        "body_model": args.body_model,
        "wo_pos_emb": args.wo_pos_emb,
        "emb_trans_dec": args.emb_trans_dec,
        "clip_version": "ViT-B/32",
        "dataset": args.dataset,
        "diffusion_steps": args.diffusion_steps,
    }


def _create_stage1_model_and_diffusion(args, dataset):
    ctor = _get_model_ctor(args.setting)
    model = ctor(**_get_stage1_model_kwargs(args, dataset))
    if args.setting != "cmdm":
        args.num_person = 1
    diffusion = _create_gaussian_diffusion(args)
    return model, diffusion


def _resolve_output_path(output_path):
    output_path = os.path.abspath(output_path)
    root, ext = os.path.splitext(output_path)
    if ext.lower() in {".npz", ".h5"}:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path
    os.makedirs(output_path, exist_ok=True)
    return os.path.join(output_path, "reaction_data.npz")


def _slice_value(value, keep):
    if torch.is_tensor(value):
        return value[:keep].detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value[:keep]
    if isinstance(value, list):
        return value[:keep]
    return np.asarray(value)[:keep]


def _object_array(values):
    out = np.empty((len(values),), dtype=object)
    for idx, value in enumerate(values):
        out[idx] = value
    return out


def _pad_array_list(values, pad_value):
    arrays = [np.asarray(v) for v in values]
    ndim = max(arr.ndim for arr in arrays)
    max_shape = []
    for dim in range(ndim):
        max_shape.append(max(arr.shape[dim] if dim < arr.ndim else 1 for arr in arrays))
    out = np.full((len(arrays),) + tuple(max_shape), pad_value, dtype=arrays[0].dtype)
    for idx, arr in enumerate(arrays):
        slices = (idx,) + tuple(slice(0, size) for size in arr.shape)
        out[slices] = arr
    return out


def _finalize_chunks(chunks, key):
    if not chunks:
        return None
    if isinstance(chunks[0], list):
        flat = []
        for chunk in chunks:
            flat.extend(chunk)
        if not flat:
            return None
        first = flat[0]
        if isinstance(first, np.ndarray):
            if first.dtype.kind in {"U", "S", "O"}:
                return _object_array(flat)
            shapes = [tuple(arr.shape) for arr in flat]
            if len(set(shapes)) == 1:
                return np.stack(flat, axis=0)
            pad_value = -1 if "frame_ix" in key else 0.0
            return _pad_array_list(flat, pad_value=pad_value)
        if isinstance(first, (str, bytes)):
            return _object_array(flat)
        return np.asarray(flat)

    flat = []
    for chunk in chunks:
        flat.append(chunk)
    first = flat[0]
    if isinstance(first, np.ndarray):
        if first.dtype.kind in {"U", "S", "O"}:
            return np.concatenate(flat, axis=0)
        shapes = [tuple(arr.shape) for arr in flat]
        if len(set(shapes)) == 1:
            return np.concatenate(flat, axis=0)
        pad_value = -1 if "frame_ix" in key else 0.0
        return _pad_array_list(flat, pad_value=pad_value)
    if isinstance(first, (str, bytes)):
        return _object_array(flat)
    return np.asarray(flat)


def _maybe_collect_extra_fields(cond_y, keep, chunks):
    aliases = {
        "data_key": "dataset_key",
        "frame_ix": "processed_frame_ix",
    }
    for target_key in OPTIONAL_REACTION_DATA_FIELDS:
        if target_key not in OPTIONAL_REACTION_DATA_FIELDS or target_key == "space_definition":
            continue
        source_key = None
        if target_key in cond_y:
            source_key = target_key
        else:
            for alias_key, alias_target in aliases.items():
                if alias_target == target_key and alias_key in cond_y:
                    source_key = alias_key
                    break
        if source_key is None:
            continue
        chunks.setdefault(target_key, [])
        chunks[target_key].append(_slice_value(cond_y[source_key], keep))


def _can_extract_restoration_metadata(cond_y):
    return all(key in cond_y for key in REQUIRED_RESTORATION_METADATA_FIELDS)


def _build_dataloader(args, args_cli):
    enable_restoration = args_cli.enable_restoration_metadata
    if enable_restoration is None:
        enable_restoration = args.dataset == "interx"
    dataset = get_dataset(
        name=args.dataset,
        num_frames=args.num_frames,
        num_person=args.num_person,
        data_path=args.data_path,
        pose_rep=args.pose_rep,
        body_model=args.body_model,
        split=args_cli.split,
        enable_restoration_metadata=enable_restoration,
        restoration_meta_path=args_cli.restoration_meta_path,
        raw_motions_root=args_cli.raw_motions_root,
        interaction_order_path=args_cli.interaction_order_path,
    )
    collate_fn = get_collate_fn(args.dataset, args.setting)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args_cli.num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )


def _save_reaction_data(output_path, payload):
    np.savez_compressed(output_path, **payload)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--dataset", default=None, choices=["chi3d", "interx"], type=str)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], type=str)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--body_model", default=None, type=str)
    parser.add_argument("--pose_rep", default=None, type=str)
    parser.add_argument("--num_frames", default=None, type=int)
    parser.add_argument("--num_person", default=None, type=int)
    parser.add_argument("--setting", default=None, type=str)
    parser.add_argument("--arch", default=None, type=str)
    parser.add_argument("--latent_dim", default=None, type=int)
    parser.add_argument("--layers", default=None, type=int)
    parser.add_argument("--guidance_param", default=None, type=float)
    parser.add_argument("--timestep_respacing", default=None, type=str)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--enable_restoration_metadata", default=None, type=lambda x: str(x).lower() in {"1", "true", "yes"})
    parser.add_argument("--restoration_meta_path", default="", type=str)
    parser.add_argument("--raw_motions_root", default="", type=str)
    parser.add_argument("--interaction_order_path", default="", type=str)
    return parser.parse_args()


def main():
    args_cli = parse_args()
    _ensure_clip_stub()
    _ensure_einops_stub()
    _ensure_timm_stub()

    args = _build_args(args_cli)
    fixseed(args_cli.seed)
    output_path = _resolve_output_path(args_cli.output_path)

    if not args.data_path:
        raise FileNotFoundError(
            "Unable to resolve dataset data_path. Pass --data_path explicitly or place the "
            "dataset under dataset/<name>/{regen|motions}/<split>.h5."
        )

    loader = _build_dataloader(args, args_cli)
    model, diffusion = _create_stage1_model_and_diffusion(args, loader.dataset)

    state_dict = torch.load(args_cli.model_path, map_location="cpu")
    _load_model_wo_clip(model, state_dict)

    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if (
        float(getattr(args, "guidance_param", 1.0)) != 1.0
        and getattr(model, "cond_mode", "") in {"text", "action"}
        and float(getattr(model, "cond_mask_prob", 0.0)) > 0.0
    ):
        model = ClassifierFreeSampleModel(model)
    elif float(getattr(args, "guidance_param", 1.0)) != 1.0:
        print(
            "[warning] guidance_param was provided but the frozen Stage1 model "
            "does not expose classifier-free sampling in text/action mode. Ignoring it."
        )

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    actor_chunks = []
    gt_chunks = []
    coarse_chunks = []
    length_chunks = []
    sample_index_chunks = []
    extra_chunks = {}
    saved_in_restored_space = False
    warned_missing_restoration = False
    total = 0

    with torch.no_grad():
        iterator = tqdm(loader, desc="Build reaction_data")
        for motion, cond in iterator:
            if args_cli.num_samples > 0 and total >= args_cli.num_samples:
                break

            motion = motion.to(device)
            cond_y = {}
            for key, value in cond["y"].items():
                if torch.is_tensor(value):
                    cond_y[key] = value.to(device)
                else:
                    cond_y[key] = value
            cond = {"y": cond_y}

            if (
                isinstance(cond["y"].get("scale", None), torch.Tensor)
                and cond["y"]["scale"].shape[0] != motion.shape[0]
            ):
                cond["y"]["scale"] = cond["y"]["scale"][: motion.shape[0]]
            elif (
                float(getattr(args, "guidance_param", 1.0)) != 1.0
                and getattr(model, "cond_mode", "") in {"text", "action"}
            ):
                cond["y"]["scale"] = torch.full(
                    (motion.shape[0],),
                    float(args.guidance_param),
                    device=device,
                )

            sample = sample_fn(
                model,
                motion.shape,
                clip_denoised=False,
                model_kwargs=cond,
                progress=False,
                noise=None,
            )

            keep = motion.shape[0]
            if args_cli.num_samples > 0:
                keep = min(keep, args_cli.num_samples - total)

            actor_motion = cond["y"]["cmotion"][:keep]
            gt_motion = motion[:keep]
            coarse_motion = sample[:keep]

            if _can_extract_restoration_metadata(cond["y"]):
                meta = extract_restoration_metadata(cond["y"], device=device)
                actor_motion, gt_motion = restore_pair_batch(actor_motion, gt_motion, meta)
                _, coarse_motion = restore_pair_batch(actor_motion, coarse_motion, meta)
                saved_in_restored_space = True
            elif not warned_missing_restoration:
                print(
                    "[warning] restoration metadata is unavailable from the current Stage1 "
                    "data path. reaction_data will be saved in Stage1 processed space and "
                    "space_definition will be omitted."
                )
                warned_missing_restoration = True

            actor_chunks.append(actor_motion.detach().cpu().numpy().astype(np.float32))
            gt_chunks.append(gt_motion.detach().cpu().numpy().astype(np.float32))
            coarse_chunks.append(coarse_motion.detach().cpu().numpy().astype(np.float32))
            length_chunks.append(cond["y"]["lengths"][:keep].detach().cpu().numpy().astype(np.int64))
            sample_index_chunks.append(np.arange(total, total + keep, dtype=np.int64))
            _maybe_collect_extra_fields(cond["y"], keep, extra_chunks)

            total += keep
            iterator.set_postfix(samples=total)

    if total == 0:
        raise RuntimeError("No samples were generated; check dataset split and num_samples.")

    payload = {
        "actor_motion": np.concatenate(actor_chunks, axis=0),
        "reactor_gt": np.concatenate(gt_chunks, axis=0),
        "reactor_coarse": np.concatenate(coarse_chunks, axis=0),
        "lengths": np.concatenate(length_chunks, axis=0),
        "sample_indices": np.concatenate(sample_index_chunks, axis=0),
    }

    for key, chunks in extra_chunks.items():
        value = _finalize_chunks(chunks, key)
        if value is not None:
            payload[key] = value

    if saved_in_restored_space:
        payload["space_definition"] = np.full(
            (payload["sample_indices"].shape[0],),
            RESTORED_PAIR_SPACE,
            dtype=object,
        )

    _save_reaction_data(output_path, payload)
    print(f"Saved reaction_data to {output_path} (samples={payload['sample_indices'].shape[0]})")


if __name__ == "__main__":
    main()
