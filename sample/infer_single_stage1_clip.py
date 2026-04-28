"""Run single-sample Stage1 inference and export a viewer-ready clip."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from data_loaders.get_data import get_dataset
from data_loaders.tensors import ccollate
from sample.cgenerate import _build_meta_payload
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.online_window import sliding_window_sample
from utils.parser_util import (
    add_base_options,
    add_data_options,
    add_generate_options,
    add_online_options,
    add_sampling_options,
    parse_and_load_from_model_wo_data,
)
from visualize.converters.convert_results_to_motions import (
    apply_world_align_root_only,
    build_params,
    build_rot_matrix_from_flags,
)


@dataclass
class _DatasetBundle:
    dataset: object
    data_path: str
    split: str
    data_index: int


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_interx_paths() -> dict[str, str]:
    repo_root = _repo_root()
    sibling_interx_root = os.path.join(os.path.dirname(repo_root), "Inter-X")
    repo_interx_root = os.path.join(repo_root, "dataset", "interx")
    repo_motions_root = os.path.join(repo_interx_root, "motions")
    sibling_motions_root = os.path.join(sibling_interx_root, "datasets", "interx", "motions")
    raw_motions_root = repo_motions_root if os.path.isdir(repo_motions_root) else sibling_motions_root
    restoration_meta_path = os.path.join(repo_interx_root, "cache", "interx_restoration_meta.npz")
    return {
        "train": os.path.join(repo_interx_root, "regen", "train.h5"),
        "val": os.path.join(repo_interx_root, "regen", "val.h5"),
        "test": os.path.join(repo_interx_root, "regen", "test.h5"),
        "raw_motions_root": raw_motions_root,
        "interaction_order": os.path.join(repo_interx_root, "annots", "interaction_order.pkl"),
        "restoration_meta_path": restoration_meta_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-sample Stage1 inference and export a viewer-ready clip.")
    add_base_options(parser)
    add_online_options(parser)
    add_data_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    parser.add_argument("--dataset_key", required=True, type=str)
    parser.add_argument("--split", default="auto", choices=["auto", "train", "val", "test"])
    parser.add_argument(
        "--shape_mode",
        choices=["canonical", "restored", "restored_shape_height"],
        default="restored_shape_height",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--raw_motions_root",
        default=_default_interx_paths()["raw_motions_root"],
        type=str,
        help="Path to Inter-X raw motions root used for restored export.",
    )
    parser.add_argument(
        "--restoration_meta_path",
        default=_default_interx_paths()["restoration_meta_path"],
        type=str,
        help="Path to Inter-X restoration metadata package used for restored export.",
    )
    return parser


def _load_args():
    parser = _build_parser()
    args = parse_and_load_from_model_wo_data(parser)
    defaults = _default_interx_paths()
    if not getattr(args, "interaction_order", ""):
        args.interaction_order = defaults["interaction_order"]
    if not getattr(args, "raw_motions_root", ""):
        args.raw_motions_root = defaults["raw_motions_root"]
    if not getattr(args, "restoration_meta_path", ""):
        args.restoration_meta_path = defaults["restoration_meta_path"]
    if args.dataset == "interx":
        if int(args.num_person) == 1:
            args.num_person = 2
        if str(args.body_model).lower() == "smpl":
            args.body_model = "smplx"
        if float(args.motion_length) == 60:
            args.motion_length = 150
    return args


def _make_dataset(args, *, data_path: str, split: str):
    return get_dataset(
        name=args.dataset,
        num_frames=150,
        num_person=args.num_person,
        data_path=data_path,
        pose_rep=args.pose_rep,
        body_model=args.body_model,
        split=split,
        enable_restoration_metadata=True,
        restoration_meta_path=args.restoration_meta_path,
        interaction_order_path=args.interaction_order,
        raw_motions_root=args.raw_motions_root,
    )


def _gender_name_from_id(value) -> str:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 0
    if value == 1:
        return "male"
    if value == 2:
        return "female"
    return "neutral"


def _resolve_dataset_bundle(args) -> _DatasetBundle:
    if args.dataset != "interx":
        if not args.data_path:
            raise ValueError("single-sample Stage1 infer currently expects --data_path for non-InterX datasets")
        split = args.split if args.split != "auto" else "test"
        dataset = _make_dataset(args, data_path=args.data_path, split=split)
        try:
            data_index = dataset.keys.index(args.dataset_key)
        except ValueError as exc:
            raise KeyError(f"dataset_key not found in {args.data_path}: {args.dataset_key}") from exc
        return _DatasetBundle(dataset=dataset, data_path=args.data_path, split=split, data_index=data_index)

    defaults = _default_interx_paths()
    candidates: list[tuple[str, str]] = []
    if args.data_path:
        split = args.split if args.split != "auto" else "test"
        candidates.append((args.data_path, split))
    else:
        split_names = ["train", "val", "test"] if args.split == "auto" else [args.split]
        for split_name in split_names:
            path = defaults[split_name]
            if os.path.exists(path):
                candidates.append((path, split_name))
    for path, split in candidates:
        dataset = _make_dataset(args, data_path=path, split=split)
        if args.dataset_key in dataset.keys:
            return _DatasetBundle(
                dataset=dataset,
                data_path=path,
                split=split,
                data_index=dataset.keys.index(args.dataset_key),
            )
    searched = ", ".join(path for path, _ in candidates) or "(none)"
    raise KeyError(f"dataset_key not found: {args.dataset_key}. searched: {searched}")


def _holder(dataset):
    class _Holder:
        pass

    out = _Holder()
    out.dataset = dataset
    return out


def _sample_one(args, bundle: _DatasetBundle):
    dataset = bundle.dataset
    item = dataset._get_item_data_index(bundle.data_index)
    action_name = item.get("action_text", dataset.action_to_action_name(item["action"]))
    args.batch_size = 1
    args.num_samples = 1
    args.num_repetitions = 1
    max_frames = 150 if args.dataset in ["chi3d", "interx"] else 60
    n_frames = min(max_frames, int(args.motion_length))

    dist_util.setup_dist()
    model, diffusion = create_model_and_diffusion(args, _holder(dataset))
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    model.to(dist_util.dev())
    model.eval()

    collate_args = [
        dict(
            tokens=None,
            lengths=n_frames,
            action=item["action"],
            action_text=action_name,
            inp=item["inp"].to(dist_util.dev()),
        )
    ]
    _, model_kwargs = ccollate(collate_args)
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    model_window_size = None
    if getattr(model, "arch", None) == "mlp":
        try:
            model_window_size = int(model.mlp.motion_mlp.mlps[0].fc0.in_channels)
        except Exception:
            model_window_size = None
    if args.reaction_mode == "online" and args.online_strategy == "sliding_window":
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
    elif args.reaction_mode == "online" and args.online_strategy == "autoregressive":
        cmotion_bak = model_kwargs["y"]["cmotion"]
        B, V, C, T = cmotion_bak.shape
        cmotion = torch.zeros_like(cmotion_bak)
        output = torch.zeros((B, V, C, T), device=cmotion_bak.device)
        for frame_idx in range(T):
            cmotion[:, :, :, frame_idx] = cmotion_bak[:, :, :, frame_idx]
            model_kwargs["y"]["cmotion"] = cmotion
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
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

    sample_gf = gaussian_filter1d(sample.detach().cpu().numpy(), sigma=1, axis=-1)
    sample = torch.from_numpy(sample_gf).to(sample.device)
    rot2xyz_pose_rep = "xyz" if model.data_rep == "xyz" else model.data_rep
    rot2xyz_mask = None if rot2xyz_pose_rep == "xyz" else model_kwargs["y"]["mask"].reshape(args.batch_size, n_frames).bool()
    motion_xyz = model.rot2xyz(
        x=sample,
        mask=rot2xyz_mask,
        pose_rep=rot2xyz_pose_rep,
        glob=True,
        translation=True,
        jointstype=args.body_model,
        vertstrans=True,
        num_person=1,
        betas=None,
        beta=0,
        glob_rot=None,
        get_rotations_back=False,
    )

    return {
        "item": item,
        "action_name": action_name,
        "reactor_output": sample_gf[0].astype(np.float32),
        "actor_cmotion": model_kwargs["y"]["cmotion"].detach().cpu().numpy()[0].astype(np.float32),
        "motion_xyz": motion_xyz.detach().cpu().numpy()[0].astype(np.float32),
        "length": int(model_kwargs["y"]["lengths"].detach().cpu().numpy()[0]),
    }


def _save_results_and_meta(args, bundle: _DatasetBundle, sample_out: dict[str, object], out_dir: str):
    item = sample_out["item"]
    length = int(sample_out["length"])
    actor_reactor_mapping = "actor=P1,reactor=P2" if int(item.get("actor_is_p1", 1)) == 1 else "actor=P2,reactor=P1"
    meta = [
        {
            "sample_idx": 0,
            "rep_i": 0,
            "action_i": 0,
            "action_id": int(item["action"]),
            "action_name": sample_out["action_name"],
            "data_index": int(item["data_index"]),
            "dataset_key": item["dataset_key"],
            "actor_reactor_mapping": actor_reactor_mapping,
            "length": int(item.get("sampled_num_frames", length)),
            "raw_nframes": int(item.get("raw_nframes", -1)),
            "start_frame": int(item["frame_ix"][0]) if len(item.get("frame_ix", [])) else -1,
            "end_frame": int(item["frame_ix"][-1]) if len(item.get("frame_ix", [])) else -1,
            "frame_ix": np.asarray(item.get("frame_ix", np.empty((0,), dtype=np.int64)), dtype=np.int64),
            "sampling": item.get("sampling", ""),
            "sampling_step": int(item.get("sampling_step", -1)),
            "num_frames": 150,
            "motion_length": int(args.motion_length),
            "split": bundle.split,
            "dataset_name": args.dataset,
            "data_path": bundle.data_path,
            "actor_is_p1": int(item.get("actor_is_p1", 1)),
            "downsample": int(item.get("downsample", 4 if args.dataset == "interx" else 1)),
        }
    ]
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "results.npy"),
        {
            "motion": sample_out["motion_xyz"][None],
            "output": sample_out["reactor_output"][None],
            "cmotion": sample_out["actor_cmotion"][None],
            "text": [sample_out["action_name"]],
            "lengths": np.asarray([length], dtype=np.int64),
            "num_samples": 1,
            "num_repetitions": 1,
        },
    )
    with open(os.path.join(out_dir, "results.txt"), "w") as f:
        f.write(f"{sample_out['action_name']}\n")
    with open(os.path.join(out_dir, "results_len.txt"), "w") as f:
        f.write(f"{length}\n")
    with open(os.path.join(out_dir, "map.txt"), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["output_index", "rep_i", "action_i", "action_name", "action_id", "data_index", "dataset_key"])
        writer.writerow([0, 0, 0, sample_out["action_name"], int(item["action"]), int(item["data_index"]), item["dataset_key"]])
    np.savez_compressed(os.path.join(out_dir, "results_meta.npz"), **_build_meta_payload(meta))


def _export_motion_clip(args, sample_out: dict[str, object], out_dir: str):
    item = sample_out["item"]
    dataset_key = str(item["dataset_key"])
    length = int(sample_out["length"])
    motions_dir = os.path.join(out_dir, "motions", dataset_key)
    os.makedirs(motions_dir, exist_ok=True)

    p1_betas = p1_gender = p2_betas = p2_gender = None
    actor_is_p1 = int(item.get("actor_is_p1", 1))
    frame_ix = np.asarray(item.get("frame_ix", np.empty((0,), dtype=np.int64)), dtype=np.int64)
    downsample = int(item.get("downsample", 4 if args.dataset == "interx" else 1))
    if args.shape_mode in {"restored", "restored_shape_height"}:
        p1_betas = np.asarray(item.get("actor_betas", np.zeros((10,), dtype=np.float32)), dtype=np.float32)
        p2_betas = np.asarray(item.get("reactor_betas", np.zeros((10,), dtype=np.float32)), dtype=np.float32)
        p1_gender = _gender_name_from_id(item.get("actor_gender_id", 0))
        p2_gender = _gender_name_from_id(item.get("reactor_gender_id", 0))

    meta_common = {
        "dataset_key": dataset_key,
        "actor_is_p1": int(actor_is_p1),
        "actor_reactor_mapping": "actor=P1,reactor=P2" if actor_is_p1 == 1 else "actor=P2,reactor=P1",
        "sample_idx": 0,
        "data_index": int(item["data_index"]),
        "action_id": int(item["action"]),
        "action_name": sample_out["action_name"],
        "rep_i": 0,
        "action_i": 0,
        "frame_ix": frame_ix,
        "start_frame": int(frame_ix[0]) if frame_ix.size else -1,
        "end_frame": int(frame_ix[-1]) if frame_ix.size else -1,
        "raw_nframes": int(item.get("raw_nframes", -1)),
        "sampling": item.get("sampling", ""),
        "sampling_step": int(item.get("sampling_step", -1)),
        "num_frames": 150,
        "motion_length": int(args.motion_length),
        "downsample": int(downsample),
        "source_model_setting": str(args.setting),
    }

    trans_p1 = trans_p2 = None
    if args.shape_mode == "restored_shape_height":
        trans_p1 = np.asarray(
            item.get("actor_raw_trans_clip", sample_out["actor_cmotion"][55, 0:3, :].T),
            dtype=np.float32,
        )[:length]
        trans_p2 = np.asarray(
            item.get("reactor_raw_trans_clip", sample_out["reactor_output"][55, 0:3, :].T),
            dtype=np.float32,
        )[:length]

    p1_params = build_params(
        sample_out["actor_cmotion"],
        length,
        betas=p1_betas,
        gender=p1_gender,
        meta={**meta_common, "source_role": "actor"},
        trans_override=trans_p1,
    )
    p2_params = build_params(
        sample_out["reactor_output"],
        length,
        betas=p2_betas,
        gender=p2_gender,
        meta={**meta_common, "source_role": "reactor"},
        trans_override=trans_p2,
    )
    rot_matrix = build_rot_matrix_from_flags(False, False, False)
    if rot_matrix is not None:
        p1_params = apply_world_align_root_only(p1_params, rot_matrix)
        p2_params = apply_world_align_root_only(p2_params, rot_matrix)

    np.savez(os.path.join(motions_dir, "P1.npz"), **p1_params)
    np.savez(os.path.join(motions_dir, "P2.npz"), **p2_params)


def main():
    args = _load_args()
    fixseed(args.seed)
    bundle = _resolve_dataset_bundle(args)
    sample_out = _sample_one(args, bundle)
    _save_results_and_meta(args, bundle, sample_out, args.output_dir)
    _export_motion_clip(args, sample_out, args.output_dir)
    manifest = {
        "dataset_key": args.dataset_key,
        "split": bundle.split,
        "data_path": bundle.data_path,
        "setting": args.setting,
        "baseline_family": getattr(args, "baseline_family", "regennet"),
        "shape_mode": args.shape_mode,
        "restoration_meta_path": args.restoration_meta_path,
        "output_dir": os.path.abspath(args.output_dir),
        "motion_clip_dir": os.path.abspath(os.path.join(args.output_dir, "motions", args.dataset_key)),
    }
    with open(os.path.join(args.output_dir, "single_infer_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved single-sample Stage1 inference: {os.path.abspath(args.output_dir)}")
    print(f"motion clip: {os.path.abspath(os.path.join(args.output_dir, 'motions', args.dataset_key))}")


if __name__ == "__main__":
    main()
