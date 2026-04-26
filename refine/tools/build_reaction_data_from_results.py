#!/usr/bin/env python3
"""Bridge one Stage1 results.npy sample into a single-sample reaction_data pack.

This is intended for exact Stage2 refinement of an already-exported Stage1 sample,
instead of re-running Stage1 sampling through refine.data.build_reaction_data.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch

from data_loaders.get_data import get_dataset
from refine.data.restored_space import (
    RESTORED_PAIR_SPACE,
    REQUIRED_RESTORATION_METADATA_FIELDS,
    OPTIONAL_RESTORATION_METADATA_FIELDS,
    extract_restoration_metadata,
    restore_pair_batch,
)


def _infer_interx_raw_motions_root() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(os.path.dirname(repo_root), "Inter-X", "datasets", "interx", "motions"),
        os.path.join(repo_root, "dataset", "interx", "motions"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return ""


def _infer_interx_interaction_order_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(repo_root, "dataset", "interx", "annots", "interaction_order.pkl"),
        os.path.join(os.path.dirname(repo_root), "Inter-X", "datasets", "interx", "annots", "interaction_order.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def _normalize_scalar(value: Any, default=None):
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _normalize_scalar(value.item(), default=default)
        if value.size == 0:
            return default
        if value.size == 1:
            return _normalize_scalar(value.reshape(-1)[0], default=default)
    return value


def _meta_at(meta: dict[str, np.ndarray], key: str, idx: int, default=None):
    if key not in meta:
        return default
    value = meta[key]
    if isinstance(value, np.ndarray) and value.ndim > 0:
        if value.shape[0] <= idx:
            return default
        return value[idx]
    return value


def _trim_frame_ix(meta: dict[str, np.ndarray], idx: int, length: int) -> np.ndarray:
    frame_ix = _meta_at(meta, "frame_ix", idx, default=None)
    if frame_ix is None:
        return np.arange(length, dtype=np.int64)
    frame_ix = np.asarray(frame_ix, dtype=np.int64).reshape(-1)
    frame_ix_len = int(_normalize_scalar(_meta_at(meta, "frame_ix_len", idx, default=0), default=0) or 0)
    if frame_ix_len > 0:
        frame_ix = frame_ix[:frame_ix_len]
    frame_ix = frame_ix[frame_ix >= 0]
    if frame_ix.size == 0:
        return np.arange(length, dtype=np.int64)
    return frame_ix


def _infer_index_from_clip_name(clip_name: str) -> int:
    prefix = str(clip_name).split("_", 1)[0]
    return int(prefix)


def _select_index(args: argparse.Namespace, meta: dict[str, np.ndarray]) -> int:
    if args.sample_index is not None:
        return int(args.sample_index)
    if args.clip_name:
        return _infer_index_from_clip_name(args.clip_name)
    if args.dataset_key:
        keys = [str(_normalize_scalar(x, default="")) for x in np.asarray(meta["dataset_key"], dtype=object)]
        try:
            return keys.index(str(args.dataset_key))
        except ValueError as exc:
            raise KeyError(f"dataset_key not found in metadata: {args.dataset_key}") from exc
    return 0


def _batchify_meta_value(value):
    value = _normalize_scalar(value, default=value)
    if isinstance(value, str):
        return np.asarray([value], dtype=object)
    if isinstance(value, bytes):
        return np.asarray([value.decode("utf-8")], dtype=object)
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.reshape(1)
    return np.expand_dims(arr, axis=0)


def _build_restoration_batch(restoration: dict[str, Any]) -> dict[str, Any]:
    out = {}
    keys = REQUIRED_RESTORATION_METADATA_FIELDS + OPTIONAL_RESTORATION_METADATA_FIELDS
    for key in keys:
        if key not in restoration:
            continue
        out[key] = _batchify_meta_value(restoration[key])
    return out


def _resolve_data_path(args: argparse.Namespace, meta: dict[str, np.ndarray], idx: int) -> str:
    if args.data_path:
        return args.data_path
    data_path = _normalize_scalar(_meta_at(meta, "data_path", idx, default=""), default="")
    if data_path:
        return str(data_path)
    dataset = args.dataset or _normalize_scalar(_meta_at(meta, "dataset_name", idx, default="interx"), default="interx")
    split = args.split or _normalize_scalar(_meta_at(meta, "split", idx, default="train"), default="train")
    candidate = os.path.join("dataset", str(dataset), "regen", f"{split}.h5")
    return candidate


def _resolve_split(args: argparse.Namespace, meta: dict[str, np.ndarray], idx: int) -> str:
    if args.split:
        return args.split
    return str(_normalize_scalar(_meta_at(meta, "split", idx, default="train"), default="train"))


def _resolve_dataset_name(args: argparse.Namespace, meta: dict[str, np.ndarray], idx: int) -> str:
    if args.dataset:
        return args.dataset
    return str(_normalize_scalar(_meta_at(meta, "dataset_name", idx, default="interx"), default="interx"))


def _load_meta(meta_path: str) -> dict[str, np.ndarray]:
    data = np.load(meta_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _extract_gt_reactor(dataset, data_index: int, frame_ix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inp, _ = dataset.get_pose_data(data_index, frame_ix)
    if torch.is_tensor(inp):
        inp = inp.detach().cpu().numpy()
    inp = np.asarray(inp, dtype=np.float32)
    if inp.ndim != 3 or inp.shape[1] % 2 != 0:
        raise ValueError(f"Unexpected dataset pose shape for two-person rot6d data: {inp.shape}")
    half = inp.shape[1] // 2
    return inp[:, :half, :], inp[:, half:, :]


def build_reaction_data(args: argparse.Namespace) -> str:
    results = np.load(args.results_path, allow_pickle=True).item()
    meta_path = args.meta_path or os.path.join(os.path.dirname(os.path.abspath(args.results_path)), "results_meta.npz")
    meta = _load_meta(meta_path)
    idx = _select_index(args, meta)

    actor_motion = np.asarray(results["cmotion"][idx], dtype=np.float32)
    coarse_motion = np.asarray(results["output"][idx], dtype=np.float32)
    length = int(np.asarray(results["lengths"])[idx])
    sample_index = int(_normalize_scalar(_meta_at(meta, "sample_idx", idx, default=idx), default=idx))
    dataset_key = str(_normalize_scalar(_meta_at(meta, "dataset_key", idx, default=f"sample_{idx}"), default=f"sample_{idx}"))
    data_index = int(_normalize_scalar(_meta_at(meta, "data_index", idx, default=-1), default=-1))
    if data_index < 0:
        raise ValueError(f"results_meta is missing a valid data_index for sample {idx}.")

    frame_ix = _trim_frame_ix(meta, idx, length)
    dataset_name = _resolve_dataset_name(args, meta, idx)
    split = _resolve_split(args, meta, idx)
    data_path = _resolve_data_path(args, meta, idx)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    dataset = get_dataset(
        name=dataset_name,
        num_frames=max(len(frame_ix), 1),
        num_person=2,
        data_path=data_path,
        pose_rep=args.pose_rep,
        body_model=args.body_model,
        split=split,
        enable_restoration_metadata=True,
        restoration_meta_path=args.restoration_meta_path,
        raw_motions_root=args.raw_motions_root or _infer_interx_raw_motions_root(),
        interaction_order_path=args.interaction_order_path or _infer_interx_interaction_order_path(),
    )

    actor_gt, reactor_gt = _extract_gt_reactor(dataset, data_index, frame_ix)
    restoration = dataset._build_restoration_metadata(data_index, dataset_key, frame_ix)
    restoration_batch = _build_restoration_batch(restoration)
    restoration_meta = extract_restoration_metadata(restoration_batch, device="cpu")

    actor_t = torch.from_numpy(actor_motion[None]).float()
    gt_t = torch.from_numpy(reactor_gt[None]).float()
    coarse_t = torch.from_numpy(coarse_motion[None]).float()
    actor_restored, gt_restored = restore_pair_batch(actor_t, gt_t, restoration_meta)
    _, coarse_restored = restore_pair_batch(actor_restored, coarse_t, restoration_meta)

    actor_mae = float(np.abs(actor_gt - actor_motion).mean())

    payload: dict[str, Any] = {
        "actor_motion": actor_restored.numpy().astype(np.float32),
        "reactor_gt": gt_restored.numpy().astype(np.float32),
        "reactor_coarse": coarse_restored.numpy().astype(np.float32),
        "lengths": np.asarray([length], dtype=np.int64),
        "sample_indices": np.asarray([sample_index], dtype=np.int64),
        "space_definition": np.asarray([RESTORED_PAIR_SPACE], dtype=object),
    }
    for key, value in restoration_batch.items():
        payload[key] = value

    output_path = os.path.abspath(args.output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **payload)

    print(f"Saved single-sample reaction_data to {output_path}")
    print(f"sample_index={sample_index} meta_row={idx} dataset_key={dataset_key}")
    print(f"data_path={data_path} split={split} frames={len(frame_ix)} length={length}")
    print(f"actor_gt_vs_results_cmotion_mae={actor_mae:.8f}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build one reaction_data sample from an existing Stage1 results run.")
    parser.add_argument("--results_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--meta_path", default="", type=str)
    parser.add_argument("--sample_index", default=None, type=int)
    parser.add_argument("--clip_name", default="", type=str)
    parser.add_argument("--dataset_key", default="", type=str)
    parser.add_argument("--dataset", default="", type=str)
    parser.add_argument("--split", default="", type=str)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument("--interaction_order_path", default="", type=str)
    parser.add_argument("--restoration_meta_path", default="", type=str)
    parser.add_argument("--raw_motions_root", default="", type=str)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    build_reaction_data(args)


if __name__ == "__main__":
    main()
