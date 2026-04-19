#!/usr/bin/env python3
"""Convert Stage2-Lite refined_pack.npz into Inter-X viewer-ready clips.

This converter is for Stage2-Lite inference packs, not Stage1 results.npy.
It directly consumes restored_pair_space pack fields and writes the viewer's
standard clip folders:

    <output_dir>/<clip_name>/P1.npz
    <output_dir>/<clip_name>/P2.npz

No restored-space transform is applied here. Stage2 packs are already produced
in restored pair space, so this script only converts rot6d motion tensors into
SMPL-X parameter npz files and preserves the actor/reactor metadata.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

from visualize.converters.convert_results_to_motions import (
    rot6d_to_rotvec,
    sanitize_name,
)


VARIANT_TO_FIELD = {
    "gt": "reactor_gt",
    "coarse": "reactor_coarse",
    "refined": "reactor_refined",
}

GENDER_ID_TO_NAME = {
    0: "neutral",
    1: "male",
    2: "female",
}


def _load_pack(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _normalize_str(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _normalize_str(value.item(), default=default)
        if value.size == 0:
            return default
        if value.size == 1:
            return _normalize_str(value.reshape(-1)[0], default=default)
    text = str(value)
    return text if text else default


def _field_at(pack: dict[str, Any], key: str, idx: int, default=None):
    if key not in pack:
        return default
    value = pack[key]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.shape[0] > idx:
            return value[idx]
    return value


def _gender_from_pack(pack: dict[str, Any], key: str, idx: int) -> str:
    value = _field_at(pack, key, idx, default=0)
    if isinstance(value, bytes):
        return _normalize_str(value, default="neutral").lower()
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    try:
        return GENDER_ID_TO_NAME.get(int(value), "neutral")
    except (TypeError, ValueError):
        return "neutral"


def _betas_from_pack(pack: dict[str, Any], key: str, idx: int) -> np.ndarray:
    value = _field_at(pack, key, idx, default=None)
    if value is None:
        return np.zeros(10, dtype=np.float32)
    betas = np.asarray(value, dtype=np.float32).reshape(-1)
    if betas.size >= 10:
        return betas[:10].astype(np.float32)
    out = np.zeros(10, dtype=np.float32)
    out[: betas.size] = betas
    return out


def _frame_array(pack: dict[str, Any], key: str, idx: int, length: int) -> np.ndarray:
    value = _field_at(pack, key, idx, default=None)
    if value is None:
        return np.empty((0,), dtype=np.int64)
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr
    return arr[:length]


def _check_motion_for_viewer(motion: np.ndarray, field_name: str):
    if motion.ndim != 3:
        raise ValueError(f"{field_name} sample must have shape [J, F, T], got {motion.shape}.")
    if motion.shape[0] < 55:
        raise ValueError(f"{field_name} needs at least 55 SMPL-X joints, got {motion.shape[0]}.")
    if motion.shape[1] < 6:
        raise ValueError(
            f"{field_name} has feature dim {motion.shape[1]}. Viewer conversion requires rot6d "
            "motion with at least 6 features; xyz-only packs cannot reconstruct SMPL-X poses."
        )


def _build_params_from_motion(
    motion: np.ndarray,
    *,
    length: int,
    betas: np.ndarray,
    gender: str,
    meta: dict[str, Any],
    raw_trans_clip: np.ndarray | None = None,
) -> dict[str, Any]:
    _check_motion_for_viewer(motion, meta.get("source_field", "motion"))
    length = min(int(length), int(motion.shape[-1]))

    rot6d = np.transpose(motion[:55, :6, :], (2, 0, 1))
    rotvec = rot6d_to_rotvec(rot6d).astype(np.float32)[:length]

    if motion.shape[0] > 55 and motion.shape[1] >= 3:
        trans = motion[55, :3, :].T.astype(np.float32)[:length]
    elif raw_trans_clip is not None and np.asarray(raw_trans_clip).size > 0:
        trans = np.asarray(raw_trans_clip, dtype=np.float32).reshape(-1, 3)[:length]
    else:
        trans = np.zeros((length, 3), dtype=np.float32)

    params = {
        "root_orient": rotvec[:, 0],
        "pose_body": rotvec[:, 1:22],
        "pose_lhand": rotvec[:, 25:40],
        "pose_rhand": rotvec[:, 40:55],
        "trans": trans,
        "betas": betas.astype(np.float32),
        "gender": gender,
    }
    params.update(meta)
    return params


def _build_common_meta(pack: dict[str, Any], idx: int, *, variant: str, length: int) -> dict[str, Any]:
    dataset_key = _normalize_str(_field_at(pack, "dataset_key", idx, default=f"sample_{idx}"))
    processed_frame_ix = _frame_array(pack, "processed_frame_ix", idx, length)
    raw_frame_ix = _frame_array(pack, "raw_frame_ix", idx, length)
    start_frame = int(raw_frame_ix[0]) if raw_frame_ix.size > 0 else -1
    end_frame = int(raw_frame_ix[-1]) if raw_frame_ix.size > 0 else -1
    sample_idx = int(_field_at(pack, "sample_indices", idx, default=idx))

    return {
        "dataset_key": dataset_key,
        "sample_idx": sample_idx,
        "stage2_variant": variant,
        "space_definition": _normalize_str(_field_at(pack, "space_definition", idx, default="")),
        "actor_is_p1": int(_field_at(pack, "actor_is_p1", idx, default=-1)),
        "reactor_is_p2": int(_field_at(pack, "reactor_is_p2", idx, default=-1)),
        "processed_frame_ix": processed_frame_ix,
        "raw_frame_ix": raw_frame_ix,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "processed_nframes": int(_field_at(pack, "processed_nframes", idx, default=-1)),
        "raw_nframes": int(_field_at(pack, "raw_nframes", idx, default=-1)),
        "processed_fps": int(_field_at(pack, "processed_fps", idx, default=-1)),
        "raw_fps": int(_field_at(pack, "raw_fps", idx, default=-1)),
        "downsample": int(_field_at(pack, "downsample", idx, default=-1)),
        "motion_length": int(length),
        "body_model_type": _normalize_str(_field_at(pack, "body_model_type", idx, default="smplx")),
    }


def _clip_name(pack: dict[str, Any], idx: int, variant: str) -> str:
    dataset_key = _normalize_str(_field_at(pack, "dataset_key", idx, default=f"sample_{idx}"))
    key_tail = dataset_key.strip("/").split("/")[-1] if dataset_key else f"sample_{idx}"
    return f"{idx:04d}_{variant}_{sanitize_name(key_tail)}"


def _write_sample(
    pack: dict[str, Any],
    *,
    idx: int,
    variant: str,
    output_dir: str,
    preserve_raw_person_order: bool,
    overwrite: bool,
):
    reactor_field = VARIANT_TO_FIELD[variant]
    actor_motion = np.asarray(pack["actor_motion"][idx], dtype=np.float32)
    reactor_motion = np.asarray(pack[reactor_field][idx], dtype=np.float32)
    length = int(_field_at(pack, "lengths", idx, default=actor_motion.shape[-1]))
    length = min(length, actor_motion.shape[-1], reactor_motion.shape[-1])

    common_meta = _build_common_meta(pack, idx, variant=variant, length=length)
    actor_params = _build_params_from_motion(
        actor_motion,
        length=length,
        betas=_betas_from_pack(pack, "actor_betas", idx),
        gender=_gender_from_pack(pack, "actor_gender_id", idx),
        raw_trans_clip=_field_at(pack, "actor_raw_trans_clip", idx, default=None),
        meta={**common_meta, "source_role": "actor", "source_field": "actor_motion"},
    )
    reactor_params = _build_params_from_motion(
        reactor_motion,
        length=length,
        betas=_betas_from_pack(pack, "reactor_betas", idx),
        gender=_gender_from_pack(pack, "reactor_gender_id", idx),
        raw_trans_clip=_field_at(pack, "reactor_raw_trans_clip", idx, default=None),
        meta={**common_meta, "source_role": "reactor", "source_field": reactor_field},
    )

    clip_dir = os.path.join(output_dir, _clip_name(pack, idx, variant))
    os.makedirs(clip_dir, exist_ok=True)
    p1_path = os.path.join(clip_dir, "P1.npz")
    p2_path = os.path.join(clip_dir, "P2.npz")
    if not overwrite and os.path.exists(p1_path) and os.path.exists(p2_path):
        return False

    actor_is_p1 = int(common_meta.get("actor_is_p1", -1))
    if preserve_raw_person_order and actor_is_p1 == 0:
        p1_params, p2_params = reactor_params, actor_params
    else:
        p1_params, p2_params = actor_params, reactor_params

    np.savez(p1_path, **p1_params)
    np.savez(p2_path, **p2_params)
    return True


def _validate_pack(pack: dict[str, Any], variants: list[str]):
    required = ["actor_motion", "lengths"]
    required.extend(VARIANT_TO_FIELD[variant] for variant in variants)
    missing = [key for key in required if key not in pack]
    if missing:
        raise KeyError(f"Stage2 pack missing required fields: {', '.join(missing)}")


def convert_pack(args: argparse.Namespace):
    pack = _load_pack(args.pack)
    variants = list(VARIANT_TO_FIELD) if args.variant == "all" else [args.variant]
    _validate_pack(pack, variants)

    num_samples = int(np.asarray(pack["actor_motion"]).shape[0])
    limit = num_samples if args.limit is None else min(int(args.limit), num_samples)
    os.makedirs(args.output_dir, exist_ok=True)

    for variant in variants:
        variant_out = os.path.join(args.output_dir, variant) if args.variant == "all" else args.output_dir
        os.makedirs(variant_out, exist_ok=True)
        written = 0
        for idx in range(limit):
            if _write_sample(
                pack,
                idx=idx,
                variant=variant,
                output_dir=variant_out,
                preserve_raw_person_order=args.preserve_raw_person_order,
                overwrite=args.overwrite,
            ):
                written += 1
        print(f"[{variant}] Converted {written}/{limit} samples to {variant_out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Stage2-Lite refined_pack.npz to viewer-ready P1/P2 clips.")
    parser.add_argument("--pack", required=True, type=str, help="Path to Stage2 refined_pack.npz.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory that will contain viewer clip folders.")
    parser.add_argument("--variant", choices=["refined", "coarse", "gt", "all"], default="refined")
    parser.add_argument("--limit", default=None, type=int, help="Only convert the first N samples.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing P1/P2 clip files.")
    parser.add_argument(
        "--preserve_raw_person_order",
        action="store_true",
        help="Write raw P1/P2 order when actor_is_p1 is available; default writes P1=actor, P2=reactor.",
    )
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    convert_pack(args)


if __name__ == "__main__":
    main()
