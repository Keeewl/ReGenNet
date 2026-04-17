#!/usr/bin/env python3
"""Convert ReGenNet results.npy into Inter-X SMPL-X npz clips."""

from __future__ import annotations

import argparse
import os
import re
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return vec / norm


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    rotmat = rot6d_to_rotmat(rot6d)
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    rotvec_flat = R.from_matrix(rotmat_flat).as_rotvec()
    return rotvec_flat.reshape(rot6d.shape[:-1] + (3,))


def sanitize_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    if not name:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def build_params(
    person_data: np.ndarray,
    length: int,
    betas: np.ndarray | None = None,
    gender: str | None = None,
    meta: dict | None = None,
    trans_override: np.ndarray | None = None,
) -> dict:
    rot6d = np.transpose(person_data[:55], (2, 0, 1))
    rotvec = rot6d_to_rotvec(rot6d).astype(np.float32)
    rotvec = rotvec[:length]

    if trans_override is None:
        trans = person_data[55, 0:3, :].T.astype(np.float32)[:length]
    else:
        trans = np.asarray(trans_override, dtype=np.float32)[:length]

    params = {
        "root_orient": rotvec[:, 0],
        "pose_body": rotvec[:, 1:22],
        "pose_lhand": rotvec[:, 25:40],
        "pose_rhand": rotvec[:, 40:55],
        "trans": trans,
        "betas": np.zeros(10, dtype=np.float32) if betas is None else betas,
        "gender": "neutral" if gender is None else gender,
    }
    if meta:
        params.update(meta)
    return params


def build_rot_matrix_from_flags(flip_x: bool, flip_y: bool, flip_z: bool):
    if not (flip_x or flip_y or flip_z):
        return None
    rot_matrix = np.eye(3, dtype=np.float32)
    if flip_x:
        rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        rot_matrix = rot_x @ rot_matrix
    if flip_y:
        rot_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
        rot_matrix = rot_y @ rot_matrix
    if flip_z:
        rot_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
        rot_matrix = rot_z @ rot_matrix
    return rot_matrix


def apply_world_align_root_only(params: dict, rot_matrix: np.ndarray) -> dict:
    if rot_matrix is None:
        return params
    rot = R.from_matrix(rot_matrix)

    root_orient = params["root_orient"].reshape(-1, 3)
    root_rot = R.from_rotvec(root_orient)
    params["root_orient"] = (rot * root_rot).as_rotvec().reshape(params["root_orient"].shape)

    trans = params["trans"].reshape(-1, 3)
    params["trans"] = (rot_matrix @ trans.T).T.reshape(params["trans"].shape)
    return params


def infer_dataset(run_name: str, dataset: str) -> str:
    if dataset != "auto":
        return dataset
    name = run_name.lower()
    if "chi3d" in name:
        return "chi3d"
    if "interx" in name:
        return "interx"
    return "interx"


def resolve_sources(results: dict, p1_source: str, p2_source: str) -> tuple[str, str]:
    if p1_source in results and p2_source in results:
        return p1_source, p2_source
    if "output" in results and "cmotion" in results:
        return "cmotion", "output"
    if "output" in results:
        return "output", "output"
    raise KeyError("results.npy missing expected keys for motion data")


def load_results_meta(meta_path: str) -> dict | None:
    if not meta_path or not os.path.exists(meta_path):
        return None
    data = np.load(meta_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_interaction_order(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_meta_value(meta: dict, key: str, idx: int, default=None):
    if meta is None or key not in meta:
        return default
    val = meta[key]
    if isinstance(val, np.ndarray) and val.shape and val.shape[0] > idx:
        return val[idx]
    return val


def _normalize_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _normalize_str(value.item())
    return str(value)


def resolve_actor_is_p1(meta: dict | None, idx: int, dataset_key: str, order_dict: dict) -> int:
    actor_is_p1 = _get_meta_value(meta, "actor_is_p1", idx, default=-1)
    if actor_is_p1 in (0, 1):
        return int(actor_is_p1)
    label = order_dict.get(dataset_key, None)
    if label is None:
        return -1
    return 1 if int(label) == 1 else 0


def load_raw_shapes(motions_root: str, dataset_key: str) -> tuple[np.ndarray, str, np.ndarray, str]:
    p1_path = os.path.join(motions_root, dataset_key, "P1.npz")
    p2_path = os.path.join(motions_root, dataset_key, "P2.npz")
    p1 = np.load(p1_path, allow_pickle=True)
    p2 = np.load(p2_path, allow_pickle=True)
    betas_p1 = np.asarray(p1["betas"]).astype(np.float32)
    betas_p2 = np.asarray(p2["betas"]).astype(np.float32)
    gender_p1 = str(p1["gender"])
    gender_p2 = str(p2["gender"])
    return betas_p1, gender_p1, betas_p2, gender_p2


def resolve_role_shapes(actor_is_p1: int, betas_p1, gender_p1, betas_p2, gender_p2):
    if actor_is_p1 == 1:
        actor = (betas_p1, gender_p1)
        reactor = (betas_p2, gender_p2)
    elif actor_is_p1 == 0:
        actor = (betas_p2, gender_p2)
        reactor = (betas_p1, gender_p1)
    else:
        actor = (None, None)
        reactor = (None, None)
    return actor, reactor


def load_raw_trans(motions_root: str, dataset_key: str) -> tuple[np.ndarray, np.ndarray]:
    p1_path = os.path.join(motions_root, dataset_key, "P1.npz")
    p2_path = os.path.join(motions_root, dataset_key, "P2.npz")
    p1 = np.load(p1_path, allow_pickle=True)
    p2 = np.load(p2_path, allow_pickle=True)
    trans_p1 = np.asarray(p1["trans"]).astype(np.float32)
    trans_p2 = np.asarray(p2["trans"]).astype(np.float32)
    return trans_p1, trans_p2


def select_raw_trans(trans_seq: np.ndarray, raw_indices: np.ndarray, length: int) -> np.ndarray:
    if raw_indices is None or raw_indices.size == 0:
        raw_indices = np.arange(min(length, trans_seq.shape[0]), dtype=np.int64)
    raw_indices = np.clip(raw_indices, 0, trans_seq.shape[0] - 1)
    trans = trans_seq[raw_indices]
    if trans.shape[0] < length:
        pad = np.repeat(trans[-1:], length - trans.shape[0], axis=0)
        trans = np.concatenate([trans, pad], axis=0)
    return trans[:length]

def convert_one_run(results_path: str, out_dir: str, args: argparse.Namespace, run_name: str) -> None:
    results = np.load(results_path, allow_pickle=True).item()
    p1_source, p2_source = resolve_sources(results, args.p1_source, args.p2_source)
    p1_data = results[p1_source]
    p2_data = results[p2_source]

    meta = None
    order_dict = {}
    if args.shape_mode in {"restored", "restored_shape_height"}:
        meta_path = args.meta_path or os.path.join(os.path.dirname(results_path), "results_meta.npz")
        meta = load_results_meta(meta_path)
        if meta is None:
            raise FileNotFoundError(f"metadata file not found: {meta_path}")
        order_dict = load_interaction_order(args.interaction_order)

    texts = results.get("text", [])
    lengths = results.get("lengths", None)
    if lengths is None:
        lengths = np.full((p1_data.shape[0],), p1_data.shape[-1], dtype=np.int32)
    else:
        lengths = np.asarray(lengths)

    os.makedirs(out_dir, exist_ok=True)
    num_samples = p1_data.shape[0]
    limit = num_samples if args.limit is None else min(args.limit, num_samples)

    dataset = infer_dataset(run_name, args.dataset)
    flip_x = args.flip_x
    flip_y = args.flip_y
    flip_z = args.flip_z
    if dataset == "chi3d" and not (flip_x or flip_y or flip_z):
        flip_x = True
    rot_matrix = build_rot_matrix_from_flags(flip_x, flip_y, flip_z)

    for i in range(limit):
        length = int(lengths[i])
        action_text = texts[i] if i < len(texts) else ""
        clip_name = f"{i:04d}_{sanitize_name(action_text)}"
        clip_dir = os.path.join(out_dir, clip_name)
        os.makedirs(clip_dir, exist_ok=True)

        p1_path = os.path.join(clip_dir, "P1.npz")
        p2_path = os.path.join(clip_dir, "P2.npz")
        if not args.overwrite and os.path.exists(p1_path) and os.path.exists(p2_path):
            continue

        p1_betas = None
        p1_gender = None
        p2_betas = None
        p2_gender = None
        meta_common = {}
        if args.shape_mode in {"restored", "restored_shape_height"}:
            dataset_key = _normalize_str(_get_meta_value(meta, "dataset_key", i, default=""))
            if not dataset_key:
                raise ValueError(f"Missing dataset_key for sample {i} in metadata.")
            actor_is_p1 = resolve_actor_is_p1(meta, i, dataset_key, order_dict)
            betas_p1, gender_p1, betas_p2, gender_p2 = load_raw_shapes(
                args.raw_motions_root, dataset_key
            )
            actor_shape, reactor_shape = resolve_role_shapes(
                actor_is_p1, betas_p1, gender_p1, betas_p2, gender_p2
            )
            p1_role = "actor" if p1_source == "cmotion" else "reactor"
            p2_role = "actor" if p2_source == "cmotion" else "reactor"
            p1_betas, p1_gender = actor_shape if p1_role == "actor" else reactor_shape
            p2_betas, p2_gender = actor_shape if p2_role == "actor" else reactor_shape

            frame_ix = _get_meta_value(meta, "frame_ix", i, default=None)
            frame_ix_len = int(_get_meta_value(meta, "frame_ix_len", i, default=0))
            if isinstance(frame_ix, np.ndarray) and frame_ix.size > 0:
                if frame_ix_len > 0:
                    frame_ix = frame_ix[:frame_ix_len]
                frame_ix = frame_ix[frame_ix >= 0]
            else:
                frame_ix = None

            start_frame = int(_get_meta_value(meta, "start_frame", i, default=-1))
            end_frame = int(_get_meta_value(meta, "end_frame", i, default=-1))
            if start_frame < 0 and frame_ix is not None and len(frame_ix) > 0:
                start_frame = int(frame_ix[0])
                end_frame = int(frame_ix[-1])

            downsample = int(_get_meta_value(meta, "downsample", i, default=1))
            if downsample < 1:
                downsample = 1

            meta_common = {
                "dataset_key": dataset_key,
                "actor_is_p1": int(actor_is_p1),
                "actor_reactor_mapping": _normalize_str(
                    _get_meta_value(meta, "actor_reactor_mapping", i, default="")
                ),
                "sample_idx": int(_get_meta_value(meta, "sample_idx", i, default=i)),
                "data_index": int(_get_meta_value(meta, "data_index", i, default=-1)),
                "action_id": int(_get_meta_value(meta, "action_id", i, default=-1)),
                "action_name": _normalize_str(_get_meta_value(meta, "action_name", i, default="")),
                "rep_i": int(_get_meta_value(meta, "rep_i", i, default=-1)),
                "action_i": int(_get_meta_value(meta, "action_i", i, default=-1)),
                "frame_ix": frame_ix if frame_ix is not None else np.empty((0,), dtype=np.int64),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "raw_nframes": int(_get_meta_value(meta, "raw_nframes", i, default=-1)),
                "sampling": _normalize_str(_get_meta_value(meta, "sampling", i, default="")),
                "sampling_step": int(_get_meta_value(meta, "sampling_step", i, default=-1)),
                "num_frames": int(_get_meta_value(meta, "num_frames", i, default=-1)),
                "motion_length": int(_get_meta_value(meta, "motion_length", i, default=-1)),
                "downsample": int(downsample),
            }

        trans_p1 = None
        trans_p2 = None
        if args.shape_mode == "restored_shape_height":
            raw_trans_p1, raw_trans_p2 = load_raw_trans(args.raw_motions_root, dataset_key)
            if frame_ix is not None and len(frame_ix) > 0:
                raw_indices = np.asarray(frame_ix, dtype=np.int64) * downsample
            else:
                raw_indices = None
            actor_trans = select_raw_trans(
                raw_trans_p1 if actor_is_p1 == 1 else raw_trans_p2,
                raw_indices,
                length,
            )
            reactor_trans = select_raw_trans(
                raw_trans_p2 if actor_is_p1 == 1 else raw_trans_p1,
                raw_indices,
                length,
            )
            p1_role = "actor" if p1_source == "cmotion" else "reactor"
            p2_role = "actor" if p2_source == "cmotion" else "reactor"
            trans_p1 = actor_trans if p1_role == "actor" else reactor_trans
            trans_p2 = actor_trans if p2_role == "actor" else reactor_trans

        p1_params = build_params(
            p1_data[i],
            length,
            betas=p1_betas,
            gender=p1_gender,
            meta={**meta_common, "source_role": "actor" if p1_source == "cmotion" else "reactor"},
            trans_override=trans_p1,
        )
        p2_params = build_params(
            p2_data[i],
            length,
            betas=p2_betas,
            gender=p2_gender,
            meta={**meta_common, "source_role": "actor" if p2_source == "cmotion" else "reactor"},
            trans_override=trans_p2,
        )
        if rot_matrix is not None:
            p1_params = apply_world_align_root_only(p1_params, rot_matrix)
            p2_params = apply_world_align_root_only(p2_params, rot_matrix)
        np.savez(p1_path, **p1_params)
        np.savez(p2_path, **p2_params)

    print(f"[{run_name}] Done. Converted {limit} samples to {out_dir}")


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sibling_interx_root = os.path.join(os.path.dirname(repo_root), 'Inter-X')
    default_motions_root = os.path.join(sibling_interx_root, 'datasets', 'interx', 'motions')
    default_order = os.path.join(repo_root, 'dataset', 'interx', 'annots', 'interaction_order.pkl')
    if not os.path.exists(default_order):
        default_order = os.path.join(sibling_interx_root, 'datasets', 'interx', 'annots', 'interaction_order.pkl')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs_root",
        default="outputs",
        help="Root directory that contains ReGenNet output folders",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Specific run folder names under outputs/ to process (default: all with results.npy)",
    )
    parser.add_argument(
        "--p1_source",
        choices=["cmotion", "output"],
        default="cmotion",
        help="Which field to use for P1",
    )
    parser.add_argument(
        "--p2_source",
        choices=["cmotion", "output"],
        default="output",
        help="Which field to use for P2",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only convert first N samples")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clips")
    parser.add_argument(
        "--shape_mode",
        choices=["canonical", "restored", "restored_shape_height"],
        default="canonical",
        help="Use neutral, restored shape, or restored shape with raw height alignment.",
    )
    parser.add_argument(
        "--meta_path",
        default="",
        help="Optional path to results_meta.npz (default: alongside results.npy).",
    )
    parser.add_argument(
        "--raw_motions_root",
        default=default_motions_root,
        help="Path to Inter-X raw motions (for restored shape).",
    )
    parser.add_argument(
        "--interaction_order",
        default=default_order,
        help="Path to interaction_order.pkl (for actor/reactor mapping).",
    )
    parser.add_argument(
        "--dataset",
        choices=["auto", "interx", "chi3d"],
        default="auto",
        help="Apply dataset-specific alignment rules (auto inferred from run name)",
    )
    parser.add_argument("--flip_x", action="store_true", help="Rotate 180° around X axis (upside-down fix)")
    parser.add_argument("--flip_y", action="store_true", help="Rotate 180° around Y axis")
    parser.add_argument("--flip_z", action="store_true", help="Rotate 180° around Z axis")
    args = parser.parse_args()

    if args.runs:
        run_names = args.runs
    else:
        run_names = [
            name for name in sorted(os.listdir(args.outputs_root))
            if os.path.isdir(os.path.join(args.outputs_root, name))
        ]

    for run_name in run_names:
        run_dir = os.path.join(args.outputs_root, run_name)
        results_path = os.path.join(run_dir, "results.npy")
        if not os.path.exists(results_path):
            continue
        out_dir = os.path.join(run_dir, "motions")
        convert_one_run(results_path, out_dir, args, run_name)


if __name__ == "__main__":
    main()
