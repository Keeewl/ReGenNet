#!/usr/bin/env python3
"""Convert processed Inter-X h5 clips into viewer-ready SMPL-X motions."""

from __future__ import annotations

import argparse
import os
import pickle
import re

import h5py
import numpy as np


INTERX_ACTION_RE = re.compile(r"A(\d+)")


def _normalize_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _normalize_str(value.item())
    return str(value)


def _load_action_names(path: str) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_action_id(dataset_key: str) -> int:
    match = INTERX_ACTION_RE.search(dataset_key)
    if not match:
        return -1
    return int(match.group(1))


def _load_interaction_order(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_actor_is_p1(order_dict: dict, dataset_key: str) -> int:
    label = order_dict.get(dataset_key, None)
    if label is None:
        return -1
    return 1 if int(label) == 1 else 0


def _load_raw_person(motions_root: str, dataset_key: str, person: str):
    path = os.path.join(motions_root, dataset_key, f"{person}.npz")
    return np.load(path, allow_pickle=True)


def _load_raw_shapes(motions_root: str, dataset_key: str):
    p1 = _load_raw_person(motions_root, dataset_key, "P1")
    p2 = _load_raw_person(motions_root, dataset_key, "P2")
    return (
        np.asarray(p1["betas"]).astype(np.float32),
        _normalize_str(p1["gender"]),
        np.asarray(p2["betas"]).astype(np.float32),
        _normalize_str(p2["gender"]),
    )


def _load_raw_trans(motions_root: str, dataset_key: str):
    p1 = _load_raw_person(motions_root, dataset_key, "P1")
    p2 = _load_raw_person(motions_root, dataset_key, "P2")
    return (
        np.asarray(p1["trans"]).astype(np.float32),
        np.asarray(p2["trans"]).astype(np.float32),
    )


def _infer_person_order(h5_path: str, requested: str) -> str:
    if requested != "auto":
        return requested
    name = os.path.basename(h5_path)
    if name in {"train.h5", "val.h5", "test.h5", "inter-x_regen.h5"}:
        return "actor_reactor"
    return "raw"


def _select_shapes_for_roles(actor_is_p1: int, betas_p1, gender_p1, betas_p2, gender_p2):
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


def _select_trans_for_roles(actor_is_p1: int, trans_p1, trans_p2):
    if actor_is_p1 == 1:
        actor = trans_p1
        reactor = trans_p2
    elif actor_is_p1 == 0:
        actor = trans_p2
        reactor = trans_p1
    else:
        actor = None
        reactor = None
    return actor, reactor


def _downsample_raw_trans(trans_seq: np.ndarray, length: int, downsample: int) -> np.ndarray:
    raw_indices = np.arange(length, dtype=np.int64) * downsample
    raw_indices = np.clip(raw_indices, 0, trans_seq.shape[0] - 1)
    trans = trans_seq[raw_indices]
    if trans.shape[0] < length:
        pad = np.repeat(trans[-1:], length - trans.shape[0], axis=0)
        trans = np.concatenate([trans, pad], axis=0)
    return trans[:length]


def build_params(
    person_data: np.ndarray,
    dataset_key: str,
    role: str,
    person_order: str,
    action_name: str,
    downsample: int,
    betas: np.ndarray | None = None,
    gender: str | None = None,
    trans_override: np.ndarray | None = None,
) -> dict:
    length = person_data.shape[0]
    action_id = _parse_action_id(dataset_key)
    processed_frame_ix = np.arange(length, dtype=np.int64)
    raw_frame_ix = processed_frame_ix * downsample

    trans = person_data[:, 55, :].astype(np.float32) if trans_override is None else np.asarray(trans_override, dtype=np.float32)

    return {
        "root_orient": person_data[:, 0, :].astype(np.float32),
        "pose_body": person_data[:, 1:22, :].astype(np.float32),
        "pose_lhand": person_data[:, 25:40, :].astype(np.float32),
        "pose_rhand": person_data[:, 40:55, :].astype(np.float32),
        "trans": trans[:length],
        "betas": np.zeros(10, dtype=np.float32) if betas is None else betas,
        "gender": "neutral" if gender is None else gender,
        "dataset_key": dataset_key,
        "action_id": int(action_id),
        "action_name": action_name,
        "source_role": role,
        "person_order": person_order,
        "processed_nframes": int(length),
        "raw_nframes": int(length * downsample),
        "processed_fps": 30,
        "raw_fps": 30 * downsample,
        "downsample": int(downsample),
        "frame_ix": processed_frame_ix,
        "raw_frame_ix": raw_frame_ix,
        "start_frame": int(processed_frame_ix[0]) if length else -1,
        "end_frame": int(processed_frame_ix[-1]) if length else -1,
    }


def convert_h5(h5_path: str, out_dir: str, args: argparse.Namespace) -> None:
    os.makedirs(out_dir, exist_ok=True)
    order_dict = _load_interaction_order(args.interaction_order)
    action_names = _load_action_names(args.action_file)
    person_order = _infer_person_order(h5_path, args.person_order)

    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys())
        limit = len(keys) if args.limit is None else min(args.limit, len(keys))
        for dataset_key in keys[:limit]:
            clip_dir = os.path.join(out_dir, dataset_key)
            os.makedirs(clip_dir, exist_ok=True)
            p1_path = os.path.join(clip_dir, "P1.npz")
            p2_path = os.path.join(clip_dir, "P2.npz")
            if not args.overwrite and os.path.exists(p1_path) and os.path.exists(p2_path):
                continue

            clip = f[dataset_key][:].astype(np.float32)
            p1_data = clip[:, :, 0:3]
            p2_data = clip[:, :, 3:6]
            length = clip.shape[0]

            action_id = _parse_action_id(dataset_key)
            action_name = action_names[action_id] if 0 <= action_id < len(action_names) else f"action_{action_id}"
            actor_is_p1 = _resolve_actor_is_p1(order_dict, dataset_key)

            p1_betas = None
            p1_gender = None
            p2_betas = None
            p2_gender = None
            trans_p1 = None
            trans_p2 = None

            if args.shape_mode in {"restored", "restored_shape_height"}:
                betas_p1, gender_p1, betas_p2, gender_p2 = _load_raw_shapes(args.raw_motions_root, dataset_key)
                if person_order == "actor_reactor":
                    actor_shape, reactor_shape = _select_shapes_for_roles(
                        actor_is_p1, betas_p1, gender_p1, betas_p2, gender_p2
                    )
                    p1_betas, p1_gender = actor_shape
                    p2_betas, p2_gender = reactor_shape
                else:
                    p1_betas, p1_gender = betas_p1, gender_p1
                    p2_betas, p2_gender = betas_p2, gender_p2

            if args.shape_mode == "restored_shape_height":
                raw_trans_p1, raw_trans_p2 = _load_raw_trans(args.raw_motions_root, dataset_key)
                ds_p1 = _downsample_raw_trans(raw_trans_p1, length, args.downsample)
                ds_p2 = _downsample_raw_trans(raw_trans_p2, length, args.downsample)
                if person_order == "actor_reactor":
                    actor_trans, reactor_trans = _select_trans_for_roles(actor_is_p1, ds_p1, ds_p2)
                    trans_p1 = actor_trans
                    trans_p2 = reactor_trans
                else:
                    trans_p1 = ds_p1
                    trans_p2 = ds_p2

            if person_order == "actor_reactor":
                p1_role = "actor"
                p2_role = "reactor"
            else:
                if actor_is_p1 == 1:
                    p1_role, p2_role = "actor", "reactor"
                elif actor_is_p1 == 0:
                    p1_role, p2_role = "reactor", "actor"
                else:
                    p1_role, p2_role = "P1", "P2"

            p1_params = build_params(
                p1_data,
                dataset_key=dataset_key,
                role=p1_role,
                person_order=person_order,
                action_name=action_name,
                downsample=args.downsample,
                betas=p1_betas,
                gender=p1_gender,
                trans_override=trans_p1,
            )
            p2_params = build_params(
                p2_data,
                dataset_key=dataset_key,
                role=p2_role,
                person_order=person_order,
                action_name=action_name,
                downsample=args.downsample,
                betas=p2_betas,
                gender=p2_gender,
                trans_override=trans_p2,
            )

            np.savez(p1_path, **p1_params)
            np.savez(p2_path, **p2_params)

    print(f"Done. Converted {limit} clips from {h5_path} to {out_dir}")


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sibling_interx_root = os.path.join(os.path.dirname(repo_root), 'Inter-X')
    default_motions_root = os.path.join(sibling_interx_root, 'datasets', 'interx', 'motions')
    default_order = os.path.join(repo_root, 'dataset', 'interx', 'annots', 'interaction_order.pkl')
    if not os.path.exists(default_order):
        default_order = os.path.join(sibling_interx_root, 'datasets', 'interx', 'annots', 'interaction_order.pkl')
    default_action_file = os.path.join(repo_root, 'dataset', 'interx', 'annots', 'action_setting.txt')
    if not os.path.exists(default_action_file):
        default_action_file = os.path.join(sibling_interx_root, 'datasets', 'interx', 'annots', 'action_setting.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", required=True, help="Path to processed Inter-X h5 file")
    parser.add_argument("--output_dir", required=True, help="Directory to write viewer-ready motions")
    parser.add_argument("--limit", type=int, default=None, help="Only convert first N clips")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clips")
    parser.add_argument(
        "--shape_mode",
        choices=["canonical", "restored", "restored_shape_height"],
        default="canonical",
        help="Use neutral, restored shape, or restored shape with raw height alignment.",
    )
    parser.add_argument(
        "--person_order",
        choices=["auto", "raw", "actor_reactor"],
        default="auto",
        help="Whether h5 stores raw P1/P2 order or actor/reactor order.",
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
        "--action_file",
        default=default_action_file,
        help="Path to action_setting.txt (for readable action names).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=4,
        help="Raw-to-processed downsample ratio used by preprocess.",
    )
    args = parser.parse_args()

    convert_h5(args.h5_path, args.output_dir, args)


if __name__ == "__main__":
    main()
