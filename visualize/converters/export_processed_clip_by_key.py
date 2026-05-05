#!/usr/bin/env python3
"""Export one processed Inter-X clip by dataset_key into viewer-ready motions."""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

from data_loaders.a2m.feeder import _load_restoration_package

from visualize.converters.convert_processed_h5_to_motions import (
    _downsample_raw_trans,
    _load_action_names,
    _load_interaction_order,
    _load_raw_shapes,
    _load_raw_trans,
    _parse_action_id,
    _resolve_actor_is_p1,
    _select_shapes_for_roles,
    _select_trans_for_roles,
    build_params,
)


GENDER_ID_TO_NAME = {0: "neutral", 1: "male", 2: "female"}


def _restoration_record(args: argparse.Namespace):
    if not getattr(args, "restoration_meta_path", ""):
        return None
    payload, key_to_index = _load_restoration_package(args.restoration_meta_path)
    idx = key_to_index.get(args.dataset_key, None)
    if idx is None:
        return None
    return payload, int(idx)


def _export_one_from_h5(h5_path: str, output_dir: str, args: argparse.Namespace) -> bool:
    if not h5_path or not os.path.exists(h5_path):
        return False

    order_dict = _load_interaction_order(args.interaction_order)
    action_names = _load_action_names(args.action_file)

    with h5py.File(h5_path, "r") as f:
        if args.dataset_key not in f:
            return False

        clip = f[args.dataset_key][:].astype("float32")
        p1_data = clip[:, :, 0:3]
        p2_data = clip[:, :, 3:6]
        length = clip.shape[0]

        action_id = _parse_action_id(args.dataset_key)
        action_name = action_names[action_id] if 0 <= action_id < len(action_names) else f"action_{action_id}"
        actor_is_p1 = _resolve_actor_is_p1(order_dict, args.dataset_key)

        p1_betas = p1_gender = p2_betas = p2_gender = None
        trans_p1 = trans_p2 = None
        restoration = _restoration_record(args)

        if args.shape_mode in {"restored", "restored_shape_height"}:
            if restoration is not None:
                payload, idx = restoration
                betas_p1 = np.asarray(payload["p1_betas"][idx], dtype=np.float32).reshape(-1)
                betas_p2 = np.asarray(payload["p2_betas"][idx], dtype=np.float32).reshape(-1)
                gender_p1 = GENDER_ID_TO_NAME.get(int(np.asarray(payload["p1_gender_id"][idx]).item()), "neutral")
                gender_p2 = GENDER_ID_TO_NAME.get(int(np.asarray(payload["p2_gender_id"][idx]).item()), "neutral")
            else:
                betas_p1, gender_p1, betas_p2, gender_p2 = _load_raw_shapes(args.raw_motions_root, args.dataset_key)
            actor_shape, reactor_shape = _select_shapes_for_roles(
                actor_is_p1, betas_p1, gender_p1, betas_p2, gender_p2
            )
            p1_betas, p1_gender = actor_shape
            p2_betas, p2_gender = reactor_shape

        if args.shape_mode == "restored_shape_height":
            if restoration is not None:
                payload, idx = restoration
                raw_trans_p1 = np.asarray(payload["p1_trans"][idx], dtype=np.float32)
                raw_trans_p2 = np.asarray(payload["p2_trans"][idx], dtype=np.float32)
                downsample = int(np.asarray(payload["downsample"][idx]).item()) if "downsample" in payload else args.downsample
                ds_p1 = _downsample_raw_trans(raw_trans_p1, length, downsample)
                ds_p2 = _downsample_raw_trans(raw_trans_p2, length, downsample)
            else:
                raw_trans_p1, raw_trans_p2 = _load_raw_trans(args.raw_motions_root, args.dataset_key)
                ds_p1 = _downsample_raw_trans(raw_trans_p1, length, args.downsample)
                ds_p2 = _downsample_raw_trans(raw_trans_p2, length, args.downsample)
            actor_trans, reactor_trans = _select_trans_for_roles(actor_is_p1, ds_p1, ds_p2)
            trans_p1 = actor_trans
            trans_p2 = reactor_trans

        p1_params = build_params(
            p1_data,
            dataset_key=args.dataset_key,
            role="actor",
            person_order="actor_reactor",
            action_name=action_name,
            downsample=args.downsample,
            betas=p1_betas,
            gender=p1_gender,
            trans_override=trans_p1,
        )
        p2_params = build_params(
            p2_data,
            dataset_key=args.dataset_key,
            role="reactor",
            person_order="actor_reactor",
            action_name=action_name,
            downsample=args.downsample,
            betas=p2_betas,
            gender=p2_gender,
            trans_override=trans_p2,
        )

        clip_dir = os.path.join(output_dir, args.dataset_key)
        os.makedirs(clip_dir, exist_ok=True)
        import numpy as np

        np.savez(os.path.join(clip_dir, "P1.npz"), **p1_params)
        np.savez(os.path.join(clip_dir, "P2.npz"), **p2_params)
        return True


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_key", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_h5", default=os.path.join(repo_root, "dataset", "interx", "regen", "train.h5"))
    parser.add_argument("--val_h5", default=os.path.join(repo_root, "dataset", "interx", "regen", "val.h5"))
    parser.add_argument("--test_h5", default=os.path.join(repo_root, "dataset", "interx", "regen", "test.h5"))
    parser.add_argument(
        "--shape_mode",
        choices=["canonical", "restored", "restored_shape_height"],
        default="restored_shape_height",
    )
    parser.add_argument("--raw_motions_root", required=True)
    parser.add_argument(
        "--restoration_meta_path",
        default=os.path.join(repo_root, "dataset", "interx", "cache", "interx_restoration_meta.npz"),
    )
    parser.add_argument(
        "--interaction_order",
        default=os.path.join(repo_root, "dataset", "interx", "annots", "interaction_order.pkl"),
    )
    parser.add_argument(
        "--action_file",
        default=os.path.join(repo_root, "dataset", "interx", "annots", "action_setting.txt"),
    )
    parser.add_argument("--downsample", type=int, default=4)
    args = parser.parse_args()

    for h5_path in (args.train_h5, args.val_h5, args.test_h5):
        if _export_one_from_h5(h5_path, args.output_dir, args):
            print(f"saved GT clip from {h5_path}: {os.path.join(args.output_dir, args.dataset_key)}")
            return

    raise KeyError(f"dataset_key not found in train/val/test regen h5: {args.dataset_key}")


if __name__ == "__main__":
    main()
