"""CLI: inspect refine_v2 refiner window data."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from refine_v2.data.schema import TARGET_REGION_NAMES, to_jsonable
from refine_v2.refiner_data.window_dataset import RefineV2WindowDataset


def _array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    if arr.size == 0:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "min": None, "max": None, "mean": None}
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _region_ratios(mask: np.ndarray) -> dict[str, float]:
    arr = np.asarray(mask, dtype=np.float32)
    return {
        str(name): float(arr[idx].mean()) if arr.shape[-1] > 0 else 0.0
        for idx, name in enumerate(TARGET_REGION_NAMES)
    }


def _region_dist_summary(dist: np.ndarray) -> dict[str, dict[str, float]]:
    arr = np.asarray(dist, dtype=np.float32)
    out = {}
    for idx, name in enumerate(TARGET_REGION_NAMES):
        row = arr[idx]
        out[str(name)] = {
            "min": float(np.min(row)) if row.size else 0.0,
            "mean": float(np.mean(row)) if row.size else 0.0,
            "max": float(np.max(row)) if row.size else 0.0,
        }
    return out


def summarize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    primary_id = int(sample["primary_target_region_id"])
    coarse_ratios = _region_ratios(sample["coarse_region_contact_mask_window"])
    gt_ratios = _region_ratios(sample["gt_region_contact_mask_window"])
    return {
        "window_index": int(sample["window_index"]),
        "sequence_window_index": int(sample["sequence_window_index"]),
        "dataset_row_index": int(sample["dataset_row_index"]),
        "sample_index": int(sample["sample_index"]),
        "dataset_key": sample["dataset_key"],
        "action_type": sample["action_type"],
        "action_label": sample["action_label"],
        "action_name": sample["action_name"],
        "bucket_label": sample["bucket_label"],
        "hand_side": sample["hand_side"],
        "hand_side_id": int(sample["hand_side_id"]),
        "start_frame": int(sample["start_frame"]),
        "end_frame": int(sample["end_frame"]),
        "raw_start_frame": int(sample["raw_start_frame"]),
        "raw_end_frame": int(sample["raw_end_frame"]),
        "window_length": int(sample["window_length"]),
        "primary_target_region": sample["primary_target_region"],
        "primary_target_region_id": primary_id,
        "topk_target_regions": sample["topk_target_regions"],
        "topk_target_region_ids": [int(x) for x in np.asarray(sample["topk_target_region_ids"]).reshape(-1).tolist()],
        "topk_region_scores_numeric": np.asarray(sample["topk_region_scores_numeric"], dtype=np.float32).tolist(),
        "motion_shapes": {
            "actor_motion_window": list(sample["actor_motion_window"].shape),
            "coarse_motion_window": list(sample["coarse_motion_window"].shape),
            "gt_motion_window": list(sample["gt_motion_window"].shape),
        },
        "contact_shapes": {
            "coarse_region_contact_mask_window": list(sample["coarse_region_contact_mask_window"].shape),
            "coarse_min_region_dist_window": list(sample["coarse_min_region_dist_window"].shape),
            "gt_region_contact_mask_window": list(sample["gt_region_contact_mask_window"].shape),
            "gt_min_region_dist_window": list(sample["gt_min_region_dist_window"].shape),
        },
        "coarse_contact_ratios_by_region": coarse_ratios,
        "gt_contact_ratios_by_region": gt_ratios,
        "primary_region_contact_ratios": {
            "coarse": float(coarse_ratios[TARGET_REGION_NAMES[primary_id]]),
            "gt": float(gt_ratios[TARGET_REGION_NAMES[primary_id]]),
        },
        "coarse_min_region_dist_summary": _region_dist_summary(sample["coarse_min_region_dist_window"]),
        "gt_min_region_dist_summary": _region_dist_summary(sample["gt_min_region_dist_window"]),
        "valid_mask": _array_summary(sample["valid_mask"]),
    }


def _print_summary(summary: dict[str, Any]):
    print("refine_v2 refiner window dataset summary")
    for key in [
        "reaction_data_path",
        "contact_labels_path",
        "subset_manifest_path",
        "selector_windows_path",
        "space_definition",
        "include_buckets",
        "selected_action_types",
        "num_sequences",
        "num_windows",
    ]:
        print(f"{key}: {summary.get(key)}")
    for key in [
        "action_type_distribution",
        "sequence_action_type_distribution",
        "window_action_type_distribution",
        "bucket_distribution",
        "hand_side_distribution",
        "primary_region_distribution",
        "topk_region_distribution",
    ]:
        print(f"\n{key}:")
        for name, count in summary.get(key, {}).items():
            print(f"  {name}: {count}")
    print("\nshapes:")
    for group in ("motion_shapes", "contact_condition_shapes", "gt_supervision_shapes"):
        print(f"  {group}:")
        for key, value in summary.get(group, {}).items():
            print(f"    {key}: {value}")


def _print_sample(report: dict[str, Any]):
    print("refine_v2 refiner window sample")
    for key in [
        "window_index",
        "sequence_window_index",
        "dataset_row_index",
        "sample_index",
        "dataset_key",
        "action_type",
        "bucket_label",
        "hand_side",
        "start_frame",
        "end_frame",
        "raw_start_frame",
        "raw_end_frame",
        "window_length",
        "primary_target_region",
        "topk_target_regions",
    ]:
        print(f"{key}: {report.get(key)}")
    print("\nmotion_shapes:")
    for key, value in report["motion_shapes"].items():
        print(f"  {key}: {value}")
    print("\ncontact_shapes:")
    for key, value in report["contact_shapes"].items():
        print(f"  {key}: {value}")
    print("\nprimary_region_contact_ratios:")
    for key, value in report["primary_region_contact_ratios"].items():
        print(f"  {key}: {value:.6f}")
    print("\ncoarse_contact_ratios_by_region:")
    for key, value in report["coarse_contact_ratios_by_region"].items():
        print(f"  {key}: {value:.6f}")
    print("\ngt_contact_ratios_by_region:")
    for key, value in report["gt_contact_ratios_by_region"].items():
        print(f"  {key}: {value:.6f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Inspect refine_v2 refiner window data.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--contact_labels_path", required=True, type=str)
    parser.add_argument("--subset_manifest_path", required=True, type=str)
    parser.add_argument("--selector_windows_path", required=True, type=str)
    parser.add_argument("--include_buckets", nargs="*", default=["GT+ / Pred+"])
    parser.add_argument("--selected_action_types", nargs="*", default=None)
    parser.add_argument("--window_index", default=None, type=int)
    parser.add_argument("--dataset_row_index", default=None, type=int)
    parser.add_argument("--start_frame", default=None, type=int)
    parser.add_argument("--hand_side", default="", choices=["", "left", "right"])
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--include_xyz", action="store_true")
    parser.add_argument("--no_strict_checks", action="store_true")
    parser.add_argument("--output_json", default="", type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = RefineV2WindowDataset(
        args.reaction_data_path,
        args.contact_labels_path,
        args.subset_manifest_path,
        args.selector_windows_path,
        include_buckets=args.include_buckets,
        selected_action_types=args.selected_action_types,
        include_xyz=args.include_xyz,
        strict_checks=not args.no_strict_checks,
    )
    if args.summary_only:
        payload = {"summary": dataset.summary()}
        _print_summary(payload["summary"])
    else:
        if args.window_index is not None:
            idx = int(args.window_index)
        else:
            idx = dataset.find_window_index(
                dataset_row_index=args.dataset_row_index,
                start_frame=args.start_frame,
                hand_side=args.hand_side,
            )
        sample = dataset[idx]
        payload = {"summary": dataset.summary(), "sample": summarize_sample(sample)}
        _print_summary(payload["summary"])
        print("")
        _print_sample(payload["sample"])

    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
        print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
