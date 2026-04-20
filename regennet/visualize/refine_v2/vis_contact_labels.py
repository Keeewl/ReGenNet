"""Text inspection for refine_v2 GT contact labels."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from regennet.refine_v2.schema import (
    HAND_SIDE_NAMES,
    TARGET_REGION_NAMES,
    object_array_to_records,
    to_jsonable,
)


def _load(path: str):
    return np.load(path, allow_pickle=True)


def _select_index(data, *, sample_index: int | None, dataset_key: str | None) -> int:
    if sample_index is not None:
        sample_indices = np.asarray(data["sample_indices"]).reshape(-1)
        hits = np.where(sample_indices == int(sample_index))[0]
        if hits.size == 0:
            raise KeyError(f"sample_index not found: {sample_index}")
        return int(hits[0])
    if dataset_key is not None:
        keys = [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in np.asarray(data["dataset_key"], dtype=object).tolist()]
        if dataset_key not in keys:
            raise KeyError(f"dataset_key not found: {dataset_key}")
        return keys.index(dataset_key)
    return 0


def _segments_for_mask(mask_1d):
    mask_1d = np.asarray(mask_1d).astype(bool)
    out = []
    idx = 0
    while idx < mask_1d.size:
        if not mask_1d[idx]:
            idx += 1
            continue
        start = idx
        idx += 1
        while idx < mask_1d.size and mask_1d[idx]:
            idx += 1
        out.append((start, idx))
    return out


def inspect_contact_labels(path: str, *, sample_index: int | None = None, dataset_key: str | None = None):
    data = _load(path)
    idx = _select_index(data, sample_index=sample_index, dataset_key=dataset_key)
    length = int(np.asarray(data["lengths"])[idx])
    sample_id = int(np.asarray(data["sample_indices"])[idx])
    key = str(np.asarray(data["dataset_key"], dtype=object)[idx])
    mask = np.asarray(data["gt_contact_mask"], dtype=np.uint8)[idx, :, :, :length]
    all_segments = object_array_to_records(data["segments"])
    row = int(np.asarray(data["dataset_row_indices"])[idx])
    segments = [item for item in all_segments if int(item["dataset_row_index"]) == row]

    grid = {}
    for hand_id, hand_name in enumerate(HAND_SIDE_NAMES):
        grid[hand_name] = {}
        for region_id, region_name in enumerate(TARGET_REGION_NAMES):
            grid[hand_name][region_name] = [
                {"start": int(start), "end": int(end), "length": int(end - start)}
                for start, end in _segments_for_mask(mask[hand_id, region_id])
            ]
    return {
        "path": path,
        "dataset_row_index": row,
        "sample_index": sample_id,
        "dataset_key": key,
        "length": length,
        "contact_runs_by_hand_region": grid,
        "gt_raw_segments": segments,
    }


def _print_report(report):
    print(f"path: {report['path']}")
    print(f"sample_index: {report['sample_index']}  dataset_row_index: {report['dataset_row_index']}")
    print(f"dataset_key: {report['dataset_key']}  length: {report['length']}")
    print("\ncontact runs by hand/region:")
    for hand_name, by_region in report["contact_runs_by_hand_region"].items():
        print(f"  {hand_name}:")
        for region_name, runs in by_region.items():
            text = ", ".join(f"[{r['start']},{r['end']})" for r in runs) if runs else "-"
            print(f"    {region_name}: {text}")
    print("\nGT raw segments:")
    if not report["gt_raw_segments"]:
        print("  -")
    for item in report["gt_raw_segments"]:
        print(
            "  "
            f"{item['hand_side']}->{item['target_region']} "
            f"[{item['raw_start_frame']},{item['raw_end_frame']}) "
            f"len={item['raw_length']} center={item['center_frame']}"
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Inspect refine_v2 GT contact labels.")
    parser.add_argument("--contact_labels_path", required=True, type=str)
    parser.add_argument("--sample_index", default=None, type=int)
    parser.add_argument("--dataset_key", default=None, type=str)
    parser.add_argument("--output_json", default="", type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = inspect_contact_labels(
        args.contact_labels_path,
        sample_index=args.sample_index,
        dataset_key=args.dataset_key,
    )
    _print_report(report)
    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

