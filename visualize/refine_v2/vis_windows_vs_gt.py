"""Text inspection for refine_v2 selector windows against GT contact labels."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from refine_v2.schema import object_array_to_records, to_jsonable


def _load_npz(path: str):
    return np.load(path, allow_pickle=True)


def _select_row(labels, *, sample_index: int | None, dataset_key: str | None) -> tuple[int, int]:
    if sample_index is not None:
        sample_indices = np.asarray(labels["sample_indices"]).reshape(-1)
        hits = np.where(sample_indices == int(sample_index))[0]
        if hits.size == 0:
            raise KeyError(f"sample_index not found: {sample_index}")
        idx = int(hits[0])
        return idx, int(np.asarray(labels["dataset_row_indices"])[idx])
    if dataset_key is not None:
        keys = [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in np.asarray(labels["dataset_key"], dtype=object).tolist()]
        if dataset_key not in keys:
            raise KeyError(f"dataset_key not found: {dataset_key}")
        idx = keys.index(dataset_key)
        return idx, int(np.asarray(labels["dataset_row_indices"])[idx])
    return 0, int(np.asarray(labels["dataset_row_indices"])[0])


def _overlap(a_start, a_end, b_start, b_end):
    return max(0, min(int(a_end), int(b_end)) - max(int(a_start), int(b_start)))


def _timeline(length: int, gt_segments: list[dict], windows: list[dict], width: int = 100) -> list[str]:
    width = max(10, int(width))
    length = max(1, int(length))

    def scale(frame):
        return min(width - 1, int((max(0, min(length, int(frame))) / length) * width))

    lines = []
    for item in gt_segments:
        chars = ["."] * width
        for pos in range(scale(item["raw_start_frame"]), max(scale(item["raw_end_frame"]), scale(item["raw_start_frame"]) + 1)):
            if 0 <= pos < width:
                chars[pos] = "#"
        lines.append(f"GT  {item['hand_side']}->{item['target_region']:<11} {''.join(chars)} [{item['raw_start_frame']},{item['raw_end_frame']})")
    for item in windows:
        chars = ["."] * width
        for pos in range(scale(item["start_frame"]), max(scale(item["end_frame"]), scale(item["start_frame"]) + 1)):
            if 0 <= pos < width:
                chars[pos] = "W"
        flag = "FP" if item.get("is_false_positive") else "OK"
        lines.append(f"WIN {item['hand_side']}->{item['target_region']:<11} {''.join(chars)} [{item['start_frame']},{item['end_frame']}) {flag}")
    return lines


def inspect_windows_vs_gt(
    contact_labels_path: str,
    selector_windows_path: str,
    audit_json_path: str,
    *,
    sample_index: int | None = None,
    dataset_key: str | None = None,
    timeline_width: int = 100,
):
    labels = _load_npz(contact_labels_path)
    windows_pack = _load_npz(selector_windows_path)
    with open(audit_json_path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    label_index, row = _select_row(labels, sample_index=sample_index, dataset_key=dataset_key)
    length = int(np.asarray(labels["lengths"])[label_index])
    sample_id = int(np.asarray(labels["sample_indices"])[label_index])
    key = str(np.asarray(labels["dataset_key"], dtype=object)[label_index])

    gt_segments = [
        item for item in object_array_to_records(labels["segments"])
        if int(item["dataset_row_index"]) == row
    ]
    pred_windows = [
        item for item in object_array_to_records(windows_pack["windows"])
        if int(item["dataset_row_index"]) == row
    ]
    per_window_debug = [
        item for item in audit.get("per_window", [])
        if int(item["window"]["dataset_row_index"]) == row
    ]
    debug_by_window = {
        (
            int(item["window"]["hand_side_id"]),
            int(item["window"]["target_region_id"]),
            int(item["window"]["start_frame"]),
            int(item["window"]["end_frame"]),
        ): item
        for item in per_window_debug
    }
    enriched_windows = []
    for window in pred_windows:
        key_tuple = (
            int(window["hand_side_id"]),
            int(window["target_region_id"]),
            int(window["start_frame"]),
            int(window["end_frame"]),
        )
        debug = debug_by_window.get(key_tuple, {})
        item = dict(window)
        item["is_false_positive"] = bool(debug.get("is_false_positive", False))
        item["best_overlap"] = int(debug.get("best_overlap", 0))
        item["window_contact_purity"] = float(debug.get("window_contact_purity", 0.0))
        item["matched_gt_segment"] = debug.get("matched_gt_segment")
        enriched_windows.append(item)

    overlaps = []
    for window in enriched_windows:
        row_overlaps = []
        for gt in gt_segments:
            if int(gt["hand_side_id"]) != int(window["hand_side_id"]):
                continue
            ov = _overlap(window["start_frame"], window["end_frame"], gt["raw_start_frame"], gt["raw_end_frame"])
            if ov > 0:
                row_overlaps.append(
                    {
                        "window": window,
                        "gt_segment": gt,
                        "overlap": int(ov),
                        "same_region": bool(int(gt["target_region_id"]) == int(window["target_region_id"])),
                    }
                )
        overlaps.extend(row_overlaps)

    return {
        "sample_index": sample_id,
        "dataset_row_index": row,
        "dataset_key": key,
        "length": length,
        "gt_segments": gt_segments,
        "predicted_windows": enriched_windows,
        "overlaps": overlaps,
        "timeline": _timeline(length, gt_segments, enriched_windows, width=timeline_width),
        "audit_metrics": audit.get("metrics", {}),
    }


def _print_report(report):
    print(f"sample_index: {report['sample_index']}  dataset_row_index: {report['dataset_row_index']}")
    print(f"dataset_key: {report['dataset_key']}  length: {report['length']}")
    print("\nGT segments:")
    if not report["gt_segments"]:
        print("  -")
    for item in report["gt_segments"]:
        print(f"  {item['hand_side']}->{item['target_region']} [{item['raw_start_frame']},{item['raw_end_frame']}) len={item['raw_length']}")
    print("\npredicted windows:")
    if not report["predicted_windows"]:
        print("  -")
    for item in report["predicted_windows"]:
        flag = "FP" if item["is_false_positive"] else "OK"
        print(
            f"  {item['hand_side']}->{item['target_region']} "
            f"raw=[{item['raw_start_frame']},{item['raw_end_frame']}) "
            f"win=[{item['start_frame']},{item['end_frame']}) "
            f"purity={item['window_contact_purity']:.3f} overlap={item['best_overlap']} {flag}"
        )
    print("\ntimeline:")
    for line in report["timeline"]:
        print("  " + line)


def build_parser():
    parser = argparse.ArgumentParser(description="Inspect refine_v2 selector windows vs GT labels.")
    parser.add_argument("--contact_labels_path", required=True, type=str)
    parser.add_argument("--selector_windows_path", required=True, type=str)
    parser.add_argument("--audit_json", required=True, type=str)
    parser.add_argument("--sample_index", default=None, type=int)
    parser.add_argument("--dataset_key", default=None, type=str)
    parser.add_argument("--timeline_width", default=100, type=int)
    parser.add_argument("--output_json", default="", type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = inspect_windows_vs_gt(
        args.contact_labels_path,
        args.selector_windows_path,
        args.audit_json,
        sample_index=args.sample_index,
        dataset_key=args.dataset_key,
        timeline_width=args.timeline_width,
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
