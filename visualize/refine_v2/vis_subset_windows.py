"""Text sanity checks for refine_v2 contact-subset selector windows."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from refine_v2.data.schema import object_array_to_records, to_jsonable


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_npz(path: str):
    return np.load(path, allow_pickle=True)


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(int(a_end), int(b_end)) - max(int(a_start), int(b_start)))


def _window_key(item: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(item["dataset_row_index"]),
        int(item["hand_side_id"]),
        int(item["start_frame"]),
        int(item["end_frame"]),
    )


def _load_window_metadata(path: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict) and "windows" in payload:
        return [dict(item) for item in payload["windows"]]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError(f"Unsupported subset window metadata format: {path}")


def _load_audit_by_window(path: str) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    audit = _load_json(path)
    out = {}
    for item in audit.get("per_window", []):
        window = item.get("window", {})
        if not window:
            continue
        out[_window_key(window)] = item
    return out


def _load_gt_segments_by_row(contact_labels_path: str) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int]]:
    if not contact_labels_path:
        return {}, {}
    labels = _load_npz(contact_labels_path)
    segments_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for segment in object_array_to_records(labels["segments"]):
        segments_by_row[int(segment["dataset_row_index"])].append(segment)
    lengths_by_row = {
        int(row): int(length)
        for row, length in zip(
            np.asarray(labels["dataset_row_indices"], dtype=np.int64).reshape(-1).tolist(),
            np.asarray(labels["lengths"], dtype=np.int64).reshape(-1).tolist(),
        )
    }
    return dict(segments_by_row), lengths_by_row


def _enrich_windows(
    metadata_windows: list[dict[str, Any]],
    audit_by_window: dict[tuple[int, int, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for window in metadata_windows:
        item = dict(window)
        debug = audit_by_window.get(_window_key(item), {})
        item["is_false_positive"] = bool(debug.get("is_false_positive", False))
        item["topk_matched"] = bool(debug.get("topk_matched", False))
        item["best_overlap"] = int(debug.get("best_overlap", 0))
        item["topk_best_overlap"] = int(debug.get("topk_best_overlap", 0))
        item["window_contact_purity"] = float(debug.get("window_contact_purity", item.get("window_contact_purity", 0.0)))
        item["matched_gt_segment"] = debug.get("matched_gt_segment")
        item["topk_best_gt_segment"] = debug.get("topk_best_gt_segment")
        item["best_same_hand_any_region_gt"] = debug.get("best_same_hand_any_region_gt")
        item["best_same_hand_any_region_overlap"] = int(debug.get("best_same_hand_any_region_overlap", 0))
        out.append(item)
    return out


def _apply_filters(
    windows: list[dict[str, Any]],
    *,
    action_types: list[str] | None,
    hand_side: str,
    region: str,
    bucket_label: str,
    fp_only: bool,
    tp_only: bool,
    topk_only: bool,
    topk_miss_only: bool,
    min_purity: float | None,
    max_purity: float | None,
) -> list[dict[str, Any]]:
    out = []
    action_set = {x.lower() for x in action_types or []}
    for item in windows:
        if action_set and str(item.get("action_type", "")).lower() not in action_set and str(item.get("action_name", "")).lower() not in action_set:
            continue
        if hand_side and str(item.get("hand_side", "")) != hand_side:
            continue
        if region:
            topk_regions = {str(x) for x in item.get("topk_target_regions", [])}
            if str(item.get("primary_target_region", "")) != region and region not in topk_regions:
                continue
        if bucket_label and str(item.get("bucket_label", "")) != bucket_label:
            continue
        if fp_only and not bool(item.get("is_false_positive", False)):
            continue
        if tp_only and bool(item.get("is_false_positive", False)):
            continue
        if topk_only and not bool(item.get("topk_matched", False)):
            continue
        if topk_miss_only and bool(item.get("topk_matched", False)):
            continue
        purity = float(item.get("window_contact_purity", 0.0))
        if min_purity is not None and purity < float(min_purity):
            continue
        if max_purity is not None and purity > float(max_purity):
            continue
        out.append(item)
    return out


def _sort_and_sample(windows: list[dict[str, Any]], *, sort_by: str, limit: int, seed: int) -> list[dict[str, Any]]:
    items = list(windows)
    if sort_by == "purity_desc":
        items.sort(key=lambda item: (-float(item.get("window_contact_purity", 0.0)), int(item["dataset_row_index"]), int(item["start_frame"])))
    elif sort_by == "purity_asc":
        items.sort(key=lambda item: (float(item.get("window_contact_purity", 0.0)), int(item["dataset_row_index"]), int(item["start_frame"])))
    elif sort_by == "fp_first":
        items.sort(key=lambda item: (not bool(item.get("is_false_positive", False)), -float(item.get("window_contact_purity", 0.0)), int(item["dataset_row_index"])))
    elif sort_by == "topk_miss_first":
        items.sort(key=lambda item: (bool(item.get("topk_matched", False)), -float(item.get("window_contact_purity", 0.0)), int(item["dataset_row_index"])))
    elif sort_by == "random":
        rng = random.Random(int(seed))
        rng.shuffle(items)
    else:
        items.sort(key=lambda item: (int(item["dataset_row_index"]), int(item["start_frame"]), int(item.get("hand_side_id", 0))))
    return items[: max(0, int(limit))]


def _scale_frame(frame: int, length: int, width: int) -> int:
    return min(width - 1, int((max(0, min(int(length), int(frame))) / max(int(length), 1)) * int(width)))


def _bar_for_interval(start: int, end: int, *, length: int, width: int, char: str) -> str:
    chars = ["."] * int(width)
    s = _scale_frame(start, length, width)
    e = max(_scale_frame(end, length, width), s + 1)
    for pos in range(s, min(e, width)):
        chars[pos] = char
    return "".join(chars)


def _timeline_for_window(
    window: dict[str, Any],
    gt_segments: list[dict[str, Any]],
    *,
    length: int,
    width: int,
) -> list[str]:
    lines = []
    same_hand_gt = [
        gt for gt in gt_segments
        if int(gt.get("hand_side_id", -1)) == int(window.get("hand_side_id", -2))
        and _overlap(window["start_frame"], window["end_frame"], gt["raw_start_frame"], gt["raw_end_frame"]) > 0
    ]
    for gt in sorted(same_hand_gt, key=lambda item: (int(item["raw_start_frame"]), int(item["target_region_id"]))):
        bar = _bar_for_interval(gt["raw_start_frame"], gt["raw_end_frame"], length=length, width=width, char="#")
        ov = _overlap(window["start_frame"], window["end_frame"], gt["raw_start_frame"], gt["raw_end_frame"])
        same_primary = int(gt["target_region_id"]) == int(window.get("primary_target_region_id", -1))
        in_topk = int(gt["target_region_id"]) in {int(x) for x in window.get("topk_target_region_ids", [])}
        flags = []
        if same_primary:
            flags.append("primary")
        if in_topk:
            flags.append("topk")
        flag_text = ",".join(flags) if flags else "region_miss"
        lines.append(
            f"GT  {gt['hand_side']}->{gt['target_region']:<18} {bar} "
            f"[{gt['raw_start_frame']},{gt['raw_end_frame']}) ov={ov} {flag_text}"
        )
    bar = _bar_for_interval(window["start_frame"], window["end_frame"], length=length, width=width, char="W")
    topk = ",".join(str(x) for x in window.get("topk_target_regions", [])[:3])
    fp = "FP" if window.get("is_false_positive") else "OK"
    topk_ok = "topkOK" if window.get("topk_matched") else "topkMISS"
    lines.append(
        f"WIN {window['hand_side']}->{window['primary_target_region']:<18} {bar} "
        f"[{window['start_frame']},{window['end_frame']}) {fp} {topk_ok} topk=[{topk}]"
    )
    return lines


def build_subset_window_report(
    subset_window_metadata_path: str,
    audit_json_path: str,
    *,
    contact_labels_path: str = "",
    action_types: list[str] | None = None,
    hand_side: str = "",
    region: str = "",
    bucket_label: str = "",
    fp_only: bool = False,
    tp_only: bool = False,
    topk_only: bool = False,
    topk_miss_only: bool = False,
    min_purity: float | None = None,
    max_purity: float | None = None,
    sort_by: str = "purity_desc",
    limit: int = 20,
    seed: int = 7,
    timeline_width: int = 100,
) -> dict[str, Any]:
    metadata = _load_window_metadata(subset_window_metadata_path)
    audit_by_window = _load_audit_by_window(audit_json_path)
    windows = _enrich_windows(metadata, audit_by_window)
    filtered = _apply_filters(
        windows,
        action_types=action_types,
        hand_side=hand_side,
        region=region,
        bucket_label=bucket_label,
        fp_only=fp_only,
        tp_only=tp_only,
        topk_only=topk_only,
        topk_miss_only=topk_miss_only,
        min_purity=min_purity,
        max_purity=max_purity,
    )
    selected = _sort_and_sample(filtered, sort_by=sort_by, limit=limit, seed=seed)
    gt_by_row, lengths_by_row = _load_gt_segments_by_row(contact_labels_path)

    action_counts = Counter(str(item.get("action_type", "")) for item in filtered)
    region_counts = Counter(str(item.get("primary_target_region", "")) for item in filtered)
    fp_count = int(sum(1 for item in filtered if bool(item.get("is_false_positive", False))))
    topk_count = int(sum(1 for item in filtered if bool(item.get("topk_matched", False))))
    purity_values = [float(item.get("window_contact_purity", 0.0)) for item in filtered]
    selected_reports = []
    for item in selected:
        row = int(item["dataset_row_index"])
        length = int(lengths_by_row.get(row, max(int(item.get("end_frame", 0)), 1)))
        selected_reports.append(
            {
                "window": item,
                "timeline": _timeline_for_window(
                    item,
                    gt_by_row.get(row, []),
                    length=length,
                    width=timeline_width,
                ),
            }
        )
    return {
        "subset_window_metadata_path": subset_window_metadata_path,
        "audit_json_path": audit_json_path,
        "contact_labels_path": contact_labels_path,
        "filters": {
            "action_types": action_types or [],
            "hand_side": hand_side,
            "region": region,
            "bucket_label": bucket_label,
            "fp_only": fp_only,
            "tp_only": tp_only,
            "topk_only": topk_only,
            "topk_miss_only": topk_miss_only,
            "min_purity": min_purity,
            "max_purity": max_purity,
            "sort_by": sort_by,
            "limit": int(limit),
            "seed": int(seed),
        },
        "summary": {
            "num_windows_total": int(len(windows)),
            "num_windows_filtered": int(len(filtered)),
            "num_windows_selected": int(len(selected)),
            "false_positive_count": fp_count,
            "false_positive_ratio": float(fp_count / max(len(filtered), 1)),
            "topk_matched_count": topk_count,
            "topk_matched_ratio": float(topk_count / max(len(filtered), 1)),
            "mean_window_contact_purity": float(np.mean(purity_values)) if purity_values else 0.0,
            "min_window_contact_purity": float(np.min(purity_values)) if purity_values else 0.0,
            "max_window_contact_purity": float(np.max(purity_values)) if purity_values else 0.0,
            "action_type_counts": dict(sorted(action_counts.items())),
            "primary_region_counts": dict(sorted(region_counts.items())),
        },
        "selected_windows": selected_reports,
    }


def _print_report(report: dict[str, Any]):
    summary = report["summary"]
    print("subset window sanity report")
    print(f"metadata: {report['subset_window_metadata_path']}")
    print(f"audit: {report['audit_json_path']}")
    print("")
    print("summary:")
    for key in [
        "num_windows_total",
        "num_windows_filtered",
        "num_windows_selected",
        "false_positive_count",
        "false_positive_ratio",
        "topk_matched_count",
        "topk_matched_ratio",
        "mean_window_contact_purity",
        "min_window_contact_purity",
        "max_window_contact_purity",
    ]:
        print(f"  {key}: {summary[key]}")
    print("")
    print("action_type_counts:")
    for key, value in sorted(summary["action_type_counts"].items(), key=lambda item: (-item[1], item[0]))[:20]:
        print(f"  {key}: {value}")
    print("")
    print("selected windows:")
    for idx, item in enumerate(report["selected_windows"], start=1):
        w = item["window"]
        topk = ",".join(str(x) for x in w.get("topk_target_regions", [])[:3])
        fp = "FP" if w.get("is_false_positive") else "OK"
        topk_status = "topkOK" if w.get("topk_matched") else "topkMISS"
        print("")
        print(
            f"[{idx}] row={w['dataset_row_index']} sample={w['sample_index']} "
            f"action={w.get('action_type', '')} key={w.get('dataset_key', '')}"
        )
        print(
            f"    hand={w['hand_side']} primary={w['primary_target_region']} topk=[{topk}] "
            f"win=[{w['start_frame']},{w['end_frame']}) raw=[{w['raw_start_frame']},{w['raw_end_frame']})"
        )
        print(
            f"    purity={float(w.get('window_contact_purity', 0.0)):.3f} "
            f"best_overlap={w.get('best_overlap', 0)} topk_overlap={w.get('topk_best_overlap', 0)} "
            f"{fp} {topk_status}"
        )
        for line in item["timeline"]:
            print("    " + line)


def _write_markdown(path: str, report: dict[str, Any]):
    lines = [
        "# refine_v2 Subset Window Sanity Report",
        "",
        f"metadata: `{report['subset_window_metadata_path']}`",
        f"audit: `{report['audit_json_path']}`",
        f"contact_labels: `{report['contact_labels_path']}`",
        "",
        "## Summary",
        "",
        "```text",
    ]
    for key, value in report["summary"].items():
        if isinstance(value, dict):
            continue
        lines.append(f"{key}: {value}")
    lines.extend(["```", ""])
    lines.extend(["## Selected Windows", ""])
    for idx, item in enumerate(report["selected_windows"], start=1):
        w = item["window"]
        topk = ", ".join(str(x) for x in w.get("topk_target_regions", [])[:3])
        fp = "FP" if w.get("is_false_positive") else "OK"
        topk_status = "topkOK" if w.get("topk_matched") else "topkMISS"
        lines.extend(
            [
                f"### Window {idx}",
                "",
                (
                    f"- row/sample: `{w['dataset_row_index']}` / `{w['sample_index']}`\n"
                    f"- action: `{w.get('action_type', '')}`\n"
                    f"- dataset_key: `{w.get('dataset_key', '')}`\n"
                    f"- hand: `{w['hand_side']}`\n"
                    f"- primary: `{w['primary_target_region']}`\n"
                    f"- topk: `{topk}`\n"
                    f"- window: `[{w['start_frame']},{w['end_frame']})`\n"
                    f"- raw: `[{w['raw_start_frame']},{w['raw_end_frame']})`\n"
                    f"- purity: `{float(w.get('window_contact_purity', 0.0)):.3f}`\n"
                    f"- overlap: primary=`{w.get('best_overlap', 0)}`, topk=`{w.get('topk_best_overlap', 0)}`\n"
                    f"- status: `{fp} {topk_status}`"
                ),
                "",
                "```text",
            ]
        )
        lines.extend(item["timeline"])
        lines.extend(["```", ""])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Inspect refine_v2 contact-subset windows.")
    parser.add_argument("--subset_window_metadata_path", required=True, type=str)
    parser.add_argument("--audit_json", required=True, type=str)
    parser.add_argument("--contact_labels_path", default="", type=str)
    parser.add_argument("--action_type", action="append", default=None)
    parser.add_argument("--hand_side", default="", choices=["", "left", "right"])
    parser.add_argument("--region", default="", type=str)
    parser.add_argument("--bucket_label", default="", type=str)
    parser.add_argument("--fp_only", action="store_true")
    parser.add_argument("--tp_only", action="store_true")
    parser.add_argument("--topk_only", action="store_true")
    parser.add_argument("--topk_miss_only", action="store_true")
    parser.add_argument("--min_purity", default=None, type=float)
    parser.add_argument("--max_purity", default=None, type=float)
    parser.add_argument(
        "--sort_by",
        default="purity_desc",
        choices=["purity_desc", "purity_asc", "fp_first", "topk_miss_first", "random", "row"],
    )
    parser.add_argument("--limit", default=20, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--timeline_width", default=100, type=int)
    parser.add_argument("--output_json", default="", type=str)
    parser.add_argument("--output_md", default="", type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = build_subset_window_report(
        args.subset_window_metadata_path,
        args.audit_json,
        contact_labels_path=args.contact_labels_path,
        action_types=args.action_type,
        hand_side=args.hand_side,
        region=args.region,
        bucket_label=args.bucket_label,
        fp_only=args.fp_only,
        tp_only=args.tp_only,
        topk_only=args.topk_only,
        topk_miss_only=args.topk_miss_only,
        min_purity=args.min_purity,
        max_purity=args.max_purity,
        sort_by=args.sort_by,
        limit=args.limit,
        seed=args.seed,
        timeline_width=args.timeline_width,
    )
    _print_report(report)
    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, indent=2, sort_keys=True)
    if args.output_md:
        _write_markdown(args.output_md, report)


if __name__ == "__main__":
    main()
