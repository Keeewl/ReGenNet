"""Build a standalone selector/window report for refine_v2 artifacts."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any

import numpy as np

from refine_v2.data.schema import loads_metadata, object_array_to_records, to_jsonable
from refine_v2.subset.reporting import read_json, write_json, write_csv, markdown_table


def _load_npz(path: str):
    return np.load(path, allow_pickle=True)


def _maybe_json(path: str) -> dict[str, Any]:
    return read_json(path) if path and os.path.exists(path) else {}


def _as_list(value) -> list[Any]:
    arr = np.asarray(value, dtype=object)
    if arr.shape == ():
        item = arr.item()
        return [] if item is None else [item]
    return arr.reshape(-1).tolist()


def _counter(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item.get(key, "")) for item in records).items()))


def _length_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_selector_window_report(
    selector_windows_path: str,
    *,
    contact_labels_path: str = "",
    selector_audit_path: str = "",
    subset_manifest_path: str = "",
) -> dict[str, Any]:
    selector = _load_npz(selector_windows_path)
    windows = object_array_to_records(selector["windows"]) if "windows" in selector.files else []
    raw_segments = object_array_to_records(selector["raw_segments"]) if "raw_segments" in selector.files else []
    params = loads_metadata(selector["selector_params_json"]) if "selector_params_json" in selector.files else {}
    stats = loads_metadata(selector["selector_stats_json"]) if "selector_stats_json" in selector.files else {}
    metadata = loads_metadata(selector["metadata_json"]) if "metadata_json" in selector.files else {}
    audit = _maybe_json(selector_audit_path)
    manifest = _maybe_json(subset_manifest_path)

    dataset_rows_value = selector["dataset_row_indices"] if "dataset_row_indices" in selector.files else np.asarray([], dtype=np.int64)
    sequence_rows = sorted({int(x) for x in np.asarray(dataset_rows_value, dtype=np.int64).reshape(-1).tolist()})
    window_rows = sorted({int(item.get("dataset_row_index", -1)) for item in windows})
    row_window_counts = Counter(int(item.get("dataset_row_index", -1)) for item in windows)
    per_sequence_windows = [float(row_window_counts[row]) for row in sequence_rows]

    topk_region_counts: Counter[str] = Counter()
    for item in windows:
        topk_region_counts.update(str(x) for x in item.get("topk_target_regions", []))

    subset_summary = {}
    if manifest:
        buckets = Counter(str(item.get("bucket_label", "")) for item in manifest.get("sequences", []))
        actions = Counter(str(item.get("action_type", "")) for item in manifest.get("sequences", []))
        subset_summary = {
            "num_manifest_sequences": len(manifest.get("sequences", [])),
            "bucket_distribution": dict(sorted(buckets.items())),
            "action_type_distribution": dict(sorted(actions.items())),
        }

    label_summary = {}
    if contact_labels_path:
        labels = _load_npz(contact_labels_path)
        label_summary = {
            "path": contact_labels_path,
            "num_label_sequences": int(np.asarray(labels["lengths"]).reshape(-1).shape[0]) if "lengths" in labels.files else 0,
            "space_definition": str(np.asarray(labels["space_definition"]).reshape(-1)[0]) if "space_definition" in labels.files else "",
        }

    report = {
        "artifact": "selector_window_report",
        "selector_windows_path": selector_windows_path,
        "contact_labels": label_summary,
        "selector_metadata": metadata,
        "selector_params": params,
        "selector_layered_stats": stats,
        "counts": {
            "num_sequences": int(len(sequence_rows)),
            "num_sequences_with_windows": int(len(window_rows)),
            "num_raw_segments": int(len(raw_segments)),
            "num_windows": int(len(windows)),
            "zero_window_sequence_count": int(len(set(sequence_rows) - set(window_rows))),
            "zero_window_sequence_ratio": float(len(set(sequence_rows) - set(window_rows)) / max(1, len(sequence_rows))),
        },
        "window_stats": {
            "per_sequence_windows": _length_stats(per_sequence_windows),
            "raw_length": _length_stats([float(item.get("raw_length", 0)) for item in windows]),
            "window_size": _length_stats([float(int(item.get("end_frame", 0)) - int(item.get("start_frame", 0))) for item in windows]),
        },
        "distributions": {
            "hand_side": _counter(windows, "hand_side"),
            "primary_target_region": _counter(windows, "primary_target_region"),
            "topk_target_regions": dict(sorted(topk_region_counts.items())),
            "action_type": _counter(windows, "action_type") if windows and "action_type" in windows[0] else {},
        },
        "audit_metrics": audit.get("metrics", audit) if audit else {},
        "subset_summary": subset_summary,
        "schema_note": "This report describes selector/window sampling only. It does not evaluate refiner contact quality.",
    }
    return to_jsonable(report)


def write_selector_report_md(path: str, report: dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = []
    for key, value in report.get("counts", {}).items():
        rows.append({"field": key, "value": value})
    for key, value in report.get("selector_params", {}).items():
        rows.append({"field": f"param.{key}", "value": value})
    metric_rows = [
        {"metric": key, "value": value}
        for key, value in sorted(report.get("audit_metrics", {}).items())
        if isinstance(value, (int, float, str, bool))
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# refine_v2 Selector/Window Report\n\n")
        f.write("This report is about selector/window sampling only. It is separate from refiner contact evaluation.\n\n")
        f.write(f"- selector_windows_path: `{report['selector_windows_path']}`\n\n")
        f.write("## Counts And Params\n\n")
        f.write(markdown_table(rows, ["field", "value"]))
        f.write("\n\n## Audit Metrics\n\n")
        f.write(markdown_table(metric_rows, ["metric", "value"]))
        f.write("\n\n## Layered Stats\n\n")
        layered = [{"field": k, "value": v} for k, v in sorted(report.get("selector_layered_stats", {}).items())]
        f.write(markdown_table(layered, ["field", "value"]))
        f.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Build standalone refine_v2 selector/window report.")
    parser.add_argument("--selector_windows_path", required=True)
    parser.add_argument("--contact_labels_path", default="")
    parser.add_argument("--selector_audit_path", default="")
    parser.add_argument("--subset_manifest_path", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", default="")
    parser.add_argument("--output_csv", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = build_selector_window_report(
        args.selector_windows_path,
        contact_labels_path=args.contact_labels_path,
        selector_audit_path=args.selector_audit_path,
        subset_manifest_path=args.subset_manifest_path,
    )
    write_json(args.output_json, report)
    if args.output_md:
        write_selector_report_md(args.output_md, report)
    if args.output_csv:
        rows = [{"field": key, "value": value} for key, value in report.get("counts", {}).items()]
        rows.extend({"field": f"param.{k}", "value": v} for k, v in report.get("selector_params", {}).items())
        write_csv(args.output_csv, rows, ["field", "value"])
    print(f"saved selector/window report: {args.output_json}")


if __name__ == "__main__":
    main()
