"""CLI: rerun frozen refine_v2 selector/window on a subset manifest."""

from __future__ import annotations

import argparse
import os

from refine_v2.data.schema import (
    DEFAULT_PER_HAND_MAX_WINDOWS,
    DEFAULT_PER_SEQ_MAX_WINDOWS,
    DEFAULT_TOP_K_REGIONS,
    DEFAULT_WINDOW_SIZE,
)
from refine_v2.subset.reporting import markdown_table, write_csv, write_json


WINDOW_METADATA_FIELDS = [
    "dataset_row_index",
    "sample_index",
    "dataset_key",
    "action_type",
    "action_label",
    "action_name",
    "bucket_label",
    "is_gt_positive",
    "is_pred_positive",
    "hand_side",
    "hand_side_id",
    "start_frame",
    "end_frame",
    "center_frame",
    "raw_start_frame",
    "raw_end_frame",
    "raw_length",
    "primary_target_region",
    "primary_target_region_id",
    "topk_target_regions",
    "topk_target_region_ids",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Rerun fixed refine_v2 selector on a subset manifest.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--contact_labels_path", required=True, type=str)
    parser.add_argument("--subset_manifest_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--region_map_path", default="", type=str)
    parser.add_argument("--include_buckets", nargs="*", default=["GT+ / Pred+"])
    parser.add_argument("--tau_contact", default=0.10, type=float)
    parser.add_argument("--gap_merge", default=4, type=int)
    parser.add_argument("--raw_L_min", default=2, type=int)
    parser.add_argument("--window_size", default=DEFAULT_WINDOW_SIZE, type=int)
    parser.add_argument("--per_hand_max_windows", default=DEFAULT_PER_HAND_MAX_WINDOWS, type=int)
    parser.add_argument("--per_seq_max_windows", default=DEFAULT_PER_SEQ_MAX_WINDOWS, type=int)
    parser.add_argument("--top_k_regions", default=DEFAULT_TOP_K_REGIONS, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--frame_chunk", default=1, type=int)
    parser.add_argument("--target_chunk", default=2048, type=int)
    parser.add_argument("--no_progress", action="store_true")
    return parser


def _write_audit_summary_md(path: str, audit_payload: dict, *, selector_path: str, audit_path: str):
    metrics = audit_payload.get("metrics", {})
    focus = [
        "num_sequences",
        "num_gt_segments",
        "num_pred_windows",
        "gt_positive_zero_window_ratio",
        "topk_gt_segment_recall",
        "topk_window_match_ratio",
        "topk_region_match_ratio",
        "window_contact_purity",
        "false_positive_window_ratio",
        "gt_negative_nonzero_window_ratio",
        "hand_only_gt_segment_recall",
        "time_only_gt_segment_recall",
    ]
    rows = [{"metric": key, "value": metrics.get(key, "")} for key in focus]
    lines = [
        "# refine_v2 Subset Selector Audit Summary",
        "",
        f"selector_windows_path: `{selector_path}`",
        f"selector_audit_path: `{audit_path}`",
        "",
        "## Key Metrics",
        "",
        markdown_table(rows, ["metric", "value"]),
        "",
        "## Diagnostic Summary Top-K",
        "",
        "```text",
    ]
    for key, value in audit_payload.get("diagnostic_summary_topk", {}).items():
        lines.append(f"{key}: {value}")
    lines.extend(["```", ""])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main(argv=None):
    args = build_parser().parse_args(argv)
    from refine_v2.eval.audit_v2 import audit_windows, save_audit_json
    from refine_v2.model.regions import load_region_map
    from refine_v2.model.selector_v2 import build_windows_for_loader, save_selector_windows
    from refine_v2.subset.subset_loader import build_subset_window_metadata, make_subset_reaction_data_loader

    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)
    region_map = load_region_map(args.region_map_path or None)
    loader = make_subset_reaction_data_loader(
        args.reaction_data_path,
        args.subset_manifest_path,
        include_buckets=args.include_buckets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dataset_row_indices = list(loader.dataset.dataset_row_indices)
    artifact = build_windows_for_loader(
        loader,
        region_map,
        tau_contact=args.tau_contact,
        gap_merge=args.gap_merge,
        raw_L_min=args.raw_L_min,
        window_size=args.window_size,
        per_hand_max_windows=args.per_hand_max_windows,
        per_seq_max_windows=args.per_seq_max_windows,
        top_k_regions=args.top_k_regions,
        device=args.device,
        frame_chunk=args.frame_chunk,
        target_chunk=args.target_chunk,
        show_progress=not args.no_progress,
    )
    selector_path = os.path.join(args.output_dir, "subset_selector_windows.npz")
    audit_path = os.path.join(args.output_dir, "subset_selector_audit.json")
    audit_summary_path = os.path.join(args.output_dir, "subset_selector_audit_summary.md")
    window_metadata_json = os.path.join(args.output_dir, "subset_window_metadata.json")
    window_metadata_csv = os.path.join(args.output_dir, "subset_window_metadata.csv")

    save_selector_windows(selector_path, artifact)
    audit_payload = audit_windows(
        args.contact_labels_path,
        selector_path,
        dataset_row_indices=dataset_row_indices,
        show_progress=not args.no_progress,
    )
    save_audit_json(audit_path, audit_payload)
    _write_audit_summary_md(audit_summary_path, audit_payload, selector_path=selector_path, audit_path=audit_path)

    window_metadata = build_subset_window_metadata(
        selector_path,
        args.subset_manifest_path,
        include_buckets=args.include_buckets,
    )
    write_json(window_metadata_json, {"artifact": "refine_v2_subset_window_metadata", "windows": window_metadata})
    write_csv(window_metadata_csv, window_metadata, WINDOW_METADATA_FIELDS)

    print(f"saved subset selector windows: {selector_path}")
    print(f"saved subset audit: {audit_path}")
    print(f"saved subset audit summary: {audit_summary_path}")
    print(f"saved subset window metadata: {window_metadata_json}")
    for key in [
        "num_sequences",
        "num_gt_segments",
        "num_pred_windows",
        "gt_positive_zero_window_ratio",
        "topk_gt_segment_recall",
        "topk_window_match_ratio",
        "topk_region_match_ratio",
        "window_contact_purity",
        "false_positive_window_ratio",
        "gt_negative_nonzero_window_ratio",
    ]:
        print(f"{key}: {audit_payload['metrics'].get(key)}")


if __name__ == "__main__":
    main()
