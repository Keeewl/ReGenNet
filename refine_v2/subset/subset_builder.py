"""Build sequence-level contact-rich subset manifests."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Any

from refine_v2.subset.interx_action_meta import INTERX_ACTION_ID_TO_NAME, INTERX_ACTION_NAME_TO_ID
from refine_v2.subset.reporting import markdown_table, read_json, write_csv, write_json


SUBSET_SEQUENCE_FIELDS = [
    "dataset_row_index",
    "sample_index",
    "dataset_key",
    "action_label",
    "action_type",
    "action_name",
    "is_gt_positive",
    "is_pred_positive",
    "bucket_label",
    "num_gt_segments",
    "total_gt_contact_frames",
    "num_selector_windows",
    "topk_gt_segment_recall_for_sequence",
    "window_contact_purity_mean_for_sequence",
    "false_positive_window_ratio_for_sequence",
]


def _canonical_action_token(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    if text in INTERX_ACTION_ID_TO_NAME:
        return text
    if text in INTERX_ACTION_NAME_TO_ID:
        return INTERX_ACTION_NAME_TO_ID[text]
    lower_to_label = {name.lower(): label for name, label in INTERX_ACTION_NAME_TO_ID.items()}
    if text.lower() in lower_to_label:
        return lower_to_label[text.lower()]
    if text.isdigit():
        return f"A{int(text):03d}"
    return text


def select_action_labels(
    action_rows: list[dict[str, Any]],
    *,
    selected_action_types: list[str] | None = None,
    min_num_sequences: int = 20,
    min_gt_positive_sequence_ratio: float = 0.5,
    min_contact_rich_score: float = 0.0,
    min_training_value_score: float = 0.0,
) -> tuple[list[str], str]:
    available = {str(row["action_label"]): row for row in action_rows}
    if selected_action_types:
        labels = []
        unknown = []
        for value in selected_action_types:
            label = _canonical_action_token(value)
            if label in available:
                labels.append(label)
            else:
                unknown.append(str(value))
        if unknown:
            raise ValueError(
                "Selected action types are not present in action_type_stats: "
                + ", ".join(unknown)
            )
        return sorted(set(labels)), "explicit_selected_action_types"

    labels = []
    for row in action_rows:
        if int(row.get("num_sequences", 0)) < int(min_num_sequences):
            continue
        if float(row.get("gt_positive_sequence_ratio", 0.0)) < float(min_gt_positive_sequence_ratio):
            continue
        if float(row.get("contact_rich_score", 0.0)) < float(min_contact_rich_score):
            continue
        if float(row.get("training_value_score", 0.0)) < float(min_training_value_score):
            continue
        labels.append(str(row["action_label"]))
    return sorted(set(labels)), "thresholds"


def _bucket_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(str(item["bucket_label"]) for item in records)
    for name in ("GT+ / Pred+", "GT+ / Pred0", "GT0 / Pred+", "GT0 / Pred0"):
        counter.setdefault(name, 0)
    return dict(sorted(counter.items()))


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_gt_segments = int(sum(int(item.get("num_gt_segments", 0)) for item in records))
    total_gt_contact_frames = int(sum(int(item.get("total_gt_contact_frames", 0)) for item in records))
    total_windows = int(sum(int(item.get("num_selector_windows", 0)) for item in records))
    purity_num = sum(
        float(item.get("window_contact_purity_mean_for_sequence", 0.0)) * int(item.get("num_selector_windows", 0))
        for item in records
    )
    fp_num = sum(
        float(item.get("false_positive_window_ratio_for_sequence", 0.0)) * int(item.get("num_selector_windows", 0))
        for item in records
    )
    topk_hits = sum(
        float(item.get("topk_gt_segment_recall_for_sequence", 0.0)) * int(item.get("num_gt_segments", 0))
        for item in records
    )
    return {
        "num_sequences": int(len(records)),
        "num_gt_positive_sequences": int(sum(1 for item in records if bool(item.get("is_gt_positive", False)))),
        "num_pred_positive_sequences": int(sum(1 for item in records if bool(item.get("is_pred_positive", False)))),
        "bucket_counts": _bucket_counts(records),
        "num_gt_segments": total_gt_segments,
        "total_gt_contact_frames": total_gt_contact_frames,
        "num_selector_windows": total_windows,
        "topk_gt_segment_recall": float(topk_hits / max(total_gt_segments, 1)),
        "window_contact_purity": float(purity_num / max(total_windows, 1)),
        "false_positive_window_ratio": float(fp_num / max(total_windows, 1)),
    }


def build_subset_manifest(
    action_type_stats_path: str,
    *,
    selected_action_types: list[str] | None = None,
    min_num_sequences: int = 20,
    min_gt_positive_sequence_ratio: float = 0.5,
    min_contact_rich_score: float = 0.0,
    min_training_value_score: float = 0.0,
) -> dict[str, Any]:
    stats = read_json(action_type_stats_path)
    action_rows = list(stats.get("rows", []))
    sequence_records = list(stats.get("sequence_records", []))
    selected_labels, selection_mode = select_action_labels(
        action_rows,
        selected_action_types=selected_action_types,
        min_num_sequences=min_num_sequences,
        min_gt_positive_sequence_ratio=min_gt_positive_sequence_ratio,
        min_contact_rich_score=min_contact_rich_score,
        min_training_value_score=min_training_value_score,
    )
    selected_set = set(selected_labels)
    selected_sequences = [
        dict(item) for item in sequence_records
        if str(item.get("action_label")) in selected_set
    ]
    selected_sequences = sorted(selected_sequences, key=lambda item: int(item["dataset_row_index"]))
    main_positive = [item for item in selected_sequences if item["bucket_label"] == "GT+ / Pred+"]
    hard_negative = [item for item in selected_sequences if item["bucket_label"] == "GT0 / Pred+"]
    by_action = defaultdict(list)
    for item in selected_sequences:
        by_action[str(item["action_label"])].append(item)
    action_summary = []
    action_by_label = {str(row["action_label"]): row for row in action_rows}
    for label in selected_labels:
        row = action_by_label[label]
        seqs = by_action.get(label, [])
        action_summary.append(
            {
                "action_label": label,
                "action_type": row.get("action_type", row.get("action_name", label)),
                "action_name": row.get("action_name", row.get("action_type", label)),
                "num_sequences": int(len(seqs)),
                "bucket_counts": _bucket_counts(seqs),
                "contact_rich_score": float(row.get("contact_rich_score", 0.0)),
                "training_value_score": float(row.get("training_value_score", 0.0)),
                "gt_positive_sequence_ratio": float(row.get("gt_positive_sequence_ratio", 0.0)),
                "topk_gt_segment_recall": float(row.get("topk_gt_segment_recall", 0.0)),
            }
        )
    payload = {
        "artifact": "refine_v2_contact_subset_manifest",
        "action_type_stats_path": action_type_stats_path,
        "selection_mode": selection_mode,
        "selection_thresholds": {
            "min_num_sequences": int(min_num_sequences),
            "min_gt_positive_sequence_ratio": float(min_gt_positive_sequence_ratio),
            "min_contact_rich_score": float(min_contact_rich_score),
            "min_training_value_score": float(min_training_value_score),
        },
        "selected_action_labels": selected_labels,
        "selected_action_types": [
            action_by_label[label].get("action_type", action_by_label[label].get("action_name", label))
            for label in selected_labels
        ],
        "selected_action_summary": action_summary,
        "sequences": selected_sequences,
        "main_positive_sequences": main_positive,
        "hard_negative_sequences": hard_negative,
        "main_positive_dataset_row_indices": [int(item["dataset_row_index"]) for item in main_positive],
        "hard_negative_dataset_row_indices": [int(item["dataset_row_index"]) for item in hard_negative],
        "all_selected_dataset_row_indices": [int(item["dataset_row_index"]) for item in selected_sequences],
        "summary": {
            "all_selected": _summarize_records(selected_sequences),
            "main_positive": _summarize_records(main_positive),
            "hard_negative": _summarize_records(hard_negative),
        },
        "notes": {
            "main_positive_bucket": "GT+ / Pred+",
            "hard_negative_or_diagnostic_bucket": "GT0 / Pred+",
            "do_not_mix_hard_negative_into_positive_subset": True,
        },
    }
    return payload


def _write_subset_md(path: str, payload: dict[str, Any]):
    action_fields = [
        "action_label",
        "action_name",
        "num_sequences",
        "gt_positive_sequence_ratio",
        "topk_gt_segment_recall",
        "contact_rich_score",
        "training_value_score",
        "bucket_counts",
    ]
    seq_fields = [
        "dataset_row_index",
        "sample_index",
        "dataset_key",
        "action_name",
        "bucket_label",
        "num_gt_segments",
        "total_gt_contact_frames",
        "num_selector_windows",
        "topk_gt_segment_recall_for_sequence",
        "window_contact_purity_mean_for_sequence",
    ]
    summary = payload["summary"]
    lines = [
        "# refine_v2 Contact-Rich Subset Manifest",
        "",
        f"action_type_stats_path: `{payload['action_type_stats_path']}`",
        f"selection_mode: `{payload['selection_mode']}`",
        "",
        "## Selected Action Types",
        "",
        markdown_table(payload["selected_action_summary"], action_fields),
        "",
        "## Bucket Summary",
        "",
        "All selected:",
        "",
        "```text",
        *[f"{k}: {v}" for k, v in summary["all_selected"].items()],
        "```",
        "",
        "Main positive subset:",
        "",
        "```text",
        *[f"{k}: {v}" for k, v in summary["main_positive"].items()],
        "```",
        "",
        "Diagnostic / hard-negative bucket:",
        "",
        "```text",
        *[f"{k}: {v}" for k, v in summary["hard_negative"].items()],
        "```",
        "",
        "## Main Positive Sequence Preview",
        "",
        markdown_table(payload["main_positive_sequences"], seq_fields, max_rows=50),
        "",
        "## Diagnostic / Hard-Negative Sequence Preview",
        "",
        markdown_table(payload["hard_negative_sequences"], seq_fields, max_rows=50),
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def write_subset_manifest_outputs(payload: dict[str, Any], output_dir: str):
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    json_path = os.path.join(output_dir, "subset_manifest.json")
    csv_path = os.path.join(output_dir, "subset_sequences.csv")
    md_path = os.path.join(output_dir, "subset_summary.md")
    hard_negative_csv_path = os.path.join(output_dir, "hard_negative_sequences.csv")
    main_positive_csv_path = os.path.join(output_dir, "main_positive_sequences.csv")
    write_json(json_path, payload)
    write_csv(csv_path, payload["sequences"], SUBSET_SEQUENCE_FIELDS)
    write_csv(main_positive_csv_path, payload["main_positive_sequences"], SUBSET_SEQUENCE_FIELDS)
    write_csv(hard_negative_csv_path, payload["hard_negative_sequences"], SUBSET_SEQUENCE_FIELDS)
    _write_subset_md(md_path, payload)
    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "md_path": md_path,
        "main_positive_csv_path": main_positive_csv_path,
        "hard_negative_csv_path": hard_negative_csv_path,
    }
