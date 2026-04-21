"""Action-type contact statistics for refine_v2 contact-rich subsets."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from statistics import mean, median
from typing import Any

import numpy as np

from refine_v2.subset.interx_action_meta import (
    INTERX_ACTION_ID_TO_NAME,
    INTERX_ACTION_NAME_TO_ID,
    action_name_for_label,
    parse_action_label_from_dataset_key,
)
from refine_v2.data.schema import object_array_to_records
from refine_v2.subset.reporting import markdown_table, write_csv, write_json


ACTION_STAT_FIELDS = [
    "action_label",
    "action_type",
    "action_name",
    "num_sequences",
    "num_gt_positive_sequences",
    "gt_positive_sequence_ratio",
    "num_gt_segments",
    "gt_segments_per_sequence",
    "total_gt_contact_frames",
    "gt_contact_frame_ratio",
    "avg_gt_segment_length",
    "median_gt_segment_length",
    "num_selector_windows",
    "windows_per_sequence",
    "topk_gt_segment_recall",
    "topk_window_match_ratio",
    "window_contact_purity",
    "false_positive_window_ratio",
    "contact_rich_score",
    "training_value_score",
    "is_small_sample",
    "recommendation",
]

SEQUENCE_STAT_FIELDS = [
    "dataset_row_index",
    "sample_index",
    "dataset_key",
    "action_label",
    "action_type",
    "action_name",
    "action_parse_source",
    "length",
    "is_gt_positive",
    "is_pred_positive",
    "bucket_label",
    "num_gt_segments",
    "total_gt_contact_frames",
    "gt_contact_frame_ratio",
    "num_selector_windows",
    "topk_gt_segment_recall_for_sequence",
    "window_contact_purity_mean_for_sequence",
    "false_positive_window_ratio_for_sequence",
]


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _as_text(value.item())
        if value.size == 1:
            return _as_text(value.reshape(-1)[0])
        return str(value.tolist())
    if isinstance(value, (np.integer,)):
        return str(int(value))
    return str(value)


def _read_reaction_value(dataset: ReactionDataDataset, field: str, row_idx: int) -> Any:
    from refine.data.cache_dataset import _read_source_value

    if field not in dataset.extra_fields:
        return None
    return _read_source_value(dataset.extra_fields[field], row_idx)


def _dataset_key_at(dataset: ReactionDataDataset, row_idx: int) -> str:
    value = _read_reaction_value(dataset, "dataset_key", row_idx)
    if value is None:
        item = dataset[row_idx]
        value = item.get("dataset_key", f"sample_{row_idx}")
    return _as_text(value)


def _normalize_action_from_value(value: Any) -> tuple[str, str, str] | None:
    if value is None:
        return None
    text = _as_text(value).strip()
    if not text:
        return None
    if text in INTERX_ACTION_ID_TO_NAME:
        return text, action_name_for_label(text), "reaction_data_action_label"
    if text in INTERX_ACTION_NAME_TO_ID:
        label = INTERX_ACTION_NAME_TO_ID[text]
        return label, action_name_for_label(label), "reaction_data_action_name"
    lower_to_label = {name.lower(): label for name, label in INTERX_ACTION_NAME_TO_ID.items()}
    if text.lower() in lower_to_label:
        label = lower_to_label[text.lower()]
        return label, action_name_for_label(label), "reaction_data_action_name_casefold"
    if text.isdigit():
        label = f"A{int(text):03d}"
        if label in INTERX_ACTION_ID_TO_NAME:
            return label, action_name_for_label(label), "reaction_data_action_id"
    parsed_label, source = parse_action_label_from_dataset_key(text)
    if parsed_label in INTERX_ACTION_ID_TO_NAME:
        return parsed_label, action_name_for_label(parsed_label), f"parsed_from_{source}"
    return None


def _action_info_for_row(dataset: ReactionDataDataset, row_idx: int, *, allow_unknown_action: bool) -> dict[str, Any]:
    dataset_key = _dataset_key_at(dataset, row_idx)
    for field in ("action_type", "action_label", "action_name", "action_text", "action"):
        parsed = _normalize_action_from_value(_read_reaction_value(dataset, field, row_idx))
        if parsed is not None:
            label, name, source = parsed
            return {
                "dataset_key": dataset_key,
                "action_label": label,
                "action_name": name,
                "action_type": name,
                "action_parse_source": f"{field}:{source}",
            }

    label, source = parse_action_label_from_dataset_key(dataset_key)
    if label in INTERX_ACTION_ID_TO_NAME:
        return {
            "dataset_key": dataset_key,
            "action_label": label,
            "action_name": action_name_for_label(label),
            "action_type": action_name_for_label(label),
            "action_parse_source": source,
        }
    if allow_unknown_action:
        return {
            "dataset_key": dataset_key,
            "action_label": label,
            "action_name": "unknown",
            "action_type": "unknown",
            "action_parse_source": source,
        }
    raise ValueError(
        "Unable to determine Inter-X action type. "
        f"dataset_row_index={row_idx}, dataset_key={dataset_key!r}. "
        "Expected action metadata in reaction_data or an Axxx/actionxxx token in dataset_key. "
        "Pass --allow_unknown_action only for diagnostics."
    )


def load_action_metadata(
    reaction_data_path: str,
    *,
    allow_unknown_action: bool = False,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    from refine.data.cache_dataset import ReactionDataDataset

    dataset = ReactionDataDataset(reaction_data_path)
    records: list[dict[str, Any]] = []
    try:
        for row_idx in range(len(dataset)):
            info = _action_info_for_row(dataset, row_idx, allow_unknown_action=allow_unknown_action)
            record = {
                "dataset_row_index": int(row_idx),
                "sample_index": int(np.asarray(dataset.sample_indices[row_idx])),
                **info,
            }
            records.append(record)
    finally:
        dataset.close()
    return records, {int(item["dataset_row_index"]): item for item in records}


def _records_by_row(records: list[dict[str, Any]], key: str = "dataset_row_index") -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[int(item[key])].append(item)
    return dict(grouped)


def _load_audit_debug(selector_audit_path: str) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    if not selector_audit_path:
        return {}, {}
    import json

    with open(selector_audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    per_gt_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in audit.get("per_gt_segment", []):
        gt = item.get("gt_segment", {})
        if "dataset_row_index" in gt:
            per_gt_by_row[int(gt["dataset_row_index"])].append(item)
    per_window_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in audit.get("per_window", []):
        window = item.get("window", {})
        if "dataset_row_index" in window:
            per_window_by_row[int(window["dataset_row_index"])].append(item)
    return dict(per_gt_by_row), dict(per_window_by_row)


def _sequence_bucket(is_gt_positive: bool, is_pred_positive: bool) -> str:
    if is_gt_positive and is_pred_positive:
        return "GT+ / Pred+"
    if is_gt_positive and not is_pred_positive:
        return "GT+ / Pred0"
    if not is_gt_positive and is_pred_positive:
        return "GT0 / Pred+"
    return "GT0 / Pred0"


def build_sequence_contact_stats(
    reaction_data_path: str,
    contact_labels_path: str,
    selector_windows_path: str,
    selector_audit_path: str = "",
    *,
    allow_unknown_action: bool = False,
) -> list[dict[str, Any]]:
    action_records, action_by_row = load_action_metadata(
        reaction_data_path,
        allow_unknown_action=allow_unknown_action,
    )
    if not action_records:
        raise ValueError(f"No sequences found in reaction_data: {reaction_data_path}")

    labels = np.load(contact_labels_path, allow_pickle=True)
    gt_mask = np.asarray(labels["gt_contact_mask"], dtype=np.uint8)
    lengths = np.asarray(labels["lengths"], dtype=np.int64).reshape(-1)
    label_rows = np.asarray(labels["dataset_row_indices"], dtype=np.int64).reshape(-1)
    label_sample_indices = np.asarray(labels["sample_indices"], dtype=np.int64).reshape(-1)
    label_dataset_keys = np.asarray(labels["dataset_key"], dtype=object).reshape(-1)
    gt_segments = object_array_to_records(labels["segments"])
    gt_segments_by_row = _records_by_row(gt_segments)

    windows_pack = np.load(selector_windows_path, allow_pickle=True)
    windows = object_array_to_records(windows_pack["windows"])
    windows_by_row = _records_by_row(windows)
    per_gt_by_row, per_window_by_row = _load_audit_debug(selector_audit_path)

    sequence_records: list[dict[str, Any]] = []
    for mask_index, row_value in enumerate(label_rows.tolist()):
        row = int(row_value)
        action = action_by_row.get(row)
        if action is None:
            raise ValueError(
                f"contact_labels row {row} is not present in reaction_data action metadata. "
                "Make sure reaction_data_path and contact_labels_path refer to the same pack."
            )
        valid_len = int(lengths[mask_index])
        valid_mask = gt_mask[mask_index, :, :, :valid_len].astype(bool)
        total_gt_contact_frames = int(valid_mask.sum())
        gt_den = max(valid_len * valid_mask.shape[0] * valid_mask.shape[1], 1)
        row_gt_segments = gt_segments_by_row.get(row, [])
        row_windows = windows_by_row.get(row, [])
        row_gt_debug = per_gt_by_row.get(row, [])
        row_window_debug = per_window_by_row.get(row, [])
        topk_hits = int(sum(1 for item in row_gt_debug if bool(item.get("topk_matched", False))))
        purity_values = [float(item.get("window_contact_purity", 0.0)) for item in row_window_debug]
        false_pos_count = int(sum(1 for item in row_window_debug if bool(item.get("is_false_positive", False))))
        is_gt_positive = total_gt_contact_frames > 0
        is_pred_positive = len(row_windows) > 0
        sequence_records.append(
            {
                "dataset_row_index": row,
                "sample_index": int(label_sample_indices[mask_index]),
                "dataset_key": _as_text(label_dataset_keys[mask_index]),
                "action_label": action["action_label"],
                "action_type": action["action_type"],
                "action_name": action["action_name"],
                "action_parse_source": action["action_parse_source"],
                "length": valid_len,
                "is_gt_positive": bool(is_gt_positive),
                "is_pred_positive": bool(is_pred_positive),
                "bucket_label": _sequence_bucket(is_gt_positive, is_pred_positive),
                "num_gt_segments": int(len(row_gt_segments)),
                "total_gt_contact_frames": total_gt_contact_frames,
                "gt_contact_frame_ratio": float(total_gt_contact_frames / gt_den),
                "num_selector_windows": int(len(row_windows)),
                "topk_gt_segment_recall_for_sequence": float(topk_hits / max(len(row_gt_segments), 1)),
                "window_contact_purity_mean_for_sequence": float(mean(purity_values)) if purity_values else 0.0,
                "false_positive_window_ratio_for_sequence": float(false_pos_count / max(len(row_windows), 1)),
            }
        )
    return sorted(sequence_records, key=lambda item: int(item["dataset_row_index"]))


def _recommendation_for_action(row: dict[str, Any], *, min_sequences_for_recommendation: int) -> str:
    if int(row["num_sequences"]) < int(min_sequences_for_recommendation):
        return "small_sample_review_only"
    if float(row["gt_positive_sequence_ratio"]) < 0.3:
        return "not_contact_rich"
    if float(row["contact_rich_score"]) <= 0.0:
        return "not_contact_rich"
    if float(row["topk_gt_segment_recall"]) < 0.5 and int(row["num_gt_segments"]) > 0:
        return "contact_rich_but_selector_weak"
    return "recommended_candidate"


def aggregate_action_type_stats(
    sequence_records: list[dict[str, Any]],
    *,
    min_sequences_for_recommendation: int = 20,
) -> list[dict[str, Any]]:
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seq in sequence_records:
        by_action[str(seq["action_label"])].append(seq)

    rows: list[dict[str, Any]] = []
    for label, items in sorted(by_action.items()):
        num_sequences = len(items)
        num_gt_positive = int(sum(1 for item in items if bool(item["is_gt_positive"])))
        num_gt_segments = int(sum(int(item["num_gt_segments"]) for item in items))
        total_gt_contact_frames = int(sum(int(item["total_gt_contact_frames"]) for item in items))
        total_den = int(sum(max(int(item["length"]) * 2 * 6, 1) for item in items))
        segment_lengths = []
        for item in items:
            if int(item["num_gt_segments"]) > 0:
                # Segment-level lengths are not stored in sequence rows; derive averages via audit/labels elsewhere
                # by using per-sequence total frames only as a fallback. Exact lengths are filled below when available.
                pass
        num_windows = int(sum(int(item["num_selector_windows"]) for item in items))
        topk_gt_hits = sum(
            float(item["topk_gt_segment_recall_for_sequence"]) * int(item["num_gt_segments"])
            for item in items
        )
        purity_num = sum(
            float(item["window_contact_purity_mean_for_sequence"]) * int(item["num_selector_windows"])
            for item in items
        )
        false_pos_num = sum(
            float(item["false_positive_window_ratio_for_sequence"]) * int(item["num_selector_windows"])
            for item in items
        )
        gt_positive_ratio = float(num_gt_positive / max(num_sequences, 1))
        gt_segments_per_sequence = float(num_gt_segments / max(num_sequences, 1))
        gt_contact_frame_ratio = float(total_gt_contact_frames / max(total_den, 1))
        topk_recall = float(topk_gt_hits / max(num_gt_segments, 1))
        topk_window_match_ratio = 0.0
        # In current sequence rows, top-k window match is only available through per-window debug.
        # It is accumulated exactly in enrich_sequence_records_with_windows().
        window_purity = float(purity_num / max(num_windows, 1))
        false_positive_ratio = float(false_pos_num / max(num_windows, 1))
        contact_rich_score = float(gt_positive_ratio * math.log1p(gt_segments_per_sequence) * gt_contact_frame_ratio)
        training_value_score = float(gt_contact_frame_ratio * (num_windows / max(num_sequences, 1)) * topk_window_match_ratio)
        row = {
            "action_label": label,
            "action_type": items[0]["action_type"],
            "action_name": items[0]["action_name"],
            "num_sequences": int(num_sequences),
            "num_gt_positive_sequences": int(num_gt_positive),
            "gt_positive_sequence_ratio": gt_positive_ratio,
            "num_gt_segments": int(num_gt_segments),
            "gt_segments_per_sequence": gt_segments_per_sequence,
            "total_gt_contact_frames": int(total_gt_contact_frames),
            "gt_contact_frame_ratio": gt_contact_frame_ratio,
            "avg_gt_segment_length": 0.0,
            "median_gt_segment_length": 0.0,
            "num_selector_windows": int(num_windows),
            "windows_per_sequence": float(num_windows / max(num_sequences, 1)),
            "topk_gt_segment_recall": topk_recall,
            "topk_window_match_ratio": topk_window_match_ratio,
            "window_contact_purity": window_purity,
            "false_positive_window_ratio": false_positive_ratio,
            "contact_rich_score": contact_rich_score,
            "training_value_score": training_value_score,
            "is_small_sample": bool(num_sequences < int(min_sequences_for_recommendation)),
        }
        row["recommendation"] = _recommendation_for_action(
            row,
            min_sequences_for_recommendation=min_sequences_for_recommendation,
        )
        rows.append(row)
    return sorted(rows, key=lambda item: (-float(item["contact_rich_score"]), -float(item["training_value_score"]), str(item["action_label"])))


def _segment_lengths_by_row(contact_labels_path: str) -> dict[int, list[int]]:
    labels = np.load(contact_labels_path, allow_pickle=True)
    out: dict[int, list[int]] = defaultdict(list)
    for seg in object_array_to_records(labels["segments"]):
        out[int(seg["dataset_row_index"])].append(int(seg["raw_length"]))
    return dict(out)


def _topk_window_hits_by_row(selector_audit_path: str) -> dict[int, tuple[int, int]]:
    if not selector_audit_path:
        return {}
    import json

    with open(selector_audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    out: dict[int, list[int]] = defaultdict(list)
    for item in audit.get("per_window", []):
        window = item.get("window", {})
        if "dataset_row_index" in window:
            out[int(window["dataset_row_index"])].append(int(bool(item.get("topk_matched", False))))
    return {row: (int(sum(values)), int(len(values))) for row, values in out.items()}


def build_action_type_stats(
    reaction_data_path: str,
    contact_labels_path: str,
    selector_windows_path: str,
    selector_audit_path: str,
    *,
    allow_unknown_action: bool = False,
    min_sequences_for_recommendation: int = 20,
) -> dict[str, Any]:
    sequence_records = build_sequence_contact_stats(
        reaction_data_path,
        contact_labels_path,
        selector_windows_path,
        selector_audit_path,
        allow_unknown_action=allow_unknown_action,
    )
    segment_lengths_by_row = _segment_lengths_by_row(contact_labels_path)
    topk_window_hits_by_row = _topk_window_hits_by_row(selector_audit_path)
    for item in sequence_records:
        lengths = segment_lengths_by_row.get(int(item["dataset_row_index"]), [])
        item["gt_segment_length_sum"] = int(sum(lengths))
        item["gt_segment_lengths"] = lengths
        hit_count, window_count = topk_window_hits_by_row.get(int(item["dataset_row_index"]), (0, 0))
        item["topk_window_match_count_for_sequence"] = int(hit_count)
        item["topk_window_match_ratio_for_sequence"] = float(hit_count / max(window_count, 1))

    action_rows = aggregate_action_type_stats(
        sequence_records,
        min_sequences_for_recommendation=min_sequences_for_recommendation,
    )
    by_label = {row["action_label"]: row for row in action_rows}
    for label, row in by_label.items():
        items = [item for item in sequence_records if item["action_label"] == label]
        seg_lengths: list[int] = []
        topk_window_hits = 0
        topk_window_total = 0
        for item in items:
            seg_lengths.extend(int(x) for x in item.get("gt_segment_lengths", []))
            topk_window_hits += int(item.get("topk_window_match_count_for_sequence", 0))
            topk_window_total += int(item.get("num_selector_windows", 0))
        row["avg_gt_segment_length"] = float(mean(seg_lengths)) if seg_lengths else 0.0
        row["median_gt_segment_length"] = float(median(seg_lengths)) if seg_lengths else 0.0
        row["topk_window_match_ratio"] = float(topk_window_hits / max(topk_window_total, 1))
        row["training_value_score"] = float(
            row["gt_contact_frame_ratio"] * row["windows_per_sequence"] * row["topk_window_match_ratio"]
        )
        row["recommendation"] = _recommendation_for_action(
            row,
            min_sequences_for_recommendation=min_sequences_for_recommendation,
        )

    action_rows = sorted(
        action_rows,
        key=lambda item: (-float(item["contact_rich_score"]), -float(item["training_value_score"]), str(item["action_label"])),
    )
    return {
        "artifact": "refine_v2_action_type_stats",
        "reaction_data_path": reaction_data_path,
        "contact_labels_path": contact_labels_path,
        "selector_windows_path": selector_windows_path,
        "selector_audit_path": selector_audit_path,
        "min_sequences_for_recommendation": int(min_sequences_for_recommendation),
        "action_type_source": (
            "reaction_data action metadata if available; otherwise Inter-X Axxx/actionxxx parsed from dataset_key "
            "via refine.protocols.interx_actions"
        ),
        "num_sequences": int(len(sequence_records)),
        "num_action_types": int(len(action_rows)),
        "rows": action_rows,
        "sequence_records": sequence_records,
    }


def _write_stats_md(path: str, payload: dict[str, Any]):
    rows = payload["rows"]
    by_training = sorted(
        rows,
        key=lambda item: (-float(item["training_value_score"]), -float(item["contact_rich_score"]), str(item["action_label"])),
    )
    recommended = [row for row in rows if row.get("recommendation") == "recommended_candidate"]
    small = [row for row in rows if row.get("is_small_sample")]
    not_recommended = [row for row in rows if row.get("recommendation") != "recommended_candidate"]
    display_fields = [
        "action_label",
        "action_name",
        "num_sequences",
        "gt_positive_sequence_ratio",
        "gt_contact_frame_ratio",
        "gt_segments_per_sequence",
        "windows_per_sequence",
        "topk_gt_segment_recall",
        "topk_window_match_ratio",
        "window_contact_purity",
        "false_positive_window_ratio",
        "contact_rich_score",
        "training_value_score",
        "recommendation",
    ]
    lines = [
        "# refine_v2 Action-Type Contact Statistics",
        "",
        f"reaction_data_path: `{payload['reaction_data_path']}`",
        f"contact_labels_path: `{payload['contact_labels_path']}`",
        f"selector_windows_path: `{payload['selector_windows_path']}`",
        f"selector_audit_path: `{payload['selector_audit_path']}`",
        "",
        "Action type source:",
        "",
        f"- {payload['action_type_source']}",
        "",
        "## Summary",
        "",
        f"- num_sequences: `{payload['num_sequences']}`",
        f"- num_action_types: `{payload['num_action_types']}`",
        f"- min_sequences_for_recommendation: `{payload['min_sequences_for_recommendation']}`",
        "",
        "## Recommended Contact-Rich Candidates",
        "",
        markdown_table(recommended, display_fields) if recommended else "No action types met the recommendation rules.",
        "",
        "## Ranked By contact_rich_score",
        "",
        markdown_table(rows, display_fields),
        "",
        "## Ranked By training_value_score",
        "",
        markdown_table(by_training, display_fields),
        "",
        "## Small-Sample Action Types",
        "",
        markdown_table(small, display_fields) if small else "None.",
        "",
        "## Not Directly Recommended",
        "",
        markdown_table(not_recommended, display_fields) if not_recommended else "None.",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def write_action_type_stats_outputs(payload: dict[str, Any], output_dir: str):
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    json_path = os.path.join(output_dir, "action_type_stats.json")
    csv_path = os.path.join(output_dir, "action_type_stats.csv")
    md_path = os.path.join(output_dir, "action_type_stats.md")
    seq_csv_path = os.path.join(output_dir, "sequence_action_contact_stats.csv")
    write_json(json_path, payload)
    write_csv(csv_path, payload["rows"], ACTION_STAT_FIELDS)
    write_csv(seq_csv_path, payload["sequence_records"], SEQUENCE_STAT_FIELDS)
    _write_stats_md(md_path, payload)
    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "md_path": md_path,
        "sequence_csv_path": seq_csv_path,
    }
