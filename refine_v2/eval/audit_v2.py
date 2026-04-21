"""Strict GT contact-label audit for refine_v2 selector windows."""

from __future__ import annotations

import json
from statistics import mean
from typing import Any

import numpy as np

from refine_v2.data.schema import (
    HAND_SIDE_NAMES,
    TARGET_REGION_NAMES,
    object_array_to_records,
    to_jsonable,
)


def _load_npz(path: str):
    return np.load(path, allow_pickle=True)


def _records(data, key: str) -> list[dict[str, Any]]:
    if key not in data.files:
        return []
    return object_array_to_records(data[key])


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(int(a_end), int(b_end)) - max(int(a_start), int(b_start)))


def _group_key(item: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(item["dataset_row_index"]),
        int(item["hand_side_id"]),
        int(item["target_region_id"]),
    )


def _seq_key(item: dict[str, Any]) -> int:
    return int(item["dataset_row_index"])


def _best_gt_match(window: dict[str, Any], gt_by_group: dict[tuple[int, int, int], list[dict[str, Any]]]):
    key = _group_key(window)
    candidates = gt_by_group.get(key, [])
    best = None
    best_overlap = 0
    for gt in candidates:
        ov = _overlap(
            window["start_frame"],
            window["end_frame"],
            gt["raw_start_frame"],
            gt["raw_end_frame"],
        )
        if ov > best_overlap:
            best = gt
            best_overlap = ov
    return best, best_overlap


def _mask_index_by_row(dataset_row_indices: np.ndarray) -> dict[int, int]:
    return {int(row): idx for idx, row in enumerate(np.asarray(dataset_row_indices).reshape(-1).tolist())}


def audit_windows(contact_labels_path: str, selector_windows_path: str) -> dict[str, Any]:
    labels = _load_npz(contact_labels_path)
    windows_pack = _load_npz(selector_windows_path)

    gt_segments = _records(labels, "segments")
    pred_windows = _records(windows_pack, "windows")
    gt_mask = np.asarray(labels["gt_contact_mask"], dtype=np.uint8)
    lengths = np.asarray(labels["lengths"], dtype=np.int64).reshape(-1)
    label_rows = np.asarray(labels["dataset_row_indices"], dtype=np.int64).reshape(-1)
    row_to_mask_index = _mask_index_by_row(label_rows)

    gt_by_group: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for gt in gt_segments:
        gt_by_group.setdefault(_group_key(gt), []).append(gt)

    windows_by_seq: dict[int, list[dict[str, Any]]] = {}
    for window in pred_windows:
        windows_by_seq.setdefault(_seq_key(window), []).append(window)

    recalled_gt = 0
    gt_debug = []
    center_distances = []
    for gt in gt_segments:
        candidates = gt_by_group.get(_group_key(gt), [])
        matching_windows = [
            window
            for window in pred_windows
            if _group_key(window) == _group_key(gt)
            and _overlap(window["start_frame"], window["end_frame"], gt["raw_start_frame"], gt["raw_end_frame"]) > 0
        ]
        best_window = None
        best_overlap = 0
        for window in matching_windows:
            ov = _overlap(window["start_frame"], window["end_frame"], gt["raw_start_frame"], gt["raw_end_frame"])
            if ov > best_overlap:
                best_window = window
                best_overlap = ov
        matched = best_window is not None
        recalled_gt += int(matched)
        gt_debug.append(
            {
                "gt_segment": gt,
                "matched": bool(matched),
                "best_overlap": int(best_overlap),
                "best_window": best_window,
            }
        )
        _ = candidates

    total_gt_contact_frames = 0
    covered_gt_contact_frames = 0
    for row, mask_index in row_to_mask_index.items():
        valid_len = int(lengths[mask_index])
        for hand_id in range(len(HAND_SIDE_NAMES)):
            for region_id in range(len(TARGET_REGION_NAMES)):
                gt_seq = gt_mask[mask_index, hand_id, region_id, :valid_len].astype(bool)
                total_gt_contact_frames += int(gt_seq.sum())
                if not gt_seq.any():
                    continue
                covered = np.zeros(valid_len, dtype=bool)
                for window in pred_windows:
                    if (
                        int(window["dataset_row_index"]) == row
                        and int(window["hand_side_id"]) == hand_id
                        and int(window["target_region_id"]) == region_id
                    ):
                        start = max(0, min(valid_len, int(window["start_frame"])))
                        end = max(0, min(valid_len, int(window["end_frame"])))
                        if end > start:
                            covered[start:end] = True
                covered_gt_contact_frames += int((gt_seq & covered).sum())

    per_window_debug = []
    matched_window_count = 0
    false_positive_count = 0
    purity_values = []
    region_match_den = 0
    region_match_num = 0
    for window in pred_windows:
        best_gt, best_overlap = _best_gt_match(window, gt_by_group)
        matched = best_gt is not None and best_overlap > 0
        matched_window_count += int(matched)
        false_positive_count += int(not matched)
        if matched:
            center_distances.append(abs(int(window["center_frame"]) - int(best_gt["center_frame"])))

        mask_index = row_to_mask_index.get(int(window["dataset_row_index"]))
        purity = 0.0
        if mask_index is not None:
            valid_len = int(lengths[mask_index])
            start = max(0, min(valid_len, int(window["start_frame"])))
            end = max(0, min(valid_len, int(window["end_frame"])))
            if end > start:
                gt_seq = gt_mask[
                    mask_index,
                    int(window["hand_side_id"]),
                    int(window["target_region_id"]),
                    start:end,
                ].astype(bool)
                purity = float(gt_seq.mean())
        purity_values.append(purity)

        any_same_hand_overlap = False
        best_any_region = None
        best_any_region_overlap = 0
        for gt in gt_segments:
            if int(gt["dataset_row_index"]) != int(window["dataset_row_index"]):
                continue
            if int(gt["hand_side_id"]) != int(window["hand_side_id"]):
                continue
            ov = _overlap(window["start_frame"], window["end_frame"], gt["raw_start_frame"], gt["raw_end_frame"])
            if ov > best_any_region_overlap:
                any_same_hand_overlap = ov > 0
                best_any_region = gt
                best_any_region_overlap = ov
        if any_same_hand_overlap and best_any_region is not None:
            region_match_den += 1
            region_match_num += int(int(best_any_region["target_region_id"]) == int(window["target_region_id"]))

        per_window_debug.append(
            {
                "window": window,
                "matched_gt_segment": best_gt,
                "best_overlap": int(best_overlap),
                "is_false_positive": bool(not matched),
                "window_contact_purity": float(purity),
                "best_same_hand_any_region_gt": best_any_region,
                "best_same_hand_any_region_overlap": int(best_any_region_overlap),
            }
        )

    num_sequences = int(len(label_rows))
    zero_window_sequences = [
        int(row)
        for row in label_rows.tolist()
        if len(windows_by_seq.get(int(row), [])) == 0
    ]
    metrics = {
        "num_sequences": num_sequences,
        "num_gt_segments": int(len(gt_segments)),
        "num_pred_windows": int(len(pred_windows)),
        "gt_segment_recall": float(recalled_gt / max(len(gt_segments), 1)),
        "gt_contact_frame_coverage": float(covered_gt_contact_frames / max(total_gt_contact_frames, 1)),
        "avg_center_distance": float(mean(center_distances)) if center_distances else 0.0,
        "zero_window_sequence_ratio": float(len(zero_window_sequences) / max(num_sequences, 1)),
        "window_contact_purity": float(mean(purity_values)) if purity_values else 0.0,
        "window_region_match_ratio": float(region_match_num / max(region_match_den, 1)),
        "false_positive_window_ratio": float(false_positive_count / max(len(pred_windows), 1)),
        "matched_window_ratio": float(matched_window_count / max(len(pred_windows), 1)),
        "total_gt_contact_frames": int(total_gt_contact_frames),
        "covered_gt_contact_frames": int(covered_gt_contact_frames),
    }
    return {
        "artifact": "selector_audit_v2",
        "contact_labels_path": contact_labels_path,
        "selector_windows_path": selector_windows_path,
        "metrics": metrics,
        "zero_window_dataset_row_indices": zero_window_sequences,
        "per_gt_segment": gt_debug,
        "per_window": per_window_debug,
        "notes": {
            "gt_source": "direct binary GT contact labels from contact_labels.py",
            "matching": "sample/dataset_row_index + hand + region, max temporal overlap, no Hungarian matching",
            "interval_semantics": "[start_frame, end_frame)",
        },
    }


def save_audit_json(path: str, payload: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
