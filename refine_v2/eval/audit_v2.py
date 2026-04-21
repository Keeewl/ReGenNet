"""Strict GT contact-label audit for refine_v2 selector windows."""

from __future__ import annotations

import json
from statistics import mean
from typing import Any

import numpy as np

from refine_v2.data.schema import (
    HAND_SIDE_NAMES,
    TARGET_REGION_NAMES,
    loads_metadata,
    object_array_to_records,
    to_jsonable,
)
from refine_v2.utils.progress import ProgressBar


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


def _hand_key(item: dict[str, Any]) -> tuple[int, int]:
    return (int(item["dataset_row_index"]), int(item["hand_side_id"]))


def _best_gt_match(window: dict[str, Any], gt_by_group: dict[tuple[int, int, int], list[dict[str, Any]]]):
    key = _group_key(window)
    return _best_gt_match_from_candidates(window, gt_by_group.get(key, []))


def _best_gt_match_from_candidates(window: dict[str, Any], candidates: list[dict[str, Any]]):
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


def _best_window_match_from_candidates(gt: dict[str, Any], candidates: list[dict[str, Any]]):
    best = None
    best_overlap = 0
    for window in candidates:
        ov = _overlap(
            window["start_frame"],
            window["end_frame"],
            gt["raw_start_frame"],
            gt["raw_end_frame"],
        )
        if ov > best_overlap:
            best = window
            best_overlap = ov
    return best, best_overlap


def _topk_region_ids(window: dict[str, Any]) -> set[int]:
    value = window.get("topk_target_region_ids")
    if value is None:
        value = window.get("top_k_target_region_ids")
    if value is None:
        return {int(window["target_region_id"])}
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (int, np.integer)):
        return {int(value)}
    try:
        ids = {int(item) for item in value}
    except TypeError:
        ids = {int(window["target_region_id"])}
    if not ids:
        ids.add(int(window["target_region_id"]))
    return ids


def _topk_region_names(window: dict[str, Any]) -> list[str]:
    value = window.get("topk_target_regions")
    if value is None:
        value = window.get("top_k_target_regions")
    if value is None:
        return [str(window.get("target_region", TARGET_REGION_NAMES[int(window["target_region_id"])]))]
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, str):
        return [value]
    try:
        names = [str(item) for item in value]
    except TypeError:
        names = [str(window.get("target_region", TARGET_REGION_NAMES[int(window["target_region_id"])]))]
    return names or [str(window.get("target_region", TARGET_REGION_NAMES[int(window["target_region_id"])]))]


def _topk_best_window_match_from_candidates(gt: dict[str, Any], candidates: list[dict[str, Any]]):
    gt_region_id = int(gt["target_region_id"])
    best = None
    best_overlap = 0
    for window in candidates:
        if gt_region_id not in _topk_region_ids(window):
            continue
        ov = _overlap(
            window["start_frame"],
            window["end_frame"],
            gt["raw_start_frame"],
            gt["raw_end_frame"],
        )
        if ov > best_overlap:
            best = window
            best_overlap = ov
    return best, best_overlap


def _mask_index_by_row(dataset_row_indices: np.ndarray) -> dict[int, int]:
    return {int(row): idx for idx, row in enumerate(np.asarray(dataset_row_indices).reshape(-1).tolist())}


def _load_optional_json_scalar(pack, key: str) -> dict[str, Any]:
    if key not in pack.files:
        return {}
    try:
        return loads_metadata(pack[key])
    except Exception:
        return {}


def _diagnostic_summary(
    strict_metrics: dict[str, Any],
    relaxed_metrics: dict[str, Any],
    region_error: dict[str, Any],
    selector_stats: dict[str, Any],
    gt_sequence_stats: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    zero_ratio = float(strict_metrics.get("zero_window_sequence_ratio", 0.0))
    if selector_stats:
        num_seq = max(int(selector_stats.get("num_sequences", 0)), 1)
        pred_contact_seq_ratio = float(selector_stats.get("num_sequences_with_pred_contact_frames", 0)) / num_seq
        raw_pre_seq_ratio = float(selector_stats.get("num_sequences_with_raw_segments_pre_filter", 0)) / num_seq
        raw_post_seq_ratio = float(selector_stats.get("num_sequences_with_raw_segments_post_filter", 0)) / num_seq
        win_pre_seq_ratio = float(selector_stats.get("num_sequences_with_windows_pre_cap", 0)) / num_seq
        win_post_seq_ratio = float(selector_stats.get("num_sequences_with_windows_post_cap", 0)) / num_seq
        hand_drops = int(selector_stats.get("num_windows_dropped_by_hand_cap", 0))
        seq_drops = int(selector_stats.get("num_windows_dropped_by_seq_cap", 0))
        pre_windows = max(int(selector_stats.get("num_windows_pre_cap", 0)), 1)
        cap_drop_ratio = float(hand_drops + seq_drops) / pre_windows
        if pred_contact_seq_ratio < 0.7:
            reasons.append("coarse_contact_mask_sparse")
        if raw_pre_seq_ratio - raw_post_seq_ratio > 0.15:
            reasons.append("raw_L_min_filter_drops_many_sequences")
        if win_pre_seq_ratio - win_post_seq_ratio > 0.15 or cap_drop_ratio > 0.3:
            reasons.append("window_cap_drops_many_candidates")
        if zero_ratio > 0.3 and win_pre_seq_ratio < 0.7:
            reasons.append("pre_cap_candidate_coverage_low")
    else:
        pred_contact_seq_ratio = None
        raw_pre_seq_ratio = None
        raw_post_seq_ratio = None
        win_pre_seq_ratio = None
        win_post_seq_ratio = None
        cap_drop_ratio = None

    strict_recall = float(strict_metrics.get("gt_segment_recall", 0.0))
    hand_recall = float(relaxed_metrics.get("hand_only_gt_segment_recall", 0.0))
    time_recall = float(relaxed_metrics.get("time_only_gt_segment_recall", 0.0))
    wrong_region_ratio = float(region_error.get("same_hand_time_overlap_but_wrong_region_ratio", 0.0))
    wrong_hand_ratio = float(region_error.get("same_sample_time_overlap_but_wrong_hand_ratio", 0.0))
    if hand_recall - strict_recall > 0.1 or wrong_region_ratio > 0.25:
        reasons.append("region_assignment_mismatch")
    if time_recall - hand_recall > 0.1 or wrong_hand_ratio > 0.25:
        reasons.append("hand_assignment_mismatch")
    if not reasons:
        reasons.append("no_single_dominant_failure_layer")

    gt_positive_zero_ratio = float(gt_sequence_stats.get("gt_positive_zero_window_ratio", 0.0))
    gt_negative_nonzero_ratio = float(gt_sequence_stats.get("gt_negative_nonzero_window_ratio", 0.0))
    return {
        "likely_failure_layers": reasons,
        "selector_sequence_ratios": {
            "with_pred_contact_frames": pred_contact_seq_ratio,
            "with_raw_segments_pre_filter": raw_pre_seq_ratio,
            "with_raw_segments_post_filter": raw_post_seq_ratio,
            "with_windows_pre_cap": win_pre_seq_ratio,
            "with_windows_post_cap": win_post_seq_ratio,
            "cap_drop_ratio_over_pre_cap_windows": cap_drop_ratio,
        },
        "relaxed_recall_gaps": {
            "hand_only_minus_strict": float(hand_recall - strict_recall),
            "time_only_minus_hand_only": float(time_recall - hand_recall),
        },
        "gt_sequence_interpretation": {
            "gt_positive_zero_window_ratio": gt_positive_zero_ratio,
            "gt_negative_nonzero_window_ratio": gt_negative_nonzero_ratio,
            "gt_positive_zero_window_is_true_sequence_recall_miss": True,
            "gt_negative_nonzero_window_is_sequence_false_positive": True,
        },
        "summary_text": (
            "Use selector_stats_json to locate candidate loss before audit; "
            "large hand_only-strict gap indicates region mismatch, and large "
            "time_only-hand_only gap indicates hand mismatch. GT+ / Pred0 is "
            "the true zero-window recall miss bucket."
        ),
    }


def _diagnostic_summary_topk(
    strict_metrics: dict[str, Any],
    relaxed_metrics: dict[str, Any],
    topk_metrics: dict[str, Any],
) -> dict[str, Any]:
    strict_recall = float(strict_metrics.get("gt_segment_recall", 0.0))
    hand_recall = float(relaxed_metrics.get("hand_only_gt_segment_recall", 0.0))
    topk_recall = float(topk_metrics.get("topk_gt_segment_recall", 0.0))
    return {
        "strict_recall": strict_recall,
        "topk_gt_segment_recall": topk_recall,
        "hand_only_gt_segment_recall": hand_recall,
        "topk_minus_strict": float(topk_recall - strict_recall),
        "hand_only_minus_topk": float(hand_recall - topk_recall),
        "top1_miss_topk_hit_count": int(topk_metrics.get("top1_miss_topk_hit_count", 0)),
        "top1_miss_topk_hit_ratio": float(topk_metrics.get("top1_miss_topk_hit_ratio", 0.0)),
        "interpretation": (
            "If top-k recall closes most of the gap between primary-region strict recall "
            "and hand-only recall, the time proposal is likely adequate and single-primary "
            "region credit is the main bottleneck."
        ),
    }


def _empty_region_confusion() -> dict[str, dict[str, int]]:
    cols = list(TARGET_REGION_NAMES) + ["unmatched"]
    return {str(row): {str(col): 0 for col in cols} for row in TARGET_REGION_NAMES}


def audit_windows(
    contact_labels_path: str,
    selector_windows_path: str,
    *,
    show_progress: bool = True,
) -> dict[str, Any]:
    labels = _load_npz(contact_labels_path)
    windows_pack = _load_npz(selector_windows_path)

    gt_segments = _records(labels, "segments")
    pred_windows = _records(windows_pack, "windows")
    gt_mask = np.asarray(labels["gt_contact_mask"], dtype=np.uint8)
    lengths = np.asarray(labels["lengths"], dtype=np.int64).reshape(-1)
    label_rows = np.asarray(labels["dataset_row_indices"], dtype=np.int64).reshape(-1)
    row_to_mask_index = _mask_index_by_row(label_rows)
    progress_total = len(gt_segments) + len(row_to_mask_index) + len(pred_windows)
    progress = ProgressBar("audit_windows", progress_total, unit="items", enabled=show_progress).start()

    gt_by_group: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    gt_by_hand: dict[tuple[int, int], list[dict[str, Any]]] = {}
    gt_by_seq: dict[int, list[dict[str, Any]]] = {}
    for gt in gt_segments:
        gt_by_group.setdefault(_group_key(gt), []).append(gt)
        gt_by_hand.setdefault(_hand_key(gt), []).append(gt)
        gt_by_seq.setdefault(_seq_key(gt), []).append(gt)

    windows_by_seq: dict[int, list[dict[str, Any]]] = {}
    windows_by_hand: dict[tuple[int, int], list[dict[str, Any]]] = {}
    windows_by_group: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for window in pred_windows:
        windows_by_seq.setdefault(_seq_key(window), []).append(window)
        windows_by_hand.setdefault(_hand_key(window), []).append(window)
        windows_by_group.setdefault(_group_key(window), []).append(window)

    pred_rows = {int(window["dataset_row_index"]) for window in pred_windows}
    gt_positive_rows = set()
    gt_negative_rows = set()
    gt_pred_buckets = {
        "gt_positive_pred_positive": 0,
        "gt_positive_pred_zero": 0,
        "gt_negative_pred_positive": 0,
        "gt_negative_pred_zero": 0,
    }
    for row, mask_index in row_to_mask_index.items():
        valid_len = int(lengths[mask_index])
        is_gt_positive = bool(gt_mask[mask_index, :, :, :valid_len].astype(bool).any())
        is_pred_positive = int(row) in pred_rows
        if is_gt_positive:
            gt_positive_rows.add(int(row))
            if is_pred_positive:
                gt_pred_buckets["gt_positive_pred_positive"] += 1
            else:
                gt_pred_buckets["gt_positive_pred_zero"] += 1
        else:
            gt_negative_rows.add(int(row))
            if is_pred_positive:
                gt_pred_buckets["gt_negative_pred_positive"] += 1
            else:
                gt_pred_buckets["gt_negative_pred_zero"] += 1

    recalled_gt = 0
    topk_recalled_gt = 0
    top1_miss_topk_hit_count = 0
    top1_miss_topk_hit_by_region = {str(name): 0 for name in TARGET_REGION_NAMES}
    hand_only_recalled_gt = 0
    time_only_recalled_gt = 0
    gt_debug = []
    center_distances = []
    for gt in gt_segments:
        best_window, best_overlap = _best_window_match_from_candidates(gt, windows_by_group.get(_group_key(gt), []))
        topk_best_window, topk_best_overlap = _topk_best_window_match_from_candidates(
            gt,
            windows_by_hand.get(_hand_key(gt), []),
        )
        best_hand_window, best_hand_overlap = _best_window_match_from_candidates(gt, windows_by_hand.get(_hand_key(gt), []))
        best_time_window, best_time_overlap = _best_window_match_from_candidates(gt, windows_by_seq.get(_seq_key(gt), []))
        matched = best_window is not None
        topk_matched = topk_best_window is not None and topk_best_overlap > 0
        hand_only_matched = best_hand_window is not None and best_hand_overlap > 0
        time_only_matched = best_time_window is not None and best_time_overlap > 0
        recalled_gt += int(matched)
        topk_recalled_gt += int(topk_matched)
        if not matched and topk_matched:
            top1_miss_topk_hit_count += 1
            top1_miss_topk_hit_by_region[str(gt["target_region"])] = (
                int(top1_miss_topk_hit_by_region.get(str(gt["target_region"]), 0)) + 1
            )
        hand_only_recalled_gt += int(hand_only_matched)
        time_only_recalled_gt += int(time_only_matched)
        gt_debug.append(
            {
                "gt_segment": gt,
                "matched": bool(matched),
                "best_overlap": int(best_overlap),
                "best_window": best_window,
                "topk_matched": bool(topk_matched),
                "topk_best_overlap": int(topk_best_overlap),
                "topk_best_window": topk_best_window,
                "hand_only_matched": bool(hand_only_matched),
                "hand_only_best_overlap": int(best_hand_overlap),
                "hand_only_best_window": best_hand_window,
                "time_only_matched": bool(time_only_matched),
                "time_only_best_overlap": int(best_time_overlap),
                "time_only_best_window": best_time_window,
            }
        )
        progress.update()

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
                for window in windows_by_group.get((int(row), int(hand_id), int(region_id)), []):
                    start = max(0, min(valid_len, int(window["start_frame"])))
                    end = max(0, min(valid_len, int(window["end_frame"])))
                    if end > start:
                        covered[start:end] = True
                covered_gt_contact_frames += int((gt_seq & covered).sum())
        progress.update()

    per_window_debug = []
    matched_window_count = 0
    false_positive_count = 0
    purity_values = []
    region_match_den = 0
    region_match_num = 0
    hand_only_window_match_count = 0
    time_only_window_match_count = 0
    wrong_region_count = 0
    same_hand_time_overlap_window_count = 0
    wrong_hand_count = 0
    same_sample_time_overlap_window_count = 0
    wrong_region_confusion = _empty_region_confusion()
    topk_window_match_count = 0
    topk_region_match_den = 0
    topk_region_match_num = 0
    strict_miss_hand_hit_topk_total = 0
    strict_miss_hand_hit_topk_hit = 0
    strict_miss_hand_hit_topk_by_pred_region = {
        str(name): {"total": 0, "best_gt_region_in_topk": 0} for name in TARGET_REGION_NAMES
    }
    strict_miss_hand_hit_topk_by_gt_region = {
        str(name): {"total": 0, "best_gt_region_in_topk": 0} for name in TARGET_REGION_NAMES
    }
    for window in pred_windows:
        best_gt, best_overlap = _best_gt_match(window, gt_by_group)
        best_hand_gt, best_hand_overlap = _best_gt_match_from_candidates(window, gt_by_hand.get(_hand_key(window), []))
        best_time_gt, best_time_overlap = _best_gt_match_from_candidates(window, gt_by_seq.get(_seq_key(window), []))
        matched = best_gt is not None and best_overlap > 0
        hand_only_window_matched = best_hand_gt is not None and best_hand_overlap > 0
        time_only_window_matched = best_time_gt is not None and best_time_overlap > 0
        topk_region_ids = _topk_region_ids(window)
        topk_region_names = _topk_region_names(window)
        topk_window_matched = False
        topk_best_gt = None
        topk_best_overlap = 0
        for gt in gt_by_hand.get(_hand_key(window), []):
            if int(gt["target_region_id"]) not in topk_region_ids:
                continue
            ov = _overlap(
                window["start_frame"],
                window["end_frame"],
                gt["raw_start_frame"],
                gt["raw_end_frame"],
            )
            if ov > topk_best_overlap:
                topk_best_gt = gt
                topk_best_overlap = ov
        topk_window_matched = topk_best_gt is not None and topk_best_overlap > 0
        matched_window_count += int(matched)
        false_positive_count += int(not matched)
        hand_only_window_match_count += int(hand_only_window_matched)
        time_only_window_match_count += int(time_only_window_matched)
        topk_window_match_count += int(topk_window_matched)
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

        any_same_hand_overlap = hand_only_window_matched
        best_any_region = best_hand_gt
        best_any_region_overlap = best_hand_overlap
        if any_same_hand_overlap and best_any_region is not None:
            region_match_den += 1
            topk_region_match_den += 1
            same_hand_time_overlap_window_count += 1
            same_region = int(best_any_region["target_region_id"]) == int(window["target_region_id"])
            topk_same_region = int(best_any_region["target_region_id"]) in topk_region_ids
            region_match_num += int(same_region)
            topk_region_match_num += int(topk_same_region)
            wrong_region_count += int(not same_region)
            pred_region_name = str(window.get("primary_target_region", window.get("target_region", TARGET_REGION_NAMES[int(window["target_region_id"])])))
            gt_region_name = str(best_any_region["target_region"])
            if pred_region_name not in wrong_region_confusion:
                wrong_region_confusion[pred_region_name] = {str(col): 0 for col in list(TARGET_REGION_NAMES) + ["unmatched"]}
            if gt_region_name not in wrong_region_confusion[pred_region_name]:
                wrong_region_confusion[pred_region_name][gt_region_name] = 0
            wrong_region_confusion[pred_region_name][gt_region_name] += 1
            if not matched:
                strict_miss_hand_hit_topk_total += 1
                if pred_region_name not in strict_miss_hand_hit_topk_by_pred_region:
                    strict_miss_hand_hit_topk_by_pred_region[pred_region_name] = {
                        "total": 0,
                        "best_gt_region_in_topk": 0,
                    }
                if gt_region_name not in strict_miss_hand_hit_topk_by_gt_region:
                    strict_miss_hand_hit_topk_by_gt_region[gt_region_name] = {
                        "total": 0,
                        "best_gt_region_in_topk": 0,
                    }
                strict_miss_hand_hit_topk_by_pred_region[pred_region_name]["total"] += 1
                strict_miss_hand_hit_topk_by_gt_region[gt_region_name]["total"] += 1
                if topk_same_region:
                    strict_miss_hand_hit_topk_hit += 1
                    strict_miss_hand_hit_topk_by_pred_region[pred_region_name]["best_gt_region_in_topk"] += 1
                    strict_miss_hand_hit_topk_by_gt_region[gt_region_name]["best_gt_region_in_topk"] += 1
        else:
            pred_region_name = str(window.get("primary_target_region", window.get("target_region", TARGET_REGION_NAMES[int(window["target_region_id"])])))
            if pred_region_name not in wrong_region_confusion:
                wrong_region_confusion[pred_region_name] = {str(col): 0 for col in list(TARGET_REGION_NAMES) + ["unmatched"]}
            wrong_region_confusion[pred_region_name]["unmatched"] += 1

        if time_only_window_matched and best_time_gt is not None:
            same_sample_time_overlap_window_count += 1
            same_hand = int(best_time_gt["hand_side_id"]) == int(window["hand_side_id"])
            wrong_hand_count += int(not same_hand)

        per_window_debug.append(
            {
                "window": window,
                "matched_gt_segment": best_gt,
                "best_overlap": int(best_overlap),
                "is_false_positive": bool(not matched),
                "window_contact_purity": float(purity),
                "topk_target_region_ids": sorted(int(item) for item in topk_region_ids),
                "topk_target_regions": topk_region_names,
                "topk_matched": bool(topk_window_matched),
                "topk_best_overlap": int(topk_best_overlap),
                "topk_best_gt_segment": topk_best_gt,
                "best_same_hand_any_region_gt": best_any_region,
                "best_same_hand_any_region_overlap": int(best_any_region_overlap),
                "hand_only_matched": bool(hand_only_window_matched),
                "hand_only_best_overlap": int(best_hand_overlap),
                "hand_only_best_gt_segment": best_hand_gt,
                "time_only_matched": bool(time_only_window_matched),
                "time_only_best_overlap": int(best_time_overlap),
                "time_only_best_gt_segment": best_time_gt,
            }
        )
        progress.update()

    num_sequences = int(len(label_rows))
    zero_window_sequences = [
        int(row)
        for row in label_rows.tolist()
        if len(windows_by_seq.get(int(row), [])) == 0
    ]
    strict_metrics = {
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
    relaxed_metrics = {
        "hand_only_gt_segment_recall": float(hand_only_recalled_gt / max(len(gt_segments), 1)),
        "hand_only_window_match_ratio": float(hand_only_window_match_count / max(len(pred_windows), 1)),
        "time_only_gt_segment_recall": float(time_only_recalled_gt / max(len(gt_segments), 1)),
        "time_only_window_match_ratio": float(time_only_window_match_count / max(len(pred_windows), 1)),
    }
    strict_miss_count = max(len(gt_segments) - recalled_gt, 0)
    topk_metrics = {
        "topk_gt_segment_recall": float(topk_recalled_gt / max(len(gt_segments), 1)),
        "topk_window_match_ratio": float(topk_window_match_count / max(len(pred_windows), 1)),
        "topk_region_match_ratio": float(topk_region_match_num / max(topk_region_match_den, 1)),
        "top1_miss_topk_hit_count": int(top1_miss_topk_hit_count),
        "top1_miss_topk_hit_ratio": float(top1_miss_topk_hit_count / max(strict_miss_count, 1)),
        "top1_miss_topk_hit_ratio_over_all_gt": float(top1_miss_topk_hit_count / max(len(gt_segments), 1)),
        "top1_miss_topk_hit_by_region": top1_miss_topk_hit_by_region,
        "topk_recalled_gt_count": int(topk_recalled_gt),
        "strict_missed_gt_count": int(strict_miss_count),
    }
    region_error_analysis = {
        "same_hand_time_overlap_window_count": int(same_hand_time_overlap_window_count),
        "same_hand_time_overlap_but_wrong_region_count": int(wrong_region_count),
        "same_hand_time_overlap_but_wrong_region_ratio": float(wrong_region_count / max(same_hand_time_overlap_window_count, 1)),
        "same_sample_time_overlap_window_count": int(same_sample_time_overlap_window_count),
        "same_sample_time_overlap_but_wrong_hand_count": int(wrong_hand_count),
        "same_sample_time_overlap_but_wrong_hand_ratio": float(wrong_hand_count / max(same_sample_time_overlap_window_count, 1)),
    }
    topk_wrong_region_recovery = {
        "strict_miss_hand_only_hit_window_count": int(strict_miss_hand_hit_topk_total),
        "strict_miss_hand_only_hit_topk_region_count": int(strict_miss_hand_hit_topk_hit),
        "strict_miss_hand_only_hit_topk_region_ratio": float(
            strict_miss_hand_hit_topk_hit / max(strict_miss_hand_hit_topk_total, 1)
        ),
        "by_pred_primary_region": strict_miss_hand_hit_topk_by_pred_region,
        "by_best_gt_region": strict_miss_hand_hit_topk_by_gt_region,
    }
    gt_sequence_stats = {
        "num_gt_positive_sequences": int(len(gt_positive_rows)),
        "num_gt_negative_sequences": int(len(gt_negative_rows)),
        "gt_positive_zero_window_count": int(gt_pred_buckets["gt_positive_pred_zero"]),
        "gt_positive_zero_window_ratio": float(gt_pred_buckets["gt_positive_pred_zero"] / max(len(gt_positive_rows), 1)),
        "gt_negative_zero_window_count": int(gt_pred_buckets["gt_negative_pred_zero"]),
        "gt_negative_zero_window_ratio": float(gt_pred_buckets["gt_negative_pred_zero"] / max(len(gt_negative_rows), 1)),
        "gt_negative_nonzero_window_count": int(gt_pred_buckets["gt_negative_pred_positive"]),
        "gt_negative_nonzero_window_ratio": float(gt_pred_buckets["gt_negative_pred_positive"] / max(len(gt_negative_rows), 1)),
        "gt_positive_pred_positive_count": int(gt_pred_buckets["gt_positive_pred_positive"]),
        "gt_positive_pred_zero_count": int(gt_pred_buckets["gt_positive_pred_zero"]),
        "gt_negative_pred_positive_count": int(gt_pred_buckets["gt_negative_pred_positive"]),
        "gt_negative_pred_zero_count": int(gt_pred_buckets["gt_negative_pred_zero"]),
    }
    metrics = dict(strict_metrics)
    metrics.update(relaxed_metrics)
    metrics.update(topk_metrics)
    metrics.update(region_error_analysis)
    metrics.update(gt_sequence_stats)
    selector_stats = _load_optional_json_scalar(windows_pack, "selector_stats_json")
    selector_params = _load_optional_json_scalar(windows_pack, "selector_params_json")
    diagnostic_summary = _diagnostic_summary(
        strict_metrics,
        relaxed_metrics,
        region_error_analysis,
        selector_stats,
        gt_sequence_stats,
    )
    diagnostic_summary_topk = _diagnostic_summary_topk(strict_metrics, relaxed_metrics, topk_metrics)
    progress.finish()
    return {
        "artifact": "selector_audit_v2",
        "contact_labels_path": contact_labels_path,
        "selector_windows_path": selector_windows_path,
        "selector_params": selector_params,
        "selector_stats_summary": selector_stats,
        "strict_metrics": strict_metrics,
        "relaxed_metrics": relaxed_metrics,
        "topk_metrics": topk_metrics,
        "region_error_analysis": region_error_analysis,
        "gt_sequence_stats": gt_sequence_stats,
        "wrong_region_confusion_matrix": wrong_region_confusion,
        "topk_wrong_region_recovery": topk_wrong_region_recovery,
        "metrics": metrics,
        "diagnostic_summary": diagnostic_summary,
        "diagnostic_summary_topk": diagnostic_summary_topk,
        "zero_window_dataset_row_indices": zero_window_sequences,
        "per_gt_segment": gt_debug,
        "per_window": per_window_debug,
        "notes": {
            "gt_source": "direct binary GT contact labels from contact_labels.py",
            "matching": "sample/dataset_row_index + hand + region, max temporal overlap, no Hungarian matching",
            "topk_matching": "sample/dataset_row_index + hand + temporal overlap, with GT region contained in pred window top-k attributed regions",
            "interval_semantics": "[start_frame, end_frame)",
        },
    }


def save_audit_json(path: str, payload: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)
