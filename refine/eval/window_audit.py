"""Development-time audit helpers for deterministic Stage2-lite windows.

This module is intentionally scoped as a development-period audit tool.

- It is used to sanity-check whether the current deterministic window selector is behaving
  reasonably on `reaction_data`.
- It is not the final paper-metric implementation.
- When `gt_motion` is available, metrics such as `gt_contact_coverage`,
  `window_precision_proxy`, and `segment_center_distance` are computed by running the same
  selector again on `gt_motion`, obtaining proxy GT windows, and then comparing those
  proxy windows against predicted windows.
- As a result, the current evaluation should be read as selector-vs-selector proxy audit,
  not as strict GT contact-label evaluation.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

import torch

from refine.model.windows import DeterministicWindowSelector, WindowConfig


def _coverage_mask(items: list[dict[str, Any]], lengths: torch.Tensor, *, target_specific: bool):
    coverage: dict[Any, torch.Tensor] = {}
    for item in items:
        batch_index = int(item["batch_index"])
        hand_id = int(item["hand_side_id"])
        key = (batch_index, hand_id)
        if target_specific:
            key = (batch_index, hand_id, int(item["target_part_id"]))
        if key not in coverage:
            coverage[key] = torch.zeros(int(lengths[batch_index].item()), dtype=torch.bool)
        start = max(0, int(item["start_frame"]))
        end = min(int(lengths[batch_index].item()), int(item["end_frame"]))
        if end > start:
            coverage[key][start:end] = True
    return coverage


def _center_lists(items: list[dict[str, Any]], *, target_specific: bool):
    centers: dict[Any, list[int]] = {}
    for item in items:
        key = (int(item["batch_index"]), int(item["hand_side_id"]))
        if target_specific:
            key = (int(item["batch_index"]), int(item["hand_side_id"]), int(item["target_part_id"]))
        centers.setdefault(key, []).append(int(item["center_frame"]))
    return centers


def _target_switch_rate(window_items: list[dict[str, Any]], batch_size: int) -> float:
    slot_rates = []
    for batch_index in range(batch_size):
        for hand_id in range(2):
            items = [
                item for item in window_items
                if int(item["batch_index"]) == batch_index and int(item["hand_side_id"]) == hand_id
            ]
            items = sorted(items, key=lambda item: int(item["start_frame"]))
            if len(items) <= 1:
                slot_rates.append(0.0)
                continue
            switches = 0
            for prev, curr in zip(items[:-1], items[1:]):
                if int(prev["target_part_id"]) != int(curr["target_part_id"]):
                    switches += 1
            slot_rates.append(switches / max(len(items) - 1, 1))
    return float(mean(slot_rates)) if slot_rates else 0.0


def audit_windows(
    window_items,
    actor_motion,
    coarse_motion,
    lengths,
    restoration_meta,
    gt_motion=None,
    *,
    selector: DeterministicWindowSelector | None = None,
    config: WindowConfig | None = None,
):
    """Audit predicted windows with lightweight development-time summary metrics.

    If `gt_motion` is provided, the function re-runs the same deterministic selector on
    `gt_motion` and compares predicted windows against those GT-side proxy windows.
    This makes the GT-related outputs useful for development and debugging, but they are
    still proxy metrics rather than strict GT contact-label scores.
    """
    selector = selector or DeterministicWindowSelector(config=config)
    lengths = lengths.long()
    batch_size = int(lengths.shape[0])
    counts_per_seq = [0 for _ in range(batch_size)]
    counts_per_hand = {(batch_index, hand_id): 0 for batch_index in range(batch_size) for hand_id in range(2)}

    total_raw_lengths = []
    total_model_lengths = []
    total_merge_count = 0
    strict_count = 0
    near_count = 0

    seq_coverage = [torch.zeros(int(lengths[idx].item()), dtype=torch.bool) for idx in range(batch_size)]
    for item in window_items:
        batch_index = int(item["batch_index"])
        hand_id = int(item["hand_side_id"])
        counts_per_seq[batch_index] += 1
        counts_per_hand[(batch_index, hand_id)] += 1
        total_raw_lengths.append(int(item.get("raw_length", 0)))
        total_model_lengths.append(max(0, int(item["end_frame"]) - int(item["start_frame"])))
        total_merge_count += int(item.get("merge_count", 0))
        if item.get("window_state") == "strict":
            strict_count += 1
        else:
            near_count += 1
        start = max(0, int(item["start_frame"]))
        end = min(int(lengths[batch_index].item()), int(item["end_frame"]))
        if end > start:
            seq_coverage[batch_index][start:end] = True

    total_valid_frames = int(lengths.sum().item())
    covered_frames = int(sum(mask.sum().item() for mask in seq_coverage))
    windows_per_hand_values = list(counts_per_hand.values())

    stats = {
        "num_sequences": batch_size,
        "total_windows": int(len(window_items)),
        "windows_per_seq": float(mean(counts_per_seq)) if counts_per_seq else 0.0,
        "windows_per_hand": float(mean(windows_per_hand_values)) if windows_per_hand_values else 0.0,
        "covered_frame_ratio": float(covered_frames / max(total_valid_frames, 1)),
        "avg_raw_len": float(mean(total_raw_lengths)) if total_raw_lengths else 0.0,
        "avg_model_len": float(mean(total_model_lengths)) if total_model_lengths else 0.0,
        "num_zero_window_seq": int(sum(1 for count in counts_per_seq if count == 0)),
        "zero_window_seq_ratio": float(sum(1 for count in counts_per_seq if count == 0) / max(batch_size, 1)),
        "target_switch_rate": _target_switch_rate(window_items, batch_size),
        "short_segment_ratio": float(
            sum(1 for length in total_raw_lengths if length < selector.config.model_W) / max(len(total_raw_lengths), 1)
        ) if total_raw_lengths else 0.0,
        "merge_count": int(total_merge_count),
        "strict_window_ratio": float(strict_count / max(len(window_items), 1)) if window_items else 0.0,
        "near_window_ratio": float(near_count / max(len(window_items), 1)) if window_items else 0.0,
    }

    if gt_motion is None:
        return stats

    gt_result = selector.build_windows_for_batch(
        actor_motion=actor_motion,
        coarse_motion=gt_motion,
        lengths=lengths,
        restoration_meta=restoration_meta,
    )
    gt_items = gt_result["window_items"]
    pred_coverage = _coverage_mask(window_items, lengths, target_specific=True)
    gt_coverage = _coverage_mask(gt_items, lengths, target_specific=True)

    overlap_frames = 0
    pred_frames = 0
    gt_frames = 0
    for key, mask in pred_coverage.items():
        pred_frames += int(mask.sum().item())
        if key in gt_coverage:
            overlap_frames += int((mask & gt_coverage[key]).sum().item())
    for mask in gt_coverage.values():
        gt_frames += int(mask.sum().item())

    pred_centers = _center_lists(window_items, target_specific=True)
    gt_centers = _center_lists(gt_items, target_specific=True)
    fallback_gt_centers = _center_lists(gt_items, target_specific=False)
    center_distances = []
    for key, centers in pred_centers.items():
        gt_center_candidates = gt_centers.get(key, None)
        if gt_center_candidates is None:
            gt_center_candidates = fallback_gt_centers.get(key[:2], [])
        for center in centers:
            if gt_center_candidates:
                center_distances.append(min(abs(center - gt_center) for gt_center in gt_center_candidates))
            else:
                center_distances.append(selector.config.model_W)

    stats.update(
        {
            "gt_contact_coverage": float(overlap_frames / max(gt_frames, 1)),
            "window_precision_proxy": float(overlap_frames / max(pred_frames, 1)),
            "segment_center_distance": float(mean(center_distances)) if center_distances else 0.0,
            "pred_raw_len_mean": float(mean(total_raw_lengths)) if total_raw_lengths else 0.0,
            "pred_windows_per_seq_mean": float(mean(counts_per_seq)) if counts_per_seq else 0.0,
            "gt_windows_per_seq_mean": float(
                mean([sum(1 for item in gt_items if int(item["batch_index"]) == batch_index) for batch_index in range(batch_size)])
            ) if batch_size > 0 else 0.0,
        }
    )
    return stats


def summarize_window_audit(stats_list):
    """Aggregate multiple audit dictionaries into one development-time summary."""
    if not stats_list:
        return {}
    summary = {"n_runs": len(stats_list)}
    keys = sorted({key for stats in stats_list for key in stats.keys()})
    for key in keys:
        values = [stats[key] for stats in stats_list if key in stats]
        if not values:
            continue
        first = values[0]
        if isinstance(first, (int, float)):
            if key.startswith("num_") or key.endswith("_count") or key in {"total_windows", "merge_count"}:
                summary[key] = float(sum(values))
            else:
                summary[key] = float(mean(values))
        else:
            summary[key] = values[-1]
    return summary
