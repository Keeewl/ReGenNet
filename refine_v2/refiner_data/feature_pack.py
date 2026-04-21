"""Build fast window feature samples from precomputed refine_v2 artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

from refine_v2.data.schema import TARGET_REGION_NAMES
from refine_v2.refiner_data.sanity_checks import validate_feature_sample, validate_window_bounds
from refine_v2.refiner_data.schema import TOPK_SCORE_FIELDS


def topk_scores_numeric(window: dict[str, Any]) -> np.ndarray:
    scores = list(window.get("topk_region_scores", []))
    ids = list(window.get("topk_target_region_ids", []))
    if len(scores) != len(ids):
        raise ValueError(
            f"topk_region_scores length {len(scores)} does not match topk ids length {len(ids)} "
            f"for window_index={window.get('window_index')}."
        )
    out = []
    for item in scores:
        row = []
        for field in TOPK_SCORE_FIELDS:
            if field not in item:
                raise KeyError(
                    f"topk_region_scores entry missing '{field}' for window_index={window.get('window_index')}."
                )
            row.append(float(item[field]))
        out.append(row)
    return np.asarray(out, dtype=np.float32)


def build_window_feature_sample(
    *,
    window: dict[str, Any],
    manifest_record: dict[str, Any],
    reaction_pack: dict[str, Any],
    label_index: int,
    selector_index: int,
    gt_contact_mask: np.ndarray,
    gt_min_region_dist: np.ndarray,
    pred_contact_mask: np.ndarray,
    pred_min_region_dist: np.ndarray,
    strict_checks: bool = True,
) -> dict[str, Any]:
    row = int(window["dataset_row_index"])
    hand_id = int(window["hand_side_id"])
    start = int(window["start_frame"])
    end = int(window["end_frame"])
    length = int(np.asarray(reaction_pack["lengths"][row]))
    motion_num_frames = int(np.asarray(reaction_pack["actor_motion"][row]).shape[-1])
    validate_window_bounds(window, length=length, motion_num_frames=motion_num_frames)

    actor_motion = np.asarray(reaction_pack["actor_motion"][row], dtype=np.float32)
    coarse_motion = np.asarray(reaction_pack["reactor_coarse"][row], dtype=np.float32)
    gt_motion = np.asarray(reaction_pack["reactor_gt"][row], dtype=np.float32)

    window_length = end - start
    sample = {
        "actor_motion_window": actor_motion[:, :, start:end].astype(np.float32),
        "coarse_motion_window": coarse_motion[:, :, start:end].astype(np.float32),
        "gt_motion_window": gt_motion[:, :, start:end].astype(np.float32),
        "coarse_region_contact_mask_window": np.asarray(
            pred_contact_mask[selector_index, hand_id, :, start:end], dtype=np.float32
        ),
        "coarse_min_region_dist_window": np.asarray(
            pred_min_region_dist[selector_index, hand_id, :, start:end], dtype=np.float32
        ),
        "gt_region_contact_mask_window": np.asarray(
            gt_contact_mask[label_index, hand_id, :, start:end], dtype=np.float32
        ),
        "gt_min_region_dist_window": np.asarray(
            gt_min_region_dist[label_index, hand_id, :, start:end], dtype=np.float32
        ),
        "valid_mask": np.ones((window_length,), dtype=bool),
        "window_length": int(window_length),
        "start_frame": start,
        "end_frame": end,
        "raw_start_frame": int(window.get("raw_start_frame", start)),
        "raw_end_frame": int(window.get("raw_end_frame", end)),
        "dataset_row_index": row,
        "sample_index": int(window.get("sample_index", manifest_record.get("sample_index", row))),
        "dataset_key": str(window.get("dataset_key", manifest_record.get("dataset_key", f"sample_{row}"))),
        "action_type": str(manifest_record.get("action_type", manifest_record.get("action_name", ""))),
        "action_label": str(manifest_record.get("action_label", "")),
        "action_name": str(manifest_record.get("action_name", manifest_record.get("action_type", ""))),
        "bucket_label": str(manifest_record.get("bucket_label", "")),
        "is_gt_positive": bool(manifest_record.get("is_gt_positive", False)),
        "is_pred_positive": bool(manifest_record.get("is_pred_positive", False)),
        "hand_side": str(window.get("hand_side", "")),
        "hand_side_id": hand_id,
        "primary_target_region": str(window.get("primary_target_region", window.get("target_region", ""))),
        "primary_target_region_id": int(window.get("primary_target_region_id", window.get("target_region_id", -1))),
        "topk_target_regions": [str(x) for x in window.get("topk_target_regions", [])],
        "topk_target_region_ids": np.asarray(window.get("topk_target_region_ids", []), dtype=np.int64),
        "topk_region_scores": list(window.get("topk_region_scores", [])),
        "topk_region_scores_numeric": topk_scores_numeric(window),
        "region_score_table": list(window.get("region_score_table", [])),
        "window_index": int(window.get("window_index", -1)),
        "sequence_window_index": int(window.get("sequence_window_index", -1)),
        "target_region_names": list(TARGET_REGION_NAMES),
    }
    if strict_checks:
        validate_feature_sample(sample)
    return sample
