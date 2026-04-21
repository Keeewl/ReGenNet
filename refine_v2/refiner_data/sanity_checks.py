"""Sanity checks for refine_v2 refiner window data."""

from __future__ import annotations

from typing import Any

import numpy as np

from refine_v2.data.schema import RESTORED_PAIR_SPACE, TARGET_REGION_NAMES, loads_metadata


def normalize_space_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return normalize_space_value(value.item())
        if value.size == 0:
            return ""
        return normalize_space_value(value.reshape(-1)[0])
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def require_restored_pair_space(value: Any, *, context: str):
    actual = normalize_space_value(value)
    if actual != RESTORED_PAIR_SPACE:
        raise ValueError(
            f"{context} must declare space_definition='{RESTORED_PAIR_SPACE}', got '{actual or '<missing>'}'. "
            "The refiner dataset only consumes already-restored artifacts."
        )


def require_keys(mapping: Any, keys: list[str] | tuple[str, ...], *, context: str):
    if hasattr(mapping, "files"):
        have = set(str(key) for key in mapping.files)
    elif hasattr(mapping, "keys"):
        have = set(str(key) for key in mapping.keys())
    else:
        raise TypeError(f"{context} does not expose keys/files.")
    missing = [key for key in keys if key not in have]
    if missing:
        raise KeyError(f"{context} missing required fields: {', '.join(missing)}")


def optional_metadata_space(pack: Any) -> str:
    if "metadata_json" not in getattr(pack, "files", []):
        return ""
    try:
        meta = loads_metadata(pack["metadata_json"])
    except Exception:
        return ""
    return str(meta.get("space_definition", ""))


def validate_topk_fields(window: dict[str, Any], *, strict: bool = True):
    topk_ids = [int(x) for x in window.get("topk_target_region_ids", [])]
    topk_regions = [str(x) for x in window.get("topk_target_regions", [])]
    topk_scores = list(window.get("topk_region_scores", []))
    if not topk_ids:
        raise ValueError(f"window_index={window.get('window_index')} has empty topk_target_region_ids.")
    if len(topk_regions) != len(topk_ids):
        raise ValueError(
            f"window_index={window.get('window_index')} top-k region id/name length mismatch: "
            f"{len(topk_ids)} ids vs {len(topk_regions)} names."
        )
    if len(topk_scores) != len(topk_ids):
        raise ValueError(
            f"window_index={window.get('window_index')} top-k score/id length mismatch: "
            f"{len(topk_scores)} scores vs {len(topk_ids)} ids."
        )
    for rid in topk_ids:
        if rid < 0 or rid >= len(TARGET_REGION_NAMES):
            raise ValueError(f"window_index={window.get('window_index')} has invalid region id: {rid}")
    primary = int(window.get("primary_target_region_id", window.get("target_region_id", -1)))
    if primary < 0 or primary >= len(TARGET_REGION_NAMES):
        raise ValueError(f"window_index={window.get('window_index')} has invalid primary region id: {primary}")
    if primary not in topk_ids:
        message = (
            f"window_index={window.get('window_index')} primary_target_region_id={primary} "
            f"is not in topk_target_region_ids={topk_ids}."
        )
        if strict:
            raise ValueError(message)
        print(f"[refiner_data warning] {message}")


def validate_window_bounds(window: dict[str, Any], *, length: int, motion_num_frames: int):
    start = int(window["start_frame"])
    end = int(window["end_frame"])
    if start < 0 or end <= start:
        raise ValueError(f"Invalid window bounds [{start},{end}) for window_index={window.get('window_index')}.")
    if end > int(length):
        raise ValueError(
            f"Window [{start},{end}) exceeds sequence valid length={length} "
            f"for dataset_row_index={window.get('dataset_row_index')}."
        )
    if end > int(motion_num_frames):
        raise ValueError(
            f"Window [{start},{end}) exceeds motion frames={motion_num_frames} "
            f"for dataset_row_index={window.get('dataset_row_index')}."
        )


def validate_feature_sample(sample: dict[str, Any]):
    window_length = int(sample["window_length"])
    if sample["actor_motion_window"].shape[-1] != window_length:
        raise ValueError("actor_motion_window time dimension does not match window_length.")
    if sample["coarse_motion_window"].shape[-1] != window_length:
        raise ValueError("coarse_motion_window time dimension does not match window_length.")
    if sample["gt_motion_window"].shape[-1] != window_length:
        raise ValueError("gt_motion_window time dimension does not match window_length.")
    for key in (
        "coarse_region_contact_mask_window",
        "coarse_min_region_dist_window",
        "gt_region_contact_mask_window",
        "gt_min_region_dist_window",
    ):
        arr = np.asarray(sample[key])
        if arr.shape != (len(TARGET_REGION_NAMES), window_length):
            raise ValueError(f"{key} expected shape {(len(TARGET_REGION_NAMES), window_length)}, got {arr.shape}.")
    valid = np.asarray(sample["valid_mask"])
    if valid.shape != (window_length,):
        raise ValueError(f"valid_mask expected shape {(window_length,)}, got {valid.shape}.")
    if valid.dtype != np.bool_:
        raise ValueError(f"valid_mask must be bool, got {valid.dtype}.")
