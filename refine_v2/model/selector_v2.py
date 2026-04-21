"""Deterministic binary-contact window selector for refine_v2."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .regions import region_map_summary
from refine_v2.data.contact_labels import compute_contact_for_batch
from refine_v2.data.restored_space import RestoredBodyModelForward
from refine_v2.data.schema import (
    DEFAULT_GAP_MERGE,
    DEFAULT_PER_HAND_MAX_WINDOWS,
    DEFAULT_PER_SEQ_MAX_WINDOWS,
    DEFAULT_RAW_L_MIN,
    DEFAULT_TAU_CONTACT,
    DEFAULT_WINDOW_SIZE,
    HAND_SIDE_NAMES,
    RESTORED_PAIR_SPACE,
    TARGET_REGION_NAMES,
    dumps_metadata,
    records_to_object_array,
)
from refine_v2.utils.progress import ProgressBar


def _window_bounds(center: int, valid_len: int, window_size: int) -> tuple[int, int]:
    valid_len = max(0, int(valid_len))
    window_size = max(1, int(window_size))
    if valid_len <= 0:
        return 0, 0
    if valid_len >= window_size:
        start = int(center) - window_size // 2
        start = max(0, min(start, valid_len - window_size))
        return int(start), int(start + window_size)
    return 0, int(valid_len)


def windows_from_segment(segment: dict[str, Any], valid_len: int, *, window_size: int) -> list[dict[str, Any]]:
    raw_start = int(segment["raw_start_frame"])
    raw_end = int(segment["raw_end_frame"])
    raw_len = int(segment["raw_length"])
    if raw_len > 45:
        centers = [
            raw_start + max(0, raw_len // 3),
            raw_start + max(0, (2 * raw_len) // 3),
        ]
    else:
        centers = [int(segment["center_frame"])]

    out = []
    seen_bounds: set[tuple[int, int]] = set()
    for center in centers:
        center = max(0, min(int(center), max(int(valid_len) - 1, 0)))
        start, end = _window_bounds(center, valid_len, window_size)
        if (start, end) in seen_bounds:
            continue
        seen_bounds.add((start, end))
        item = dict(segment)
        item.update(
            {
                "batch_index": int(segment.get("batch_index", -1)),
                "raw_start_frame": raw_start,
                "raw_end_frame": raw_end,
                "raw_length": raw_len,
                "start_frame": int(start),
                "end_frame": int(end),
                "center_frame": int(center),
                "model_window_size": int(window_size),
            }
        )
        out.append(item)
    return out


def _add_batch_index_to_segments(
    segments: list[dict[str, Any]],
    dataset_row_indices: list[int],
) -> list[dict[str, Any]]:
    row_to_batch = {int(row): idx for idx, row in enumerate(dataset_row_indices)}
    out = []
    for segment in segments:
        item = dict(segment)
        item["batch_index"] = int(row_to_batch.get(int(item["dataset_row_index"]), -1))
        out.append(item)
    return out


def _window_contact_ratio(window: dict[str, Any], pred_mask: np.ndarray) -> float:
    batch_index = int(window["batch_index"])
    if batch_index < 0:
        return 0.0
    hand_id = int(window["hand_side_id"])
    region_id = int(window["target_region_id"])
    start = int(window["start_frame"])
    end = int(window["end_frame"])
    if end <= start:
        return 0.0
    return float(np.asarray(pred_mask[batch_index, hand_id, region_id, start:end], dtype=bool).mean())


def _limit_windows(
    windows: list[dict[str, Any]],
    pred_mask: np.ndarray,
    *,
    per_hand_max_windows: int,
    per_seq_max_windows: int,
) -> list[dict[str, Any]]:
    def sort_key(item: dict[str, Any]):
        return (
            -int(item.get("raw_length", 0)),
            -float(item.get("contact_frame_ratio", 0.0)),
            int(item.get("raw_start_frame", 0)),
            int(item.get("hand_side_id", 0)),
            int(item.get("target_region_id", 0)),
            int(item.get("start_frame", 0)),
        )

    for item in windows:
        item["contact_frame_ratio"] = _window_contact_ratio(item, pred_mask)

    by_sample: dict[int, list[dict[str, Any]]] = {}
    for item in windows:
        by_sample.setdefault(int(item["dataset_row_index"]), []).append(item)

    kept: list[dict[str, Any]] = []
    for _, sample_items in sorted(by_sample.items()):
        hand_kept: list[dict[str, Any]] = []
        for hand_id in range(len(HAND_SIDE_NAMES)):
            hand_items = [item for item in sample_items if int(item["hand_side_id"]) == hand_id]
            hand_kept.extend(sorted(hand_items, key=sort_key)[: int(per_hand_max_windows)])
        seq_kept = sorted(hand_kept, key=sort_key)[: int(per_seq_max_windows)]
        kept.extend(sorted(seq_kept, key=lambda item: (int(item["dataset_row_index"]), int(item["start_frame"]), int(item["hand_side_id"]), int(item["target_region_id"]))))
    return kept


def build_windows_for_loader(
    loader,
    region_map: dict[str, np.ndarray],
    *,
    tau_contact: float = DEFAULT_TAU_CONTACT,
    gap_merge: int = DEFAULT_GAP_MERGE,
    raw_L_min: int = DEFAULT_RAW_L_MIN,
    window_size: int = DEFAULT_WINDOW_SIZE,
    per_hand_max_windows: int = DEFAULT_PER_HAND_MAX_WINDOWS,
    per_seq_max_windows: int = DEFAULT_PER_SEQ_MAX_WINDOWS,
    device: str = "cpu",
    frame_chunk: int = 1,
    target_chunk: int = 2048,
    show_progress: bool = True,
) -> dict[str, Any]:
    device_t = torch.device(device)
    body_forward = RestoredBodyModelForward(device=device_t)
    masks = []
    dists = []
    lengths_all: list[int] = []
    sample_indices_all: list[int] = []
    dataset_row_indices_all: list[int] = []
    dataset_keys_all: list[str] = []
    raw_segments_all: list[dict[str, Any]] = []
    windows_all: list[dict[str, Any]] = []
    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else None
    progress = ProgressBar("select_windows", total_samples, unit="samples", enabled=show_progress).start()

    for batch in loader:
        actor_motion = batch["actor_motion"].to(device_t)
        coarse_motion = batch["coarse_motion"].to(device_t)
        lengths = batch["lengths"].to(device_t)
        result = compute_contact_for_batch(
            actor_motion,
            coarse_motion,
            lengths,
            batch,
            region_map,
            tau_contact=tau_contact,
            gap_merge=gap_merge,
            raw_L_min=raw_L_min,
            body_forward=body_forward,
            frame_chunk=frame_chunk,
            target_chunk=target_chunk,
        )
        pred_mask = result["contact_mask"]
        batch_segments = _add_batch_index_to_segments(
            result["segments"],
            result["dataset_row_indices"],
        )
        batch_windows: list[dict[str, Any]] = []
        for segment in batch_segments:
            batch_index = int(segment["batch_index"])
            valid_len = int(result["lengths"][batch_index]) if batch_index >= 0 else 0
            batch_windows.extend(windows_from_segment(segment, valid_len, window_size=window_size))
        batch_windows = _limit_windows(
            batch_windows,
            pred_mask,
            per_hand_max_windows=per_hand_max_windows,
            per_seq_max_windows=per_seq_max_windows,
        )

        masks.append(pred_mask)
        dists.append(result["min_region_dist"])
        lengths_all.extend(int(x) for x in result["lengths"].tolist())
        sample_indices_all.extend(result["sample_indices"])
        dataset_row_indices_all.extend(result["dataset_row_indices"])
        dataset_keys_all.extend(result["dataset_keys"])
        raw_segments_all.extend(batch_segments)
        windows_all.extend(batch_windows)
        progress.update(len(result["lengths"]))
    progress.finish()

    metadata = {
        "artifact": "selector_windows_v2",
        "space_definition": RESTORED_PAIR_SPACE,
        "tau_contact": float(tau_contact),
        "gap_merge": int(gap_merge),
        "raw_L_min": int(raw_L_min),
        "window_size": int(window_size),
        "per_hand_max_windows": int(per_hand_max_windows),
        "per_seq_max_windows": int(per_seq_max_windows),
        "hand_side_names": HAND_SIDE_NAMES,
        "target_region_names": TARGET_REGION_NAMES,
        "region_map_summary": region_map_summary(region_map),
        "interval_semantics": "[start_frame, end_frame)",
        "ranking": "raw_length desc, contact_frame_ratio desc, raw_start_frame asc",
    }
    return {
        "pred_contact_mask": np.concatenate(masks, axis=0) if masks else np.zeros((0, 2, 6, 0), dtype=np.uint8),
        "pred_min_region_dist": np.concatenate(dists, axis=0) if dists else np.zeros((0, 2, 6, 0), dtype=np.float32),
        "lengths": np.asarray(lengths_all, dtype=np.int64),
        "sample_indices": np.asarray(sample_indices_all, dtype=np.int64),
        "dataset_row_indices": np.asarray(dataset_row_indices_all, dtype=np.int64),
        "dataset_key": np.asarray(dataset_keys_all, dtype=object),
        "raw_segments": records_to_object_array(raw_segments_all),
        "windows": records_to_object_array(windows_all),
        "space_definition": np.asarray(RESTORED_PAIR_SPACE),
        "hand_side_names": np.asarray(HAND_SIDE_NAMES, dtype=object),
        "target_region_names": np.asarray(TARGET_REGION_NAMES, dtype=object),
        "metadata_json": np.asarray(dumps_metadata(metadata)),
    }


def save_selector_windows(path: str, artifact: dict[str, Any]):
    np.savez_compressed(path, **artifact)
