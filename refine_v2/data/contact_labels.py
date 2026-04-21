"""Binary mesh-region contact labels for refine_v2."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .schema import (
    DEFAULT_GAP_MERGE,
    DEFAULT_RAW_L_MIN,
    DEFAULT_TAU_CONTACT,
    HAND_SIDE_IDS,
    HAND_SIDE_NAMES,
    RESTORED_PAIR_SPACE,
    TARGET_REGION_IDS,
    TARGET_REGION_NAMES,
    dumps_metadata,
    records_to_object_array,
)
from .restored_space import RestoredBodyModelForward, motions_to_vertices, restore_pair_if_needed
from refine_v2.model.regions import region_map_summary
from refine_v2.utils.progress import ProgressBar


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _batch_strings(batch: dict[str, Any], key: str, batch_size: int, fallback_prefix: str) -> list[str]:
    if key not in batch:
        return [f"{fallback_prefix}_{idx}" for idx in range(batch_size)]
    values = batch[key]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        values = [values for _ in range(batch_size)]
    out = []
    for idx, value in enumerate(values):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        out.append(str(value) if value is not None else f"{fallback_prefix}_{idx}")
    return out


def _batch_ints(batch: dict[str, Any], key: str, batch_size: int) -> list[int]:
    if key not in batch:
        return list(range(batch_size))
    values = batch[key]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values).reshape(-1)
    return [int(v) for v in values.tolist()]


@torch.no_grad()
def region_to_region_min_distance(
    source_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    frame_chunk: int = 1,
    target_chunk: int = 2048,
) -> torch.Tensor:
    """Minimum distance per frame between two vertex sets.

    source_vertices/target_vertices have shape [B, V, 3, T]. Output is [B, T].
    The implementation chunks the target set and time dimension to keep the
    default CPU/GPU memory footprint predictable for large torso regions.
    """

    if source_vertices.dim() != 4 or target_vertices.dim() != 4:
        raise ValueError("vertices must have shape [B, V, 3, T].")
    if len(source_ids) == 0 or len(target_ids) == 0:
        raise ValueError("source_ids and target_ids must be non-empty.")

    device = source_vertices.device
    source_ids_t = torch.as_tensor(source_ids, dtype=torch.long, device=device)
    target_ids_t = torch.as_tensor(target_ids, dtype=torch.long, device=device)
    source_sel = source_vertices.index_select(1, source_ids_t)
    target_sel = target_vertices.index_select(1, target_ids_t)
    batch_size, _, _, num_frames = source_sel.shape
    out = source_vertices.new_full((batch_size, num_frames), float("inf"))

    frame_chunk = max(1, int(frame_chunk))
    target_chunk = max(1, int(target_chunk))
    for frame_start in range(0, num_frames, frame_chunk):
        frame_end = min(num_frames, frame_start + frame_chunk)
        chunk_len = frame_end - frame_start
        src = source_sel[:, :, :, frame_start:frame_end].permute(0, 3, 1, 2)
        src = src.reshape(batch_size * chunk_len, source_sel.shape[1], 3).contiguous()
        local = source_vertices.new_full((batch_size * chunk_len,), float("inf"))
        for target_start in range(0, target_sel.shape[1], target_chunk):
            target_end = min(target_sel.shape[1], target_start + target_chunk)
            tgt = target_sel[:, target_start:target_end, :, frame_start:frame_end].permute(0, 3, 1, 2)
            tgt = tgt.reshape(batch_size * chunk_len, target_end - target_start, 3).contiguous()
            dist = torch.cdist(src, tgt).amin(dim=(1, 2))
            local = torch.minimum(local, dist)
        out[:, frame_start:frame_end] = local.view(batch_size, chunk_len)
    return out


def binary_segments_from_mask(
    mask_1d: np.ndarray,
    *,
    sample_index: int,
    dataset_row_index: int,
    dataset_key: str,
    hand_side: str,
    hand_side_id: int,
    target_region: str,
    target_region_id: int,
    gap_merge: int = DEFAULT_GAP_MERGE,
    raw_L_min: int = DEFAULT_RAW_L_MIN,
) -> list[dict[str, Any]]:
    mask_1d = np.asarray(mask_1d).astype(bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < mask_1d.size:
        if not mask_1d[idx]:
            idx += 1
            continue
        start = idx
        idx += 1
        while idx < mask_1d.size and mask_1d[idx]:
            idx += 1
        runs.append((start, idx))

    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= int(gap_merge):
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    records = []
    for start, end in merged:
        raw_length = int(end - start)
        if raw_length < int(raw_L_min):
            continue
        records.append(
            {
                "sample_index": int(sample_index),
                "dataset_row_index": int(dataset_row_index),
                "dataset_key": str(dataset_key),
                "hand_side": str(hand_side),
                "hand_side_id": int(hand_side_id),
                "target_region": str(target_region),
                "target_region_id": int(target_region_id),
                "raw_start_frame": int(start),
                "raw_end_frame": int(end),
                "raw_length": int(raw_length),
                "center_frame": int((start + end - 1) // 2),
            }
        )
    return records


def segments_from_contact_mask(
    contact_mask: np.ndarray,
    lengths: np.ndarray,
    sample_indices: list[int],
    dataset_row_indices: list[int],
    dataset_keys: list[str],
    *,
    gap_merge: int = DEFAULT_GAP_MERGE,
    raw_L_min: int = DEFAULT_RAW_L_MIN,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch_index in range(contact_mask.shape[0]):
        valid_len = int(lengths[batch_index])
        for hand_side, hand_id in HAND_SIDE_IDS.items():
            for region_name, region_id in TARGET_REGION_IDS.items():
                records.extend(
                    binary_segments_from_mask(
                        contact_mask[batch_index, hand_id, region_id, :valid_len],
                        sample_index=sample_indices[batch_index],
                        dataset_row_index=dataset_row_indices[batch_index],
                        dataset_key=dataset_keys[batch_index],
                        hand_side=hand_side,
                        hand_side_id=hand_id,
                        target_region=region_name,
                        target_region_id=region_id,
                        gap_merge=gap_merge,
                        raw_L_min=raw_L_min,
                    )
                )
    return records


@torch.no_grad()
def compute_contact_for_batch(
    actor_motion: torch.Tensor,
    reactor_motion: torch.Tensor,
    lengths: torch.Tensor,
    batch: dict[str, Any],
    region_map: dict[str, np.ndarray],
    *,
    tau_contact: float = DEFAULT_TAU_CONTACT,
    gap_merge: int = DEFAULT_GAP_MERGE,
    raw_L_min: int = DEFAULT_RAW_L_MIN,
    body_forward: RestoredBodyModelForward | None = None,
    frame_chunk: int = 1,
    target_chunk: int = 2048,
) -> dict[str, Any]:
    actor_motion, reactor_motion, _ = restore_pair_if_needed(actor_motion, reactor_motion, batch)
    lengths = lengths.to(device=actor_motion.device, dtype=torch.long)
    actor_vertices, reactor_vertices = motions_to_vertices(
        actor_motion,
        reactor_motion,
        lengths,
        batch,
        body_forward=body_forward,
    )

    batch_size = int(actor_motion.shape[0])
    num_frames = int(actor_motion.shape[-1])
    min_dist = actor_motion.new_full(
        (batch_size, len(HAND_SIDE_NAMES), len(TARGET_REGION_NAMES), num_frames),
        float("inf"),
    )
    for hand_side, hand_id in HAND_SIDE_IDS.items():
        hand_ids = region_map[f"{hand_side}_hand"]
        for region_name, region_id in TARGET_REGION_IDS.items():
            min_dist[:, hand_id, region_id] = region_to_region_min_distance(
                reactor_vertices,
                actor_vertices,
                hand_ids,
                region_map[region_name],
                frame_chunk=frame_chunk,
                target_chunk=target_chunk,
            )

    frame_ids = torch.arange(num_frames, device=lengths.device).view(1, 1, 1, -1)
    valid = frame_ids < lengths.view(-1, 1, 1, 1)
    contact_mask = (min_dist < float(tau_contact)) & valid
    min_dist = torch.where(valid, min_dist, torch.full_like(min_dist, float("inf")))

    lengths_np = _to_numpy(lengths).astype(np.int64)
    sample_indices = _batch_ints(batch, "sample_index", batch_size)
    dataset_row_indices = _batch_ints(batch, "dataset_row_index", batch_size)
    if "dataset_key" in batch:
        dataset_keys = _batch_strings(batch, "dataset_key", batch_size, "sample")
    else:
        dataset_keys = [f"sample_{sample_indices[idx]}" for idx in range(batch_size)]
    mask_np = _to_numpy(contact_mask).astype(np.uint8)
    dist_np = _to_numpy(min_dist).astype(np.float32)
    segments = segments_from_contact_mask(
        mask_np,
        lengths_np,
        sample_indices,
        dataset_row_indices,
        dataset_keys,
        gap_merge=gap_merge,
        raw_L_min=raw_L_min,
    )
    return {
        "contact_mask": mask_np,
        "min_region_dist": dist_np,
        "segments": segments,
        "lengths": lengths_np,
        "sample_indices": sample_indices,
        "dataset_row_indices": dataset_row_indices,
        "dataset_keys": dataset_keys,
    }


def build_contact_labels_for_loader(
    loader,
    region_map: dict[str, np.ndarray],
    *,
    tau_contact: float = DEFAULT_TAU_CONTACT,
    gap_merge: int = DEFAULT_GAP_MERGE,
    raw_L_min: int = DEFAULT_RAW_L_MIN,
    device: str = "cpu",
    frame_chunk: int = 1,
    target_chunk: int = 2048,
    show_progress: bool = True,
) -> dict[str, Any]:
    device_t = torch.device(device)
    body_forward = RestoredBodyModelForward(device=device_t)
    masks = []
    dists = []
    lengths_all = []
    sample_indices_all = []
    dataset_row_indices_all = []
    dataset_keys_all = []
    segments_all: list[dict[str, Any]] = []
    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else None
    progress = ProgressBar("build_contact_labels", total_samples, unit="samples", enabled=show_progress).start()

    for batch in loader:
        actor_motion = batch["actor_motion"].to(device_t)
        gt_motion = batch["gt_motion"].to(device_t)
        lengths = batch["lengths"].to(device_t)
        result = compute_contact_for_batch(
            actor_motion,
            gt_motion,
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
        masks.append(result["contact_mask"])
        dists.append(result["min_region_dist"])
        lengths_all.extend(int(x) for x in result["lengths"].tolist())
        sample_indices_all.extend(result["sample_indices"])
        dataset_row_indices_all.extend(result["dataset_row_indices"])
        dataset_keys_all.extend(result["dataset_keys"])
        segments_all.extend(result["segments"])
        progress.update(len(result["lengths"]))
    progress.finish()

    metadata = {
        "artifact": "contact_labels_gt",
        "space_definition": RESTORED_PAIR_SPACE,
        "tau_contact": float(tau_contact),
        "gap_merge": int(gap_merge),
        "raw_L_min": int(raw_L_min),
        "hand_side_names": HAND_SIDE_NAMES,
        "target_region_names": TARGET_REGION_NAMES,
        "region_map_summary": region_map_summary(region_map),
        "segment_interval_semantics": "[raw_start_frame, raw_end_frame)",
    }
    return {
        "gt_contact_mask": np.concatenate(masks, axis=0) if masks else np.zeros((0, 2, 6, 0), dtype=np.uint8),
        "gt_min_region_dist": np.concatenate(dists, axis=0) if dists else np.zeros((0, 2, 6, 0), dtype=np.float32),
        "lengths": np.asarray(lengths_all, dtype=np.int64),
        "sample_indices": np.asarray(sample_indices_all, dtype=np.int64),
        "dataset_row_indices": np.asarray(dataset_row_indices_all, dtype=np.int64),
        "dataset_key": np.asarray(dataset_keys_all, dtype=object),
        "segments": records_to_object_array(segments_all),
        "space_definition": np.asarray(RESTORED_PAIR_SPACE),
        "hand_side_names": np.asarray(HAND_SIDE_NAMES, dtype=object),
        "target_region_names": np.asarray(TARGET_REGION_NAMES, dtype=object),
        "metadata_json": np.asarray(dumps_metadata(metadata)),
    }


def save_contact_labels(path: str, artifact: dict[str, Any]):
    np.savez_compressed(path, **artifact)
