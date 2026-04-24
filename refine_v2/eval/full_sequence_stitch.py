"""Full-sequence stitching for refine_v2 window refiner outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from refine_v2.refiner_data.window_dataset import RefineV2WindowDataset
from refine_v2.refiner_data.window_loader import collate_refine_v2_window_batch
from refine_v2.subset.reporting import read_json
from refine_v2.train.eval_window import batch_to_device


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _matches_selected_action(record: dict[str, Any], selected: set[str]) -> bool:
    if not selected:
        return True
    candidates = {
        str(record.get("action_type", "")),
        str(record.get("action_name", "")),
        str(record.get("action_label", "")),
    }
    candidates = {x.lower() for x in candidates if x}
    return bool(candidates & selected)


def select_subset_sequence_records(
    subset_manifest_path: str,
    *,
    include_buckets: list[str] | None = None,
    selected_action_types: list[str] | None = None,
    max_sequences_per_action_type: int = 100,
    sample_seed: int = 1234,
) -> list[dict[str, Any]]:
    import random

    manifest = read_json(subset_manifest_path)
    include_set = set(include_buckets or ["GT+ / Pred+"])
    selected_actions = {str(x).lower() for x in (selected_action_types or [])}
    records = [
        dict(item)
        for item in manifest.get("sequences", [])
        if str(item.get("bucket_label", "")) in include_set
        and _matches_selected_action(item, selected_actions)
    ]
    if int(max_sequences_per_action_type) <= 0:
        return sorted(records, key=lambda item: int(item["dataset_row_index"]))

    rng = random.Random(int(sample_seed))
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        action = str(item.get("action_type", item.get("action_name", "")))
        by_action[action].append(item)
    selected: list[dict[str, Any]] = []
    for action in sorted(by_action):
        bucket = list(by_action[action])
        rng.shuffle(bucket)
        limit = min(int(max_sequences_per_action_type), len(bucket))
        selected.extend(bucket[:limit])
    return sorted(selected, key=lambda item: (str(item.get("action_type", "")), int(item["dataset_row_index"])))


def build_center_weight(length: int) -> np.ndarray:
    length = int(length)
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    if length == 1:
        return np.ones((1,), dtype=np.float32)
    center = 0.5 * float(length - 1)
    radius = center + 1.0
    pos = np.arange(length, dtype=np.float32)
    weight = (radius - np.abs(pos - center)) / max(radius, 1e-6)
    return np.clip(weight.astype(np.float32), 1e-4, None)


def _slice_reaction_value(arr: np.ndarray, rows: np.ndarray):
    arr = np.asarray(arr)
    if arr.shape == ():
        return np.repeat(arr.reshape(1), rows.shape[0], axis=0)
    if arr.shape[0] == 1 and rows.shape[0] > 1:
        return np.repeat(arr, rows.shape[0], axis=0)
    return arr[rows]


@torch.no_grad()
def stitch_refiner_full_sequences(
    *,
    checkpoint_path: str,
    reaction_data_path: str,
    contact_labels_path: str,
    subset_manifest_path: str,
    selector_windows_path: str,
    include_buckets: list[str],
    geometry_feature_cache_path: str = "",
    selected_action_types: list[str] | None = None,
    max_sequences_per_action_type: int = 100,
    sample_seed: int = 1234,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cuda",
) -> dict[str, Any]:
    from refine_v2.model.refiner_v2 import RefineV2WindowRefiner, RefineV2WindowRefinerConfig

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    state = torch.load(checkpoint_path, map_location=dev)
    train_cfg = state.get("config", {})
    resolved_geometry_cache = geometry_feature_cache_path or str(train_cfg.get("geometry_feature_cache_path", ""))
    if bool(state.get("model_config", {}).get("use_geometry_features", False)) and not resolved_geometry_cache:
        raise ValueError("Checkpoint uses geometry features; pass geometry_feature_cache_path.")

    sequence_records = select_subset_sequence_records(
        subset_manifest_path,
        include_buckets=include_buckets,
        selected_action_types=selected_action_types,
        max_sequences_per_action_type=max_sequences_per_action_type,
        sample_seed=sample_seed,
    )
    if not sequence_records:
        raise ValueError("No subset sequences remain after bucket/action sampling.")

    dataset = RefineV2WindowDataset(
        reaction_data_path,
        contact_labels_path,
        subset_manifest_path,
        selector_windows_path,
        include_buckets=include_buckets,
        geometry_feature_cache_path=resolved_geometry_cache,
        selected_action_types=selected_action_types,
        strict_checks=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=dev.type == "cuda",
        collate_fn=collate_refine_v2_window_batch,
    )
    model = RefineV2WindowRefiner(RefineV2WindowRefinerConfig(**state["model_config"])).to(dev)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    rows = np.asarray([int(item["dataset_row_index"]) for item in sequence_records], dtype=np.int64)
    row_to_seq = {int(row): idx for idx, row in enumerate(rows.tolist())}
    reaction = dataset.reaction
    actor_motion = np.asarray(reaction["actor_motion"])[rows].astype(np.float32, copy=True)
    coarse_motion = np.asarray(reaction["reactor_coarse"])[rows].astype(np.float32, copy=True)
    gt_motion = np.asarray(reaction["reactor_gt"])[rows].astype(np.float32, copy=True)
    lengths = np.asarray(reaction["lengths"])[rows].astype(np.int64, copy=True)
    sample_indices = np.asarray(reaction["sample_indices"])[rows].astype(np.int64, copy=True)
    max_frames = int(actor_motion.shape[-1])

    delta_sum = np.zeros_like(coarse_motion, dtype=np.float32)
    weight_sum = np.zeros((rows.shape[0], max_frames), dtype=np.float32)
    coverage_count = np.zeros((rows.shape[0], max_frames), dtype=np.int32)
    sequence_window_counts = np.zeros((rows.shape[0],), dtype=np.int32)

    for batch in loader:
        batch = batch_to_device(batch, dev)
        outputs = model(batch)
        pred_delta = outputs["pred_delta_motion_window"].detach().cpu().numpy().astype(np.float32)
        valid_mask = batch["valid_mask"].detach().cpu().numpy().astype(bool)
        start_frames = batch["start_frame"].detach().cpu().numpy().astype(np.int64)
        end_frames = batch["end_frame"].detach().cpu().numpy().astype(np.int64)
        batch_rows = batch["dataset_row_index"].detach().cpu().numpy().astype(np.int64)
        for i in range(pred_delta.shape[0]):
            row = int(batch_rows[i])
            seq_idx = row_to_seq.get(row)
            if seq_idx is None:
                continue
            start = int(start_frames[i])
            end = int(end_frames[i])
            local_valid = np.flatnonzero(valid_mask[i])
            if local_valid.size == 0 or end <= start:
                continue
            local_weights = build_center_weight(pred_delta.shape[-1])[local_valid]
            global_frames = start + local_valid
            keep = (global_frames >= 0) & (global_frames < max_frames)
            if not np.any(keep):
                continue
            global_frames = global_frames[keep]
            local_valid = local_valid[keep]
            local_weights = local_weights[keep]
            # Use np.take on the temporal axis to avoid numpy advanced-indexing
            # reordering the dimensions into [T, J, C].
            window_delta = np.take(pred_delta[i], local_valid, axis=-1)
            delta_sum[seq_idx, :, :, global_frames] += (
                window_delta * local_weights.reshape(1, 1, -1)
            )
            weight_sum[seq_idx, global_frames] += local_weights
            coverage_count[seq_idx, global_frames] += 1
            sequence_window_counts[seq_idx] += 1

    merged_delta = np.zeros_like(delta_sum, dtype=np.float32)
    covered = weight_sum > 0
    for seq_idx in range(rows.shape[0]):
        mask = covered[seq_idx]
        if not np.any(mask):
            continue
        merged_delta[seq_idx, :, :, mask] = (
            delta_sum[seq_idx, :, :, mask] / weight_sum[seq_idx, mask].reshape(1, 1, -1)
        )
    refined_motion = coarse_motion + merged_delta

    pack: dict[str, Any] = {
        "actor_motion": actor_motion,
        "reactor_gt": gt_motion,
        "reactor_coarse": coarse_motion,
        "reactor_refined": refined_motion.astype(np.float32, copy=False),
        "reactor_refined_delta": merged_delta.astype(np.float32, copy=False),
        "lengths": lengths,
        "sample_indices": sample_indices,
        "dataset_row_indices": rows,
        "dataset_key": np.asarray([_as_str(item.get("dataset_key", "")) for item in sequence_records], dtype=object),
        "action_type": np.asarray([str(item.get("action_type", item.get("action_name", ""))) for item in sequence_records], dtype=object),
        "bucket_label": np.asarray([str(item.get("bucket_label", "")) for item in sequence_records], dtype=object),
        "space_definition": _slice_reaction_value(reaction["space_definition"], rows),
    }
    optional_fields = (
        "loader_base_trans",
        "pair_base_trans",
        "ground_offset_y_actor",
        "ground_offset_y_reactor",
        "actor_betas",
        "reactor_betas",
        "actor_gender_id",
        "reactor_gender_id",
        "body_model_type",
    )
    for key in optional_fields:
        if key in reaction.files:
            pack[key] = _slice_reaction_value(reaction[key], rows)

    sequence_stats = []
    for seq_idx, item in enumerate(sequence_records):
        valid_len = int(lengths[seq_idx])
        covered_frames = int(np.count_nonzero(covered[seq_idx, :valid_len]))
        overlap_frames = int(np.count_nonzero(coverage_count[seq_idx, :valid_len] > 1))
        sequence_stats.append(
            {
                "dataset_row_index": int(rows[seq_idx]),
                "sample_index": int(sample_indices[seq_idx]),
                "dataset_key": _as_str(item.get("dataset_key", "")),
                "action_type": str(item.get("action_type", item.get("action_name", ""))),
                "bucket_label": str(item.get("bucket_label", "")),
                "length": valid_len,
                "num_windows": int(sequence_window_counts[seq_idx]),
                "covered_frames": covered_frames,
                "covered_frame_ratio": float(covered_frames / max(valid_len, 1)),
                "overlap_frames": overlap_frames,
                "overlap_frame_ratio": float(overlap_frames / max(valid_len, 1)),
            }
        )

    summary = {
        "num_sequences": int(rows.shape[0]),
        "num_sequences_with_windows": int(np.count_nonzero(sequence_window_counts > 0)),
        "mean_windows_per_sequence": float(sequence_window_counts.mean()) if rows.size else 0.0,
        "mean_covered_frame_ratio": float(np.mean([x["covered_frame_ratio"] for x in sequence_stats])) if sequence_stats else 0.0,
        "mean_overlap_frame_ratio": float(np.mean([x["overlap_frame_ratio"] for x in sequence_stats])) if sequence_stats else 0.0,
        "max_sequences_per_action_type": int(max_sequences_per_action_type),
        "sample_seed": int(sample_seed),
    }
    return {
        "pack": pack,
        "sequence_records": sequence_records,
        "sequence_stats": sequence_stats,
        "summary": summary,
        "resolved_geometry_feature_cache_path": resolved_geometry_cache,
    }
