"""Full-sequence Stage1-only vs Stage1+Stage2 evaluation for refine_v2."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from refine.eval.global_motion import evaluate_global_motion
from refine_v2.data.contact_labels import compute_contact_for_batch
from refine_v2.data.schema import to_jsonable
from refine_v2.data.restored_space import RestoredBodyModelForward
from refine_v2.model.regions import load_region_map, region_map_summary
from refine_v2.eval.full_sequence_stitch import stitch_refiner_full_sequences


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0 else 0.0


def _binary_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool) & valid.astype(bool)
    gt = gt.astype(bool) & valid.astype(bool)
    tp = float(np.count_nonzero(pred & gt))
    fp = float(np.count_nonzero(pred & ~gt))
    fn = float(np.count_nonzero(~pred & gt))
    tn = float(np.count_nonzero(~pred & ~gt & valid.astype(bool)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _segments_1d(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask).astype(bool).reshape(-1)
    out = []
    idx = 0
    while idx < mask.size:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        idx += 1
        while idx < mask.size and mask[idx]:
            idx += 1
        out.append((start, idx))
    return out


def _duration_frequency(mask: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    ratios = []
    durations = []
    freqs = []
    transitions = []
    for seq_mask, seq_valid in zip(mask.astype(bool), valid_mask.astype(bool)):
        valid_len = int(np.count_nonzero(seq_valid))
        if valid_len <= 0:
            continue
        m = seq_mask[:valid_len]
        segs = _segments_1d(m)
        ratios.append(float(m.mean()))
        durations.extend(float(e - s) for s, e in segs)
        freqs.append(float(len(segs)))
        transitions.append(float(np.count_nonzero(m[1:] != m[:-1])) / float(max(1, valid_len - 1)))
    return {
        "contact_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "avg_contact_duration": float(np.mean(durations)) if durations else 0.0,
        "contact_frequency": float(np.mean(freqs)) if freqs else 0.0,
        "contact_jitter": float(np.mean(transitions)) if transitions else 0.0,
        "num_contact_segments": int(len(durations)),
    }


def _contact_metrics(
    *,
    coarse_dist: np.ndarray,
    refined_dist: np.ndarray,
    gt_dist: np.ndarray,
    gt_mask: np.ndarray,
    lengths: np.ndarray,
    tau_contact: float,
    penetration_margin: float,
) -> dict[str, float]:
    num_frames = int(gt_dist.shape[-1])
    valid = np.arange(num_frames, dtype=np.int64).reshape(1, 1, 1, -1) < lengths.reshape(-1, 1, 1, 1)
    gt_contact = gt_mask.astype(bool) & valid
    coarse_contact = (coarse_dist < float(tau_contact)) & valid
    refined_contact = (refined_dist < float(tau_contact)) & valid

    def dist_block(scope_name: str, scope: np.ndarray) -> dict[str, float]:
        count = float(np.count_nonzero(scope))
        if count <= 0:
            return {f"{scope_name}_count": 0.0}
        coarse_l1 = np.abs(coarse_dist - gt_dist)
        refined_l1 = np.abs(refined_dist - gt_dist)
        return {
            f"{scope_name}_count": count,
            f"{scope_name}_coarse_dist_l1": float((coarse_l1 * scope).sum() / count),
            f"{scope_name}_refined_dist_l1": float((refined_l1 * scope).sum() / count),
            f"{scope_name}_dist_l1_improvement": float(((coarse_l1 - refined_l1) * scope).sum() / count),
            f"{scope_name}_coarse_min_dist": float((coarse_dist * scope).sum() / count),
            f"{scope_name}_refined_min_dist": float((refined_dist * scope).sum() / count),
            f"{scope_name}_gt_min_dist": float((gt_dist * scope).sum() / count),
            f"{scope_name}_contact_dist_improvement": float(((coarse_dist - refined_dist) * scope).sum() / count),
        }

    metrics: dict[str, float] = {}
    metrics.update(dist_block("all_valid", valid.astype(np.float32)))
    metrics.update(dist_block("gt_contact", gt_contact.astype(np.float32)))
    for prefix, mask in (("coarse", coarse_contact), ("refined", refined_contact)):
        bm = _binary_metrics(mask, gt_contact, valid)
        metrics.update({f"{prefix}_contact_{k}": v for k, v in bm.items()})
    for prefix, mask in (("gt", gt_contact), ("coarse", coarse_contact), ("refined", refined_contact)):
        union = mask.any(axis=(1, 2))
        dur = _duration_frequency(union, lengths.reshape(-1, 1) > np.arange(num_frames).reshape(1, -1))
        metrics.update({f"{prefix}_{k}": v for k, v in dur.items()})
    metrics["contact_ratio_error_improvement"] = abs(metrics["coarse_contact_ratio"] - metrics["gt_contact_ratio"]) - abs(
        metrics["refined_contact_ratio"] - metrics["gt_contact_ratio"]
    )
    metrics["contact_frequency_error_improvement"] = abs(
        metrics["coarse_contact_frequency"] - metrics["gt_contact_frequency"]
    ) - abs(metrics["refined_contact_frequency"] - metrics["gt_contact_frequency"])
    metrics["contact_duration_error_improvement"] = abs(
        metrics["coarse_avg_contact_duration"] - metrics["gt_avg_contact_duration"]
    ) - abs(metrics["refined_avg_contact_duration"] - metrics["gt_avg_contact_duration"])
    metrics["contact_jitter_error_improvement"] = abs(
        metrics["coarse_contact_jitter"] - metrics["gt_contact_jitter"]
    ) - abs(metrics["refined_contact_jitter"] - metrics["gt_contact_jitter"])

    for prefix, dist in (("gt", gt_dist), ("coarse", coarse_dist), ("refined", refined_dist)):
        too_close = np.maximum(float(penetration_margin) - dist, 0.0)
        valid_count = float(np.count_nonzero(valid))
        metrics[f"{prefix}_surrogate_penetration_rate"] = float(np.count_nonzero((too_close > 0) & valid) / max(valid_count, 1.0))
        metrics[f"{prefix}_surrogate_penetration_depth"] = float((too_close * valid.astype(np.float32)).sum() / max(valid_count, 1.0))
    metrics["surrogate_penetration_rate_improvement"] = (
        metrics["coarse_surrogate_penetration_rate"] - metrics["refined_surrogate_penetration_rate"]
    )
    metrics["surrogate_penetration_depth_improvement"] = (
        metrics["coarse_surrogate_penetration_depth"] - metrics["refined_surrogate_penetration_depth"]
    )
    metrics["coarse_vs_gt_surrogate_penetration_rate_gap"] = abs(
        metrics["coarse_surrogate_penetration_rate"] - metrics["gt_surrogate_penetration_rate"]
    )
    metrics["refined_vs_gt_surrogate_penetration_rate_gap"] = abs(
        metrics["refined_surrogate_penetration_rate"] - metrics["gt_surrogate_penetration_rate"]
    )
    metrics["surrogate_penetration_rate_gap_improvement"] = (
        metrics["coarse_vs_gt_surrogate_penetration_rate_gap"]
        - metrics["refined_vs_gt_surrogate_penetration_rate_gap"]
    )
    metrics["coarse_vs_gt_surrogate_penetration_depth_gap"] = abs(
        metrics["coarse_surrogate_penetration_depth"] - metrics["gt_surrogate_penetration_depth"]
    )
    metrics["refined_vs_gt_surrogate_penetration_depth_gap"] = abs(
        metrics["refined_surrogate_penetration_depth"] - metrics["gt_surrogate_penetration_depth"]
    )
    metrics["surrogate_penetration_depth_gap_improvement"] = (
        metrics["coarse_vs_gt_surrogate_penetration_depth_gap"]
        - metrics["refined_vs_gt_surrogate_penetration_depth_gap"]
    )
    metrics["refined_penetration_depth_excess_over_gt"] = max(
        0.0,
        metrics["refined_surrogate_penetration_depth"] - metrics["gt_surrogate_penetration_depth"],
    )
    return metrics


def _accumulate_by_action(
    action_types: list[str],
    per_sequence: list[dict[str, Any]],
    metric_keys: list[str],
) -> dict[str, dict[str, float]]:
    bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for action, item in zip(action_types, per_sequence):
        for key in metric_keys:
            if key in item:
                bucket[action][key].append(float(item[key]))
    out: dict[str, dict[str, float]] = {}
    for action in sorted(bucket):
        out[action] = {key: float(np.mean(values)) for key, values in sorted(bucket[action].items()) if values}
    return out


@torch.no_grad()
def evaluate_full_sequence(
    *,
    checkpoint_path: str,
    reaction_data_path: str,
    contact_labels_path: str,
    subset_manifest_path: str,
    selector_windows_path: str,
    region_map_path: str,
    stgcn_model_path: str,
    include_buckets: list[str],
    geometry_feature_cache_path: str = "",
    selected_action_types: list[str] | None = None,
    max_sequences_per_action_type: int = 100,
    sample_seed: int = 1234,
    batch_size: int = 32,
    stgcn_batch_size: int = 64,
    num_workers: int = 0,
    device: str = "cuda",
    tau_contact: float = 0.05,
    penetration_margin: float = 0.015,
    frame_chunk: int = 1,
    target_chunk: int = 2048,
    dataset: str = "interx",
    body_model: str = "smplx",
    num_classes: int = 0,
) -> dict[str, Any]:
    stitched = stitch_refiner_full_sequences(
        checkpoint_path=checkpoint_path,
        reaction_data_path=reaction_data_path,
        contact_labels_path=contact_labels_path,
        subset_manifest_path=subset_manifest_path,
        selector_windows_path=selector_windows_path,
        include_buckets=include_buckets,
        geometry_feature_cache_path=geometry_feature_cache_path,
        selected_action_types=selected_action_types,
        max_sequences_per_action_type=max_sequences_per_action_type,
        sample_seed=sample_seed,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    pack = stitched["pack"]
    region_map = load_region_map(region_map_path or None)
    body_forward = RestoredBodyModelForward(device=device)

    labels = np.load(contact_labels_path, allow_pickle=True)
    label_rows = np.asarray(labels["dataset_row_indices"], dtype=np.int64)
    label_row_to_idx = {int(row): idx for idx, row in enumerate(label_rows.tolist())}

    action_types = [str(x) for x in np.asarray(pack["action_type"]).reshape(-1).tolist()]
    rows = np.asarray(pack["dataset_row_indices"], dtype=np.int64)
    lengths = np.asarray(pack["lengths"], dtype=np.int64)
    coarse_dist_parts = []
    refined_dist_parts = []
    gt_dist_parts = []
    gt_mask_parts = []

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    seq_count = int(rows.shape[0])
    for start in range(0, seq_count, max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), seq_count)
        batch = {
            "actor_motion": torch.as_tensor(pack["actor_motion"][start:end], device=dev),
            "space_definition": pack["space_definition"][start:end],
            "actor_betas": pack["actor_betas"][start:end],
            "reactor_betas": pack["reactor_betas"][start:end],
            "actor_gender_id": pack["actor_gender_id"][start:end],
            "reactor_gender_id": pack["reactor_gender_id"][start:end],
            "body_model_type": pack["body_model_type"][start:end],
        }
        for key in ("loader_base_trans", "pair_base_trans", "ground_offset_y_actor", "ground_offset_y_reactor"):
            if key in pack:
                batch[key] = pack[key][start:end]
        lengths_batch = torch.as_tensor(lengths[start:end], device=dev)
        coarse_payload = compute_contact_for_batch(
            batch["actor_motion"],
            torch.as_tensor(pack["reactor_coarse"][start:end], device=dev),
            lengths_batch,
            batch,
            region_map,
            tau_contact=tau_contact,
            body_forward=body_forward,
            frame_chunk=frame_chunk,
            target_chunk=target_chunk,
        )
        refined_payload = compute_contact_for_batch(
            batch["actor_motion"],
            torch.as_tensor(pack["reactor_refined"][start:end], device=dev),
            lengths_batch,
            batch,
            region_map,
            tau_contact=tau_contact,
            body_forward=body_forward,
            frame_chunk=frame_chunk,
            target_chunk=target_chunk,
        )
        coarse_dist_parts.append(np.asarray(coarse_payload["min_dist"], dtype=np.float32))
        refined_dist_parts.append(np.asarray(refined_payload["min_dist"], dtype=np.float32))
        gt_idx = [label_row_to_idx[int(row)] for row in rows[start:end].tolist()]
        gt_dist_parts.append(np.asarray(labels["gt_min_region_dist"])[gt_idx].astype(np.float32))
        gt_mask_parts.append(np.asarray(labels["gt_contact_mask"])[gt_idx].astype(np.uint8))

    coarse_dist = np.concatenate(coarse_dist_parts, axis=0) if coarse_dist_parts else np.zeros((0, 2, 6, 0), dtype=np.float32)
    refined_dist = np.concatenate(refined_dist_parts, axis=0) if refined_dist_parts else np.zeros((0, 2, 6, 0), dtype=np.float32)
    gt_dist = np.concatenate(gt_dist_parts, axis=0) if gt_dist_parts else np.zeros((0, 2, 6, 0), dtype=np.float32)
    gt_mask = np.concatenate(gt_mask_parts, axis=0) if gt_mask_parts else np.zeros((0, 2, 6, 0), dtype=np.uint8)
    contact_metrics = _contact_metrics(
        coarse_dist=coarse_dist,
        refined_dist=refined_dist,
        gt_dist=gt_dist,
        gt_mask=gt_mask,
        lengths=lengths,
        tau_contact=tau_contact,
        penetration_margin=penetration_margin,
    )

    per_sequence_contact: list[dict[str, Any]] = []
    for i in range(seq_count):
        metrics = _contact_metrics(
            coarse_dist=coarse_dist[i : i + 1],
            refined_dist=refined_dist[i : i + 1],
            gt_dist=gt_dist[i : i + 1],
            gt_mask=gt_mask[i : i + 1],
            lengths=lengths[i : i + 1],
            tau_contact=tau_contact,
            penetration_margin=penetration_margin,
        )
        metrics["dataset_row_index"] = int(rows[i])
        metrics["sample_index"] = int(np.asarray(pack["sample_indices"])[i])
        metrics["dataset_key"] = str(np.asarray(pack["dataset_key"], dtype=object)[i])
        metrics["action_type"] = action_types[i]
        per_sequence_contact.append(metrics)

    stgcn_payload = evaluate_global_motion(
        pack,
        dataset=dataset,
        stgcn_model_path=stgcn_model_path,
        body_model=body_model,
        batch_size=stgcn_batch_size,
        device=device,
        num_classes=num_classes if int(num_classes) > 0 else None,
    )

    action_breakdown = {
        "contact": _accumulate_by_action(
            action_types,
            per_sequence_contact,
            [
                "gt_contact_contact_dist_improvement",
                "refined_contact_f1",
                "coarse_contact_f1",
                "surrogate_penetration_depth_gap_improvement",
                "contact_ratio_error_improvement",
            ],
        )
    }
    return to_jsonable(
        {
            "artifact": "refine_v2_full_sequence_eval",
            "checkpoint_path": checkpoint_path,
            "paths": {
                "reaction_data_path": reaction_data_path,
                "contact_labels_path": contact_labels_path,
                "subset_manifest_path": subset_manifest_path,
                "selector_windows_path": selector_windows_path,
                "geometry_feature_cache_path": stitched["resolved_geometry_feature_cache_path"],
                "region_map_path": region_map_path,
                "stgcn_model_path": stgcn_model_path,
            },
            "params": {
                "include_buckets": list(include_buckets),
                "selected_action_types": list(selected_action_types or []),
                "max_sequences_per_action_type": int(max_sequences_per_action_type),
                "sample_seed": int(sample_seed),
                "tau_contact": float(tau_contact),
                "penetration_margin": float(penetration_margin),
                "batch_size": int(batch_size),
                "stgcn_batch_size": int(stgcn_batch_size),
                "frame_chunk": int(frame_chunk),
                "target_chunk": int(target_chunk),
                "dataset": str(dataset),
                "body_model": str(body_model),
            },
            "counts": {
                "num_sequences": int(seq_count),
                "num_action_types": int(len(sorted(set(action_types)))),
            },
            "stitch_summary": stitched["summary"],
            "region_map_summary": region_map_summary(region_map),
            "stgcn_metrics": stgcn_payload,
            "contact_metrics": contact_metrics,
            "action_breakdown": action_breakdown,
            "sequence_stats": stitched["sequence_stats"],
            "per_sequence_contact": per_sequence_contact,
            "notes": [
                "This is the formal Stage2 full-sequence evaluation.",
                "Full-sequence refined motion is built by center-weighted residual stitching over overlapping windows.",
                "STGCN metrics are computed in canonical/Stage1-aligned processed space after inverse restore.",
                "Contact metrics are computed in restored pair space with restored shape.",
                "Stage2 is interpreted as a contact-refine module, not a motion-reconstruction module.",
            ],
            "pack": pack,
        }
    )
