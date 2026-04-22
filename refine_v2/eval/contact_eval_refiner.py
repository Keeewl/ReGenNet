"""Window-level contact evaluation for refine_v2 refiner checkpoints."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from refine_v2.data.contact_labels import region_to_region_min_distance
from refine_v2.data.schema import HAND_SIDE_NAMES, TARGET_REGION_NAMES, TARGET_REGION_IDS, to_jsonable
from refine_v2.data.restored_space import RestoredBodyModelForward, lengths_to_mask
from refine_v2.model.regions import load_region_map, region_map_summary
from refine_v2.model.refiner_v2 import RefineV2WindowRefiner, RefineV2WindowRefinerConfig
from refine_v2.refiner_data.window_dataset import RefineV2WindowDataset
from refine_v2.refiner_data.window_loader import collate_refine_v2_window_batch
from refine_v2.train.eval_window import batch_to_device


@dataclass
class ContactEvalConfig:
    tau_contact: float = 0.05
    penetration_margin: float = 0.015
    frame_chunk: int = 1
    target_chunk: int = 2048
    max_debug_windows: int = 500


def _as_tensor(value, *, device, dtype=None):
    out = value if torch.is_tensor(value) else torch.as_tensor(value)
    out = out.to(device=device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _metadata_for_rows(dataset: RefineV2WindowDataset, rows: torch.Tensor, *, device, dtype) -> dict[str, Any]:
    rows_np = rows.detach().cpu().numpy().astype(np.int64).reshape(-1)
    reaction = dataset.reaction
    required = ("actor_betas", "reactor_betas", "actor_gender_id", "reactor_gender_id", "body_model_type")
    missing = [key for key in required if key not in reaction.files]
    if missing:
        raise KeyError(
            "eval_contact_refiner requires body metadata in reaction_data: " + ", ".join(missing)
        )
    max_row = int(rows_np.max()) if rows_np.size else 0

    def take_rows(key: str):
        arr = np.asarray(reaction[key])
        if arr.shape == ():
            return np.repeat(arr.reshape(1), rows_np.shape[0], axis=0)
        if arr.shape[0] == 1 and max_row >= 1:
            return np.repeat(arr, rows_np.shape[0], axis=0)
        return arr[rows_np]

    body_model_type_values = take_rows("body_model_type")
    first = body_model_type_values.reshape(-1)[0]
    if isinstance(first, bytes):
        first = first.decode("utf-8")
    body_model_type = str(first).lower()
    if body_model_type != "smplx":
        raise ValueError(f"eval_contact_refiner currently supports body_model_type=smplx, got {body_model_type}")
    return {
        "actor_betas": _as_tensor(take_rows("actor_betas"), device=device, dtype=dtype),
        "reactor_betas": _as_tensor(take_rows("reactor_betas"), device=device, dtype=dtype),
        "actor_gender_id": _as_tensor(take_rows("actor_gender_id"), device=device, dtype=torch.long).view(-1),
        "reactor_gender_id": _as_tensor(take_rows("reactor_gender_id"), device=device, dtype=torch.long).view(-1),
        "body_model_type": body_model_type,
    }


@torch.no_grad()
def _motion_to_vertices(
    body_forward: RestoredBodyModelForward,
    motion: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    betas: torch.Tensor,
    gender_id: torch.Tensor,
    body_model_type: str,
) -> torch.Tensor:
    return body_forward.motion_to_xyz(
        motion,
        jointstype="vertices",
        betas=betas,
        gender_id=gender_id,
        mask=valid_mask.bool(),
        body_model_type=body_model_type,
    )


@torch.no_grad()
def compute_selected_hand_region_dist(
    actor_vertices: torch.Tensor,
    reactor_vertices: torch.Tensor,
    hand_side_id: torch.Tensor,
    region_map: dict[str, np.ndarray],
    *,
    frame_chunk: int = 1,
    target_chunk: int = 2048,
) -> torch.Tensor:
    batch_size = int(actor_vertices.shape[0])
    num_frames = int(actor_vertices.shape[-1])
    out = actor_vertices.new_full((batch_size, len(TARGET_REGION_NAMES), num_frames), float("inf"))
    for hand_id, hand_side in enumerate(HAND_SIDE_NAMES):
        idx = torch.nonzero(hand_side_id.long() == hand_id, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        actor_sub = actor_vertices.index_select(0, idx)
        reactor_sub = reactor_vertices.index_select(0, idx)
        hand_ids = region_map[f"{hand_side}_hand"]
        for region_name, region_id in TARGET_REGION_IDS.items():
            dist = region_to_region_min_distance(
                reactor_sub,
                actor_sub,
                hand_ids,
                region_map[region_name],
                frame_chunk=frame_chunk,
                target_chunk=target_chunk,
            )
            out_sub = out.index_select(0, idx)
            out_sub[:, region_id, :] = dist
            out.index_copy_(0, idx, out_sub)
    return out


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0 else 0.0


def _binary_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    pred = pred.bool() & valid.bool()
    gt = gt.bool() & valid.bool()
    tp = float((pred & gt).sum().item())
    fp = float((pred & ~gt).sum().item())
    fn = float((~pred & gt).sum().item())
    tn = float((~pred & ~gt & valid.bool()).sum().item())
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


def _duration_frequency(mask: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, float]:
    mask_np = (mask.bool() & valid_mask.bool()).detach().cpu().numpy()
    valid_np = valid_mask.bool().detach().cpu().numpy()
    ratios = []
    durations = []
    freqs = []
    transitions = []
    for seq_mask, seq_valid in zip(mask_np, valid_np):
        valid_len = int(seq_valid.sum())
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


class _Accumulator:
    def __init__(self):
        self.sum = defaultdict(float)
        self.count = defaultdict(float)
        self.windows: list[dict[str, Any]] = []
        self.groups: dict[str, dict[str, "_Accumulator"]] = defaultdict(dict)

    def add_scalar(self, key: str, value: float, count: float = 1.0):
        self.sum[key] += float(value) * float(count)
        self.count[key] += float(count)

    def add_metrics(self, metrics: dict[str, float], count: float = 1.0):
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.add_scalar(key, float(value), count=count)

    def add_group(self, group: str, name: str, metrics: dict[str, float], count: float = 1.0):
        if name not in self.groups[group]:
            self.groups[group][name] = _Accumulator()
        self.groups[group][name].add_metrics(metrics, count=count)

    def finalize(self) -> dict[str, float]:
        return {key: _safe_div(self.sum[key], self.count[key]) for key in sorted(self.sum)}

    def finalize_groups(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            group: {name: acc.finalize() for name, acc in sorted(values.items())}
            for group, values in sorted(self.groups.items())
        }


def _topk_region_mask(topk_ids: torch.Tensor, num_regions: int, num_frames: int) -> torch.Tensor:
    mask = torch.zeros(topk_ids.shape[0], num_regions, device=topk_ids.device, dtype=torch.bool)
    mask.scatter_(1, topk_ids.long().clamp(0, num_regions - 1), True)
    return mask[:, :, None].expand(-1, -1, num_frames)


def _window_metrics(
    *,
    coarse_dist: torch.Tensor,
    refined_dist: torch.Tensor,
    gt_dist: torch.Tensor,
    gt_mask: torch.Tensor,
    topk_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    tau_contact: float,
    penetration_margin: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    b, r, t = refined_dist.shape
    valid = valid_mask[:, None, :].expand(b, r, t)
    topk_valid = valid & _topk_region_mask(topk_ids, r, t)
    gt_contact = gt_mask.bool() & valid
    coarse_contact = (coarse_dist < tau_contact) & valid
    refined_contact = (refined_dist < tau_contact) & valid

    def dist_block(scope_name: str, scope: torch.Tensor) -> dict[str, float]:
        count = float(scope.float().sum().item())
        if count <= 0:
            return {f"{scope_name}_count": 0.0}
        coarse_l1 = torch.abs(coarse_dist - gt_dist)
        refined_l1 = torch.abs(refined_dist - gt_dist)
        coarse_mean = float((coarse_l1 * scope.float()).sum().item() / count)
        refined_mean = float((refined_l1 * scope.float()).sum().item() / count)
        coarse_dist_mean = float((coarse_dist * scope.float()).sum().item() / count)
        refined_dist_mean = float((refined_dist * scope.float()).sum().item() / count)
        gt_dist_mean = float((gt_dist * scope.float()).sum().item() / count)
        return {
            f"{scope_name}_count": count,
            f"{scope_name}_coarse_dist_l1": coarse_mean,
            f"{scope_name}_refined_dist_l1": refined_mean,
            f"{scope_name}_dist_l1_improvement": coarse_mean - refined_mean,
            f"{scope_name}_coarse_min_dist": coarse_dist_mean,
            f"{scope_name}_refined_min_dist": refined_dist_mean,
            f"{scope_name}_gt_min_dist": gt_dist_mean,
            f"{scope_name}_contact_dist_improvement": coarse_dist_mean - refined_dist_mean,
        }

    metrics = {}
    metrics.update(dist_block("all_valid", valid))
    metrics.update(dist_block("gt_contact", gt_contact))
    metrics.update(dist_block("topk_valid", topk_valid))
    metrics.update(dist_block("topk_gt_contact", gt_contact & topk_valid))

    for prefix, mask in (("coarse", coarse_contact), ("refined", refined_contact)):
        bm = _binary_metrics(mask, gt_contact, valid)
        metrics.update({f"{prefix}_contact_{k}": v for k, v in bm.items()})
        bm_topk = _binary_metrics(mask, gt_contact, topk_valid)
        metrics.update({f"topk_{prefix}_contact_{k}": v for k, v in bm_topk.items()})

    for prefix, mask in (("gt", gt_contact), ("coarse", coarse_contact), ("refined", refined_contact)):
        union = mask.any(dim=1)
        dur = _duration_frequency(union, valid_mask)
        metrics.update({f"{prefix}_{k}": v for k, v in dur.items()})
    metrics["contact_ratio_error_improvement"] = abs(metrics["coarse_contact_ratio"] - metrics["gt_contact_ratio"]) - abs(
        metrics["refined_contact_ratio"] - metrics["gt_contact_ratio"]
    )
    metrics["contact_frequency_error_improvement"] = abs(metrics["coarse_contact_frequency"] - metrics["gt_contact_frequency"]) - abs(
        metrics["refined_contact_frequency"] - metrics["gt_contact_frequency"]
    )
    metrics["contact_duration_error_improvement"] = abs(metrics["coarse_avg_contact_duration"] - metrics["gt_avg_contact_duration"]) - abs(
        metrics["refined_avg_contact_duration"] - metrics["gt_avg_contact_duration"]
    )
    metrics["contact_jitter_error_improvement"] = abs(metrics["coarse_contact_jitter"] - metrics["gt_contact_jitter"]) - abs(
        metrics["refined_contact_jitter"] - metrics["gt_contact_jitter"]
    )

    # This is an unsigned-distance surrogate, not a true signed penetration metric.
    for prefix, dist in (("coarse", coarse_dist), ("refined", refined_dist)):
        too_close = (penetration_margin - dist).clamp_min(0.0)
        valid_count = valid.float().sum().clamp_min(1.0)
        metrics[f"{prefix}_surrogate_penetration_rate"] = float(((too_close > 0) & valid).float().sum().item() / valid_count.item())
        metrics[f"{prefix}_surrogate_penetration_depth"] = float((too_close * valid.float()).sum().item() / valid_count.item())
    metrics["surrogate_penetration_rate_improvement"] = (
        metrics["coarse_surrogate_penetration_rate"] - metrics["refined_surrogate_penetration_rate"]
    )
    metrics["surrogate_penetration_depth_improvement"] = (
        metrics["coarse_surrogate_penetration_depth"] - metrics["refined_surrogate_penetration_depth"]
    )
    return metrics, {"coarse_contact": coarse_contact, "refined_contact": refined_contact, "gt_contact": gt_contact}, {
        "valid": valid,
        "topk_valid": topk_valid,
    }


@torch.no_grad()
def evaluate_contact_refiner(
    *,
    checkpoint_path: str,
    reaction_data_path: str,
    contact_labels_path: str,
    subset_manifest_path: str,
    selector_windows_path: str,
    region_map_path: str,
    include_buckets: list[str],
    selected_action_types: list[str] | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "cuda",
    tau_contact: float = 0.05,
    penetration_margin: float = 0.005,
    frame_chunk: int = 1,
    target_chunk: int = 2048,
    max_batches: int = 0,
    max_debug_windows: int = 500,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    dataset = RefineV2WindowDataset(
        reaction_data_path,
        contact_labels_path,
        subset_manifest_path,
        selector_windows_path,
        include_buckets=include_buckets,
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
    state = torch.load(checkpoint_path, map_location=dev)
    model = RefineV2WindowRefiner(RefineV2WindowRefinerConfig(**state["model_config"])).to(dev)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    body_forward = RestoredBodyModelForward(device=dev)
    region_map = load_region_map(region_map_path or None)
    acc = _Accumulator()
    processed_windows = 0
    processed_batches = 0

    for batch_idx, batch in enumerate(loader):
        batch = batch_to_device(batch, dev)
        outputs = model(batch)
        rows = batch["dataset_row_index"]
        meta = _metadata_for_rows(dataset, rows, device=dev, dtype=batch["actor_motion_window"].dtype)
        valid_mask = batch["valid_mask"].bool()
        actor_vertices = _motion_to_vertices(
            body_forward,
            batch["actor_motion_window"].float(),
            valid_mask,
            betas=meta["actor_betas"],
            gender_id=meta["actor_gender_id"],
            body_model_type=meta["body_model_type"],
        )
        refined_vertices = _motion_to_vertices(
            body_forward,
            outputs["pred_motion_window"].float(),
            valid_mask,
            betas=meta["reactor_betas"],
            gender_id=meta["reactor_gender_id"],
            body_model_type=meta["body_model_type"],
        )
        refined_dist = compute_selected_hand_region_dist(
            actor_vertices,
            refined_vertices,
            batch["hand_side_id"],
            region_map,
            frame_chunk=frame_chunk,
            target_chunk=target_chunk,
        )
        coarse_dist = batch["coarse_min_region_dist_window"].float()
        gt_dist = batch["gt_min_region_dist_window"].float()
        gt_mask = batch["gt_region_contact_mask_window"].float() > 0
        metrics, _, _ = _window_metrics(
            coarse_dist=coarse_dist,
            refined_dist=refined_dist,
            gt_dist=gt_dist,
            gt_mask=gt_mask,
            topk_ids=batch["topk_target_region_ids"],
            valid_mask=valid_mask,
            tau_contact=float(tau_contact),
            penetration_margin=float(penetration_margin),
        )
        bsz = int(refined_dist.shape[0])
        processed_windows += bsz
        processed_batches += 1
        acc.add_metrics(metrics, count=bsz)
        for i in range(bsz):
            per_metrics, _, _ = _window_metrics(
                coarse_dist=coarse_dist[i : i + 1],
                refined_dist=refined_dist[i : i + 1],
                gt_dist=gt_dist[i : i + 1],
                gt_mask=gt_mask[i : i + 1],
                topk_ids=batch["topk_target_region_ids"][i : i + 1],
                valid_mask=valid_mask[i : i + 1],
                tau_contact=float(tau_contact),
                penetration_margin=float(penetration_margin),
            )
            action_type = str(batch.get("action_type", [""])[i])
            hand_side = str(batch.get("hand_side", [""])[i])
            primary_region = str(batch.get("primary_target_region", [""])[i])
            acc.add_group("action_type", action_type, per_metrics)
            acc.add_group("hand_side", hand_side, per_metrics)
            acc.add_group("primary_target_region", primary_region, per_metrics)
            if len(acc.windows) < int(max_debug_windows):
                acc.windows.append(
                    {
                        "dataset_row_index": int(batch["dataset_row_index"][i].detach().cpu().item()),
                        "sample_index": int(batch["sample_index"][i].detach().cpu().item()),
                        "dataset_key": str(batch.get("dataset_key", [""])[i]),
                        "action_type": action_type,
                        "hand_side": hand_side,
                        "primary_target_region": primary_region,
                        "topk_target_regions": list(batch.get("topk_target_regions", [[]])[i]),
                        "start_frame": int(batch["start_frame"][i].detach().cpu().item()),
                        "end_frame": int(batch["end_frame"][i].detach().cpu().item()),
                        "metrics": {k: float(v) for k, v in per_metrics.items() if isinstance(v, (int, float))},
                    }
                )
        if max_batches and batch_idx + 1 >= int(max_batches):
            break

    payload = {
        "artifact": "eval_contact_refiner_window_level",
        "checkpoint_path": checkpoint_path,
        "paths": {
            "reaction_data_path": reaction_data_path,
            "contact_labels_path": contact_labels_path,
            "subset_manifest_path": subset_manifest_path,
            "selector_windows_path": selector_windows_path,
            "region_map_path": region_map_path,
        },
        "params": {
            "tau_contact": float(tau_contact),
            "penetration_margin": float(penetration_margin),
            "batch_size": int(batch_size),
            "frame_chunk": int(frame_chunk),
            "target_chunk": int(target_chunk),
            "include_buckets": list(include_buckets),
            "selected_action_types": list(selected_action_types or []),
        },
        "region_map_summary": region_map_summary(region_map),
        "counts": {
            "dataset_windows": int(len(dataset)),
            "processed_windows": int(processed_windows),
            "processed_batches": int(processed_batches),
            "max_batches": int(max_batches),
        },
        "metrics": acc.finalize(),
        "breakdown": acc.finalize_groups(),
        "windows_debug": acc.windows,
        "notes": [
            "Metrics are window-level and compare coarse vs refined vs direct GT binary mesh-region labels.",
            "surrogate_penetration_* uses unsigned min-distance below penetration_margin; it is not signed mesh penetration.",
            "Selector/window sampling parameters are intentionally not mixed into this contact-quality report.",
        ],
    }
    return to_jsonable(payload)
