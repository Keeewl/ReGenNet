"""Window-level evaluation for refine_v2 refiner."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def scalarize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if torch.is_tensor(value) and value.ndim == 0:
            out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _masked_l1_per_sample(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    err = torch.abs(pred - target)
    weights = valid_mask.float().view(valid_mask.shape[0], 1, 1, valid_mask.shape[1]).expand_as(err)
    return (err * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1e-8)


def _contact_l1_per_sample(pred: torch.Tensor, target: torch.Tensor, contact_frame: torch.Tensor) -> torch.Tensor:
    err = torch.abs(pred - target)
    weights = contact_frame.float().view(contact_frame.shape[0], 1, 1, contact_frame.shape[1]).expand_as(err)
    return (err * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1e-8)


def eval_batch_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pred = outputs["pred_motion_window"].float()
    coarse = batch["coarse_motion_window"].float()
    gt = batch["gt_motion_window"].float()
    valid = batch["valid_mask"].bool()
    contact_frame = batch["gt_region_contact_mask_window"].float().amax(dim=1).gt(0.0) & valid
    coarse_err = _masked_l1_per_sample(coarse, gt, valid)
    pred_err = _masked_l1_per_sample(pred, gt, valid)
    coarse_contact_err = _contact_l1_per_sample(coarse, gt, contact_frame)
    pred_contact_err = _contact_l1_per_sample(pred, gt, contact_frame)
    return {
        "coarse_motion_error": coarse_err,
        "pred_motion_error": pred_err,
        "motion_improvement": coarse_err - pred_err,
        "coarse_contact_motion_error": coarse_contact_err,
        "pred_contact_motion_error": pred_contact_err,
        "contact_motion_improvement": coarse_contact_err - pred_contact_err,
        "has_contact_frame": contact_frame.any(dim=1).float(),
    }


class _Averager:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value: float, n: int = 1):
        self.sum += float(value) * int(n)
        self.count += int(n)

    def mean(self) -> float:
        return self.sum / max(1, self.count)


def _add_breakdown(bucket: dict[str, dict[str, _Averager]], name: str, key: str, metrics: dict[str, float]):
    if key not in bucket[name]:
        bucket[name][key] = _Averager()
    for metric_name, metric_value in metrics.items():
        compound = f"{key}::{metric_name}"
        if compound not in bucket[name]:
            bucket[name][compound] = _Averager()
        bucket[name][compound].add(metric_value)


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    loss_fn,
    *,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    scalar_totals: dict[str, _Averager] = defaultdict(_Averager)
    breakdown: dict[str, dict[str, _Averager]] = {
        "action_type": {},
        "hand_side": {},
        "primary_target_region": {},
    }
    num_batches = 0
    num_samples = 0
    for batch in dataloader:
        batch = batch_to_device(batch, device)
        outputs = model(batch)
        loss_metrics = scalarize_metrics(loss_fn(outputs, batch))
        per_sample = eval_batch_metrics(outputs, batch)
        bsz = int(outputs["pred_motion_window"].shape[0])
        num_batches += 1
        num_samples += bsz
        for key, value in loss_metrics.items():
            scalar_totals[key].add(value, bsz)
        for key, value in per_sample.items():
            if key not in loss_metrics:
                scalar_totals[key].add(float(value.mean().detach().cpu().item()), bsz)

        per_sample_cpu = {key: value.detach().cpu() for key, value in per_sample.items()}
        for i in range(bsz):
            item_metrics = {
                "coarse_motion_error": float(per_sample_cpu["coarse_motion_error"][i]),
                "pred_motion_error": float(per_sample_cpu["pred_motion_error"][i]),
                "motion_improvement": float(per_sample_cpu["motion_improvement"][i]),
                "coarse_contact_motion_error": float(per_sample_cpu["coarse_contact_motion_error"][i]),
                "pred_contact_motion_error": float(per_sample_cpu["pred_contact_motion_error"][i]),
                "contact_motion_improvement": float(per_sample_cpu["contact_motion_improvement"][i]),
            }
            for group_name in breakdown:
                value = str(batch[group_name][i]) if group_name in batch else ""
                for metric_name, metric_value in item_metrics.items():
                    compound = f"{value}::{metric_name}"
                    if compound not in breakdown[group_name]:
                        breakdown[group_name][compound] = _Averager()
                    breakdown[group_name][compound].add(metric_value)

        if max_batches and num_batches >= int(max_batches):
            break

    flat_breakdown: dict[str, dict[str, dict[str, float]]] = {}
    for group_name, group_values in breakdown.items():
        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for compound, avg in group_values.items():
            if "::" not in compound:
                continue
            value, metric_name = compound.split("::", 1)
            grouped[value][metric_name] = avg.mean()
        flat_breakdown[group_name] = dict(sorted(grouped.items()))

    return {
        "num_batches": int(num_batches),
        "num_samples": int(num_samples),
        "metrics": {key: avg.mean() for key, avg in sorted(scalar_totals.items())},
        "breakdown": flat_breakdown,
    }
