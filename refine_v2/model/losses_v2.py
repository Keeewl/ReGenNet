"""Losses for the first refine_v2 residual refiner."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class RefineV2LossConfig:
    lambda_motion: float = 1.0
    lambda_contact: float = 1.0
    lambda_smooth: float = 0.05
    lambda_region_dist: float = 0.0
    contact_frame_weight: float = 2.0
    smooth_l1_beta: float = 0.05


def _valid_frame_weights(valid_mask: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.float().view(valid_mask.shape[0], 1, 1, valid_mask.shape[1])
    return weights.expand_as(motion)


def _masked_mean(value: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (value * weights).sum() / weights.sum().clamp_min(eps)


def _smooth_l1_none(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, reduction="none", beta=float(beta))


class RefineV2Loss(nn.Module):
    def __init__(self, config: RefineV2LossConfig):
        super().__init__()
        self.config = config

    def _contact_frame_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        gt_mask = batch["gt_region_contact_mask_window"].float()
        return gt_mask.amax(dim=1).gt(0.0)

    def _coarse_contact_frame_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pred_mask = batch["coarse_region_contact_mask_window"].float()
        return pred_mask.amax(dim=1).gt(0.0)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if float(self.config.lambda_region_dist) != 0.0:
            raise NotImplementedError(
                "lambda_region_dist > 0 requires optional geometry forward, which is intentionally disabled "
                "for the first fast-path training loop."
            )
        pred = outputs["pred_motion_window"].float()
        delta = outputs["pred_delta_motion_window"].float()
        coarse = batch["coarse_motion_window"].float()
        gt = batch["gt_motion_window"].float()
        valid_mask = batch["valid_mask"].bool()
        valid_weights = _valid_frame_weights(valid_mask, pred)

        pred_err = _smooth_l1_none(pred, gt, self.config.smooth_l1_beta)
        coarse_err = _smooth_l1_none(coarse, gt, self.config.smooth_l1_beta)
        loss_motion = _masked_mean(pred_err, valid_weights)

        contact_frame = self._contact_frame_mask(batch) & valid_mask
        contact_weights = (1.0 + float(self.config.contact_frame_weight) * contact_frame.float()).view(
            pred.shape[0], 1, 1, pred.shape[-1]
        )
        loss_contact_weighted = _masked_mean(pred_err, valid_weights * contact_weights)

        if delta.shape[-1] > 1:
            delta_diff = torch.abs(delta[..., 1:] - delta[..., :-1])
            valid_pair = (valid_mask[:, 1:] & valid_mask[:, :-1]).float().view(delta.shape[0], 1, 1, -1)
            loss_smooth = _masked_mean(delta_diff, valid_pair.expand_as(delta_diff))
        else:
            loss_smooth = delta.sum() * 0.0

        loss_region_dist = pred.sum() * 0.0
        loss_total = (
            float(self.config.lambda_motion) * loss_motion
            + float(self.config.lambda_contact) * loss_contact_weighted
            + float(self.config.lambda_smooth) * loss_smooth
            + float(self.config.lambda_region_dist) * loss_region_dist
        )

        contact_weights_plain = contact_frame.float().view(pred.shape[0], 1, 1, pred.shape[-1]).expand_as(pred)
        contact_weight_sum = contact_weights_plain.sum()
        if contact_weight_sum > 0:
            contact_frame_motion_error = (torch.abs(pred - gt) * contact_weights_plain).sum() / contact_weight_sum
            coarse_contact_motion_error = (torch.abs(coarse - gt) * contact_weights_plain).sum() / contact_weight_sum
        else:
            contact_frame_motion_error = pred.sum() * 0.0
            coarse_contact_motion_error = pred.sum() * 0.0

        pred_motion_error = _masked_mean(torch.abs(pred - gt), valid_weights)
        coarse_motion_error = _masked_mean(torch.abs(coarse - gt), valid_weights)
        coarse_contact_frame = self._coarse_contact_frame_mask(batch) & valid_mask
        return {
            "loss_total": loss_total,
            "loss_motion": loss_motion,
            "loss_contact_weighted": loss_contact_weighted,
            "loss_smooth": loss_smooth,
            "loss_region_dist": loss_region_dist,
            "coarse_motion_error": coarse_motion_error,
            "pred_motion_error": pred_motion_error,
            "motion_improvement": coarse_motion_error - pred_motion_error,
            "coarse_contact_motion_error": coarse_contact_motion_error,
            "pred_contact_motion_error": contact_frame_motion_error,
            "contact_motion_improvement": coarse_contact_motion_error - contact_frame_motion_error,
            "contact_frame_ratio": contact_frame.float().mean(),
            "coarse_contact_frame_ratio": coarse_contact_frame.float().mean(),
        }
