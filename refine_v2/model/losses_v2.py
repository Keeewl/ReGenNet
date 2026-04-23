"""Losses for the first refine_v2 residual refiner."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from refine_v2.model.joint_groups import (
    ContactGroupWeights,
    MotionGroupWeights,
    PhasePreserveGroupWeights,
    contact_group_weight_tensor,
    group_weight_tensor,
    phase_preserve_group_weight_tensor,
)


@dataclass
class RefineV2LossConfig:
    lambda_motion: float = 1.0
    lambda_contact: float = 1.0
    lambda_smooth: float = 0.05
    lambda_region_dist: float = 0.0
    lambda_boundary_trans: float = 0.0
    lambda_phase_preserve: float = 0.0
    lambda_contact_geometry: float = 0.0
    lambda_gt_relative_overclose: float = 0.0
    boundary_trans_frames: int = 2
    phase_preserve_power: float = 2.0
    phase_preserve_transl_weight: float = 2.0
    phase_preserve_root_weight: float = 1.0
    phase_preserve_lower_body_weight: float = 0.5
    phase_preserve_torso_weight: float = 0.3
    phase_preserve_arm_weight: float = 0.1
    phase_preserve_hand_weight: float = 0.05
    contact_frame_weight: float = 2.0
    smooth_l1_beta: float = 0.05
    use_group_weighted_loss: bool = False
    selected_hand_motion_weight: float = 3.0
    same_side_arm_motion_weight: float = 2.0
    other_hand_arm_motion_weight: float = 1.0
    torso_root_motion_weight: float = 0.75
    lower_body_motion_weight: float = 0.25
    transl_motion_weight: float = 0.25
    use_hand_arm_contact_loss: bool = False
    selected_hand_contact_weight: float = 4.0
    same_side_arm_contact_weight: float = 3.0
    other_upper_contact_weight: float = 1.0
    body_contact_weight: float = 0.5
    contact_geometry_weight_scale: float = 0.05
    gt_relative_overclose_margin: float = 0.005


def _valid_frame_weights(valid_mask: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.float().view(valid_mask.shape[0], 1, 1, valid_mask.shape[1])
    return weights.expand_as(motion)


def _masked_mean(value: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (value * weights).sum() / weights.sum().clamp_min(eps)


def _smooth_l1_none(pred: torch.Tensor, target: torch.Tensor, beta: float) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, reduction="none", beta=float(beta))


def _boundary_frame_mask(valid_mask: torch.Tensor, num_frames: int, k: int) -> torch.Tensor:
    k = max(0, int(k))
    mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    if k <= 0 or num_frames <= 0:
        return mask
    k = min(k, int(num_frames))
    arange = torch.arange(num_frames, device=valid_mask.device).view(1, -1)
    lengths = valid_mask.long().sum(dim=1).clamp_min(1).view(-1, 1)
    start_mask = arange < k
    end_mask = arange >= (lengths - k).clamp_min(0)
    return (start_mask | end_mask) & valid_mask.bool()


def _phase_preserve_weights(valid_mask: torch.Tensor, motion: torch.Tensor, power: float) -> torch.Tensor:
    num_frames = int(motion.shape[-1])
    if num_frames <= 1:
        phase = torch.ones((1, num_frames), device=motion.device, dtype=motion.dtype)
    else:
        frame = torch.arange(num_frames, device=motion.device, dtype=motion.dtype)
        center = (float(num_frames) - 1.0) / 2.0
        denom = max(center, 1e-6)
        phase = torch.abs(frame - center) / denom
        phase = phase.clamp(0.0, 1.0).pow(float(power)).view(1, num_frames)
    weights = phase * valid_mask.to(device=motion.device, dtype=motion.dtype)
    return weights.view(valid_mask.shape[0], 1, 1, num_frames).expand_as(motion)


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
        if bool(self.config.use_group_weighted_loss):
            motion_group_weights = group_weight_tensor(
                hand_side_id=batch["hand_side_id"],
                num_joints=pred.shape[1],
                num_channels=pred.shape[2],
                num_frames=pred.shape[-1],
                device=pred.device,
                dtype=pred.dtype,
                weights=MotionGroupWeights(
                    selected_hand=float(self.config.selected_hand_motion_weight),
                    same_side_arm=float(self.config.same_side_arm_motion_weight),
                    other_hand_arm=float(self.config.other_hand_arm_motion_weight),
                    torso_root=float(self.config.torso_root_motion_weight),
                    lower_body=float(self.config.lower_body_motion_weight),
                    transl=float(self.config.transl_motion_weight),
                ),
            )
            loss_motion = _masked_mean(pred_err, valid_weights * motion_group_weights)
        else:
            loss_motion = _masked_mean(pred_err, valid_weights)

        contact_frame = self._contact_frame_mask(batch) & valid_mask
        if bool(self.config.use_hand_arm_contact_loss):
            contact_group_weights = contact_group_weight_tensor(
                hand_side_id=batch["hand_side_id"],
                num_joints=pred.shape[1],
                num_channels=pred.shape[2],
                num_frames=pred.shape[-1],
                device=pred.device,
                dtype=pred.dtype,
                weights=ContactGroupWeights(
                    selected_hand=float(self.config.selected_hand_contact_weight),
                    same_side_arm=float(self.config.same_side_arm_contact_weight),
                    other_upper=float(self.config.other_upper_contact_weight),
                    body=float(self.config.body_contact_weight),
                ),
            )
            contact_weights = 1.0 + contact_frame.float().view(pred.shape[0], 1, 1, pred.shape[-1]) * contact_group_weights
        else:
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
        if float(self.config.lambda_contact_geometry) != 0.0:
            if "contact_geometry_weight_window" not in batch:
                raise KeyError(
                    "lambda_contact_geometry > 0 requires geometry cache v2 field "
                    "contact_geometry_weight_window."
                )
            geom_frame_weight = batch["contact_geometry_weight_window"].float().amax(dim=1)
            geom_frame_weight = torch.nan_to_num(geom_frame_weight, nan=0.0, posinf=0.0, neginf=0.0)
            geom_frame_weight = (
                geom_frame_weight / max(float(self.config.contact_geometry_weight_scale), 1e-8)
            ).clamp(0.0, 1.0)
            geom_frame_weight = geom_frame_weight * valid_mask.float()
            geom_group_weights = contact_group_weight_tensor(
                hand_side_id=batch["hand_side_id"],
                num_joints=pred.shape[1],
                num_channels=pred.shape[2],
                num_frames=pred.shape[-1],
                device=pred.device,
                dtype=pred.dtype,
                weights=ContactGroupWeights(
                    selected_hand=float(self.config.selected_hand_contact_weight),
                    same_side_arm=float(self.config.same_side_arm_contact_weight),
                    other_upper=0.0,
                    body=0.0,
                ),
            )
            loss_contact_geometry = _masked_mean(
                pred_err,
                geom_frame_weight.view(pred.shape[0], 1, 1, pred.shape[-1]) * geom_group_weights,
            )
        else:
            loss_contact_geometry = pred.sum() * 0.0

        if float(self.config.lambda_gt_relative_overclose) != 0.0:
            if "topk_nearest_dist_gap_window" not in batch:
                raise KeyError(
                    "lambda_gt_relative_overclose > 0 requires geometry cache v2 field "
                    "topk_nearest_dist_gap_window."
                )
            overclose_weight = (
                -batch["topk_nearest_dist_gap_window"].float()
                - float(self.config.gt_relative_overclose_margin)
            ).clamp_min(0.0).amax(dim=1)
            overclose_weight = (
                overclose_weight / max(float(self.config.contact_geometry_weight_scale), 1e-8)
            ).clamp(0.0, 1.0)
            overclose_weight = overclose_weight * valid_mask.float()
            overclose_group_weights = contact_group_weight_tensor(
                hand_side_id=batch["hand_side_id"],
                num_joints=pred.shape[1],
                num_channels=pred.shape[2],
                num_frames=pred.shape[-1],
                device=pred.device,
                dtype=pred.dtype,
                weights=ContactGroupWeights(
                    selected_hand=1.0,
                    same_side_arm=0.5,
                    other_upper=0.0,
                    body=0.0,
                ),
            )
            loss_gt_relative_overclose = _masked_mean(
                torch.abs(delta),
                overclose_weight.view(pred.shape[0], 1, 1, pred.shape[-1]) * overclose_group_weights,
            )
        else:
            loss_gt_relative_overclose = pred.sum() * 0.0

        if float(self.config.lambda_phase_preserve) != 0.0:
            phase_weights = _phase_preserve_weights(valid_mask, pred, float(self.config.phase_preserve_power))
            group_weights = phase_preserve_group_weight_tensor(
                num_joints=pred.shape[1],
                num_channels=pred.shape[2],
                num_frames=pred.shape[-1],
                device=pred.device,
                dtype=pred.dtype,
                weights=PhasePreserveGroupWeights(
                    hand=float(self.config.phase_preserve_hand_weight),
                    arm=float(self.config.phase_preserve_arm_weight),
                    torso=float(self.config.phase_preserve_torso_weight),
                    root=float(self.config.phase_preserve_root_weight),
                    transl=float(self.config.phase_preserve_transl_weight),
                    lower_body=float(self.config.phase_preserve_lower_body_weight),
                ),
            )
            preserve_err = _smooth_l1_none(pred, coarse, self.config.smooth_l1_beta)
            loss_phase_preserve = _masked_mean(preserve_err, phase_weights * group_weights)
        else:
            loss_phase_preserve = pred.sum() * 0.0
        if (
            float(self.config.lambda_boundary_trans) != 0.0
            and pred.shape[0] > 0
            and pred.shape[1] > 55
            and pred.shape[2] >= 3
        ):
            boundary_mask = _boundary_frame_mask(
                valid_mask,
                int(pred.shape[-1]),
                int(self.config.boundary_trans_frames),
            )
            boundary_weights = boundary_mask.float().view(pred.shape[0], 1, 1, pred.shape[-1])
            pred_trans = pred[:, 55:56, :3, :]
            coarse_trans = coarse[:, 55:56, :3, :]
            boundary_err = _smooth_l1_none(pred_trans, coarse_trans, self.config.smooth_l1_beta)
            loss_boundary_trans = _masked_mean(boundary_err, boundary_weights.expand_as(boundary_err))
        else:
            loss_boundary_trans = pred.sum() * 0.0
        loss_total = (
            float(self.config.lambda_motion) * loss_motion
            + float(self.config.lambda_contact) * loss_contact_weighted
            + float(self.config.lambda_smooth) * loss_smooth
            + float(self.config.lambda_region_dist) * loss_region_dist
            + float(self.config.lambda_boundary_trans) * loss_boundary_trans
            + float(self.config.lambda_phase_preserve) * loss_phase_preserve
            + float(self.config.lambda_contact_geometry) * loss_contact_geometry
            + float(self.config.lambda_gt_relative_overclose) * loss_gt_relative_overclose
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
            "loss_boundary_trans": loss_boundary_trans,
            "loss_phase_preserve": loss_phase_preserve,
            "loss_contact_geometry": loss_contact_geometry,
            "loss_gt_relative_overclose": loss_gt_relative_overclose,
            "coarse_motion_error": coarse_motion_error,
            "pred_motion_error": pred_motion_error,
            "motion_improvement": coarse_motion_error - pred_motion_error,
            "coarse_contact_motion_error": coarse_contact_motion_error,
            "pred_contact_motion_error": contact_frame_motion_error,
            "contact_motion_improvement": coarse_contact_motion_error - contact_frame_motion_error,
            "contact_frame_ratio": contact_frame.float().mean(),
            "coarse_contact_frame_ratio": coarse_contact_frame.float().mean(),
        }
