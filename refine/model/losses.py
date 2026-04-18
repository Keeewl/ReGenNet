"""Joint-based local refinement losses for Stage2-lite.

This module implements the first loss baseline for the current hand-centric,
contact-oriented local refiner. The total loss is:

    L = lambda_res * L_res
      + lambda_smooth * L_smooth
      + lambda_contact * L_contact_proxy
      + lambda_id * L_identity

The design intentionally stays local and joint-based. It does not use mesh
losses, diffusion losses, or any old Stage2 runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class JointRefinementLossConfig:
    residual_loss_type: str = "smooth_l1"
    lambda_res: float = 1.0
    lambda_smooth: float = 0.1
    lambda_contact: float = 0.2
    lambda_identity: float = 0.05
    core_weight: float = 1.0
    support_weight: float = 0.5
    identity_core_weight: float = 0.25
    identity_support_weight: float = 1.0
    contact_use_min_joint_distance: bool = True
    contact_coord_dim: int = 3
    eps: float = 1e-6


def _zero_like_reference(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


class JointRefinementLoss(nn.Module):
    """Contact-oriented local refinement loss for the joint-based Stage2-lite baseline."""

    def __init__(self, config: JointRefinementLossConfig | None = None):
        super().__init__()
        self.config = config or JointRefinementLossConfig()

    def _build_joint_weights(
        self,
        core_mask: torch.Tensor,
        support_mask: torch.Tensor,
        *,
        core_weight: float,
        support_weight: float,
    ) -> torch.Tensor:
        core_mask = core_mask.bool()
        support_mask = support_mask.bool()
        weights = torch.full_like(core_mask, float(support_weight), dtype=torch.float32)
        weights = weights.masked_fill(core_mask, float(core_weight))
        weights = weights.masked_fill(support_mask, float(support_weight))
        return weights.view(1, -1, 1, 1)

    def _build_time_pair_mask(self, time_mask: torch.Tensor) -> torch.Tensor:
        return time_mask[:, 1:] & time_mask[:, :-1]

    def _masked_loss_reduce(
        self,
        loss_tensor: torch.Tensor,
        time_mask: torch.Tensor,
        joint_weights: torch.Tensor,
    ) -> torch.Tensor:
        valid = time_mask[:, None, None, :].to(dtype=loss_tensor.dtype, device=loss_tensor.device)
        weighted = loss_tensor * valid * joint_weights.to(device=loss_tensor.device, dtype=loss_tensor.dtype)
        denom = (
            valid.expand_as(loss_tensor)
            * joint_weights.to(device=loss_tensor.device, dtype=loss_tensor.dtype)
        ).sum()
        return weighted.sum() / denom.clamp_min(self.config.eps)

    def _masked_residual_loss(
        self,
        pred_delta: torch.Tensor,
        target_delta: torch.Tensor,
        time_mask: torch.Tensor,
        joint_weights: torch.Tensor,
    ) -> torch.Tensor:
        loss_type = self.config.residual_loss_type.lower()
        if loss_type == "l1":
            loss_tensor = (pred_delta - target_delta).abs()
        elif loss_type in {"l2", "mse"}:
            loss_tensor = (pred_delta - target_delta).pow(2)
        elif loss_type == "smooth_l1":
            loss_tensor = F.smooth_l1_loss(pred_delta, target_delta, reduction="none")
        else:
            raise ValueError(f"Unsupported residual_loss_type: {self.config.residual_loss_type}")
        return self._masked_loss_reduce(loss_tensor, time_mask, joint_weights)

    def _temporal_smooth_loss(
        self,
        delta_local: torch.Tensor,
        time_mask: torch.Tensor,
        joint_weights: torch.Tensor,
    ) -> torch.Tensor:
        if delta_local.shape[-1] <= 1:
            return _zero_like_reference(delta_local)
        diff = delta_local[..., 1:] - delta_local[..., :-1]
        pair_mask = self._build_time_pair_mask(time_mask)
        return self._masked_loss_reduce(diff.pow(2), pair_mask, joint_weights)

    def _proxy_coord_dim(self, reactor_local: torch.Tensor, actor_target_local: torch.Tensor) -> int:
        return min(
            int(self.config.contact_coord_dim),
            int(reactor_local.shape[2]),
            int(actor_target_local.shape[2]),
        )

    def _compute_hand_center_from_local(
        self,
        reactor_local: torch.Tensor,
        core_mask: torch.Tensor,
        coord_dim: int,
    ) -> torch.Tensor:
        hand_local = reactor_local[:, core_mask.bool(), :coord_dim, :]
        return hand_local.mean(dim=1)

    def _compute_target_center_from_local(
        self,
        actor_target_local: torch.Tensor,
        actor_target_mask: torch.Tensor,
        coord_dim: int,
    ) -> torch.Tensor:
        target_local = actor_target_local[:, :, :coord_dim, :]
        target_mask = actor_target_mask[:, :, None, None].to(dtype=target_local.dtype, device=target_local.device)
        summed = (target_local * target_mask).sum(dim=1)
        denom = target_mask.sum(dim=1).clamp_min(self.config.eps)
        return summed / denom

    def _contact_proxy_distance(
        self,
        reactor_local: torch.Tensor,
        actor_target_local: torch.Tensor,
        actor_target_mask: torch.Tensor,
        core_mask: torch.Tensor,
    ) -> torch.Tensor:
        coord_dim = self._proxy_coord_dim(reactor_local, actor_target_local)
        if coord_dim <= 0:
            raise ValueError("contact proxy requires at least one coordinate channel.")

        if self.config.contact_use_min_joint_distance:
            reactor_core = reactor_local[:, core_mask.bool(), :coord_dim, :].permute(0, 3, 1, 2).contiguous()
            actor_target = actor_target_local[:, :, :coord_dim, :].permute(0, 3, 1, 2).contiguous()
            pairwise = torch.cdist(reactor_core, actor_target)
            target_invalid = (~actor_target_mask.bool())[:, None, None, :]
            pairwise = pairwise.masked_fill(target_invalid, float("inf"))
            min_dist = pairwise.amin(dim=-1).amin(dim=-1)
            finite_mask = torch.isfinite(min_dist)
            if not bool(finite_mask.all()):
                min_dist = min_dist.masked_fill(~finite_mask, 0.0)
            return min_dist

        hand_center = self._compute_hand_center_from_local(reactor_local, core_mask, coord_dim)
        target_center = self._compute_target_center_from_local(actor_target_local, actor_target_mask, coord_dim)
        rel = hand_center - target_center
        return rel.norm(dim=1)

    def _contact_improvement_loss(
        self,
        coarse_local: torch.Tensor,
        refined_local: torch.Tensor,
        actor_target_local: torch.Tensor,
        actor_target_mask: torch.Tensor,
        core_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        coarse_dist = self._contact_proxy_distance(
            coarse_local,
            actor_target_local,
            actor_target_mask,
            core_mask,
        )
        refined_dist = self._contact_proxy_distance(
            refined_local,
            actor_target_local,
            actor_target_mask,
            core_mask,
        )
        deterioration = F.relu(refined_dist - coarse_dist)
        valid = time_mask.to(dtype=deterioration.dtype, device=deterioration.device)
        denom = valid.sum()
        return (deterioration * valid).sum() / denom.clamp_min(self.config.eps)

    def _identity_regularization(
        self,
        delta_local: torch.Tensor,
        time_mask: torch.Tensor,
        core_mask: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        joint_weights = self._build_joint_weights(
            core_mask,
            support_mask,
            core_weight=self.config.identity_core_weight,
            support_weight=self.config.identity_support_weight,
        )
        return self._masked_loss_reduce(delta_local.pow(2), time_mask, joint_weights)

    def forward(self, model_out, window_batch):
        coarse_local = window_batch["coarse_local"]
        gt_local = window_batch.get("gt_local")
        time_mask = window_batch["time_mask"].bool().to(coarse_local.device)
        core_mask = window_batch["core_mask"].bool().to(coarse_local.device)
        support_mask = window_batch["support_mask"].bool().to(coarse_local.device)
        actor_target_local = window_batch["actor_target_local"].to(coarse_local.device)
        actor_target_mask = window_batch["actor_target_mask"].bool().to(coarse_local.device)

        delta_local = model_out["delta_local"]
        refined_local = model_out["refined_local"]

        if gt_local is None:
            raise ValueError("JointRefinementLoss requires window_batch['gt_local'] for residual supervision.")
        gt_local = gt_local.to(coarse_local.device)

        if coarse_local.shape[0] == 0:
            zero = _zero_like_reference(coarse_local)
            return {
                "loss_total": zero,
                "loss_res": zero,
                "loss_smooth": zero,
                "loss_contact_proxy": zero,
                "loss_identity": zero,
            }

        residual_joint_weights = self._build_joint_weights(
            core_mask,
            support_mask,
            core_weight=self.config.core_weight,
            support_weight=self.config.support_weight,
        ).to(coarse_local.device)

        target_delta = gt_local - coarse_local
        loss_res = self._masked_residual_loss(
            delta_local,
            target_delta,
            time_mask,
            residual_joint_weights,
        )
        loss_smooth = self._temporal_smooth_loss(
            delta_local,
            time_mask,
            residual_joint_weights,
        )
        loss_contact_proxy = self._contact_improvement_loss(
            coarse_local,
            refined_local,
            actor_target_local,
            actor_target_mask,
            core_mask,
            time_mask,
        )
        loss_identity = self._identity_regularization(
            delta_local,
            time_mask,
            core_mask,
            support_mask,
        )

        loss_total = (
            self.config.lambda_res * loss_res
            + self.config.lambda_smooth * loss_smooth
            + self.config.lambda_contact * loss_contact_proxy
            + self.config.lambda_identity * loss_identity
        )
        return {
            "loss_total": loss_total,
            "loss_res": loss_res,
            "loss_smooth": loss_smooth,
            "loss_contact_proxy": loss_contact_proxy,
            "loss_identity": loss_identity,
        }
