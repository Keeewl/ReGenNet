import torch
import torch.nn as nn

from model.refine.active_window import (
    ActiveWindowSelector,
    ActiveWindowSelectorV2,
    _default_refine_joint_ids,
    build_oracle_active_mask,
    compute_overlap_metrics,
    expand_time_mask,
)
from model.refine.surface_features import (
    SurfaceFeatureBuilder,
    default_candidate_contact_pairs,
    default_part_joint_ids,
)
from model.refine.refine_head import RNetV1Head
from model.refine.refine_head_v2 import RNetV2Lite
from model.refine.refine_head_v3 import RNetV3Lite


class RNetV1(nn.Module):
    """
    Minimal refinement model for local pose residuals.
    """

    def __init__(
        self,
        njoints=56,
        nfeats=6,
        body_model="smplx",
        pose_rep="rot6d",
        refine_joint_ids=None,
        top_k=5,
        window_size=5,
        vel_threshold=None,
        geom_sigma=0.1,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.version = "v1"
        self.refine_joint_ids = refine_joint_ids or _default_refine_joint_ids()

        self.active_selector = ActiveWindowSelector(
            joint_ids=self.refine_joint_ids,
            top_k=top_k,
            window_size=window_size,
            vel_threshold=vel_threshold,
        )
        self.surface_builder = SurfaceFeatureBuilder(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=True,
            glob=True,
            sigma=geom_sigma,
        )

        input_dim = self.nfeats * 2 + self.surface_builder.feature_dim
        self.head = RNetV1Head(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.nfeats,
            dropout=dropout,
        )

    def forward(self, actor_motion, coarse_motion, gt_motion=None, lengths=None, return_aux=True):
        """
        actor_motion/coarse_motion: [B, J, 6, T]
        """
        device = coarse_motion.device
        batch_size, num_joints, _, num_frames = coarse_motion.shape
        if lengths is None:
            lengths = torch.full((batch_size,), num_frames, device=device, dtype=torch.long)

        actor_xyz = self.surface_builder.to_xyz(actor_motion)
        reactor_xyz = self.surface_builder.to_xyz(coarse_motion)

        active_mask, joint_mask, scores = self.active_selector.select(
            actor_xyz, reactor_xyz, lengths=lengths
        )

        geom_feat = self.surface_builder.build(
            actor_xyz,
            reactor_xyz,
            joint_ids=self.refine_joint_ids,
            lengths=lengths,
            active_mask=active_mask,
        )

        joint_ids = torch.as_tensor(self.refine_joint_ids, device=device, dtype=torch.long)
        coarse_local = coarse_motion.index_select(1, joint_ids)
        actor_local = actor_motion.index_select(1, joint_ids)
        coarse_local = coarse_local.permute(0, 3, 1, 2)
        actor_local = actor_local.permute(0, 3, 1, 2)

        head_in = torch.cat([coarse_local, actor_local, geom_feat], dim=-1)
        delta = self.head(head_in)

        delta = delta * active_mask[:, :, None, None].float()
        delta_full = torch.zeros_like(coarse_motion)
        delta_full.index_copy_(1, joint_ids, delta.permute(0, 2, 3, 1))
        joint_mask_full = torch.zeros(num_joints, device=device, dtype=torch.bool)
        joint_mask_full[joint_ids] = True
        delta_full = delta_full * joint_mask_full[None, :, None, None].float()

        refined = coarse_motion + delta_full

        if return_aux:
            aux = {
                "delta": delta,
                "active_mask": active_mask,
                "joint_mask": joint_mask,
                "scores": scores,
                "geom_feat": geom_feat,
            }
            if gt_motion is not None:
                aux["gt_motion"] = gt_motion
            return refined, aux
        return refined


class RNetV2(nn.Module):
    """
    Prior-guided refinement model (v2 lite).
    """

    def __init__(
        self,
        njoints=56,
        nfeats=6,
        body_model="smplx",
        pose_rep="rot6d",
        refine_joint_ids=None,
        top_k=5,
        window_size=5,
        vel_threshold=None,
        geom_sigma=0.1,
        selector_sigma=0.1,
        selector_alpha=1.0,
        selector_beta=0.5,
        selector_gamma=0.5,
        hidden_dim=256,
        num_temporal_blocks=2,
        dropout=0.1,
    ):
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.version = "v2"
        self.refine_joint_ids = refine_joint_ids or _default_refine_joint_ids()

        self.active_selector = ActiveWindowSelectorV2(
            joint_ids=self.refine_joint_ids,
            top_k=top_k,
            window_size=window_size,
            vel_threshold=vel_threshold,
            sigma_contact=selector_sigma,
            alpha=selector_alpha,
            beta=selector_beta,
            gamma=selector_gamma,
        )
        self.surface_builder = SurfaceFeatureBuilder(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=True,
            glob=True,
            sigma=geom_sigma,
        )

        input_dim = self.nfeats + self.surface_builder.feature_dim
        self.head = RNetV2Lite(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.nfeats,
            num_temporal_blocks=num_temporal_blocks,
            dropout=dropout,
            refine_joint_ids=self.refine_joint_ids,
        )

    def forward(self, actor_motion, coarse_motion, gt_motion=None, lengths=None, return_aux=True):
        """
        actor_motion/coarse_motion: [B, J, 6, T]
        """
        device = coarse_motion.device
        batch_size, num_joints, _, num_frames = coarse_motion.shape
        if lengths is None:
            lengths = torch.full((batch_size,), num_frames, device=device, dtype=torch.long)

        actor_xyz = self.surface_builder.to_xyz(actor_motion)
        reactor_xyz = self.surface_builder.to_xyz(coarse_motion)

        active_mask, joint_mask, scores = self.active_selector.select(
            actor_xyz, reactor_xyz, lengths=lengths
        )

        geom_feat = self.surface_builder.build(
            actor_xyz,
            reactor_xyz,
            joint_ids=self.refine_joint_ids,
            lengths=lengths,
            active_mask=active_mask,
        )

        joint_ids = torch.as_tensor(self.refine_joint_ids, device=device, dtype=torch.long)
        coarse_local = coarse_motion.index_select(1, joint_ids)
        coarse_local = coarse_local.permute(0, 3, 1, 2)

        delta = self.head(coarse_local, geom_feat)

        delta = delta * active_mask[:, :, None, None].float()
        delta_full = torch.zeros_like(coarse_motion)
        delta_full.index_copy_(1, joint_ids, delta.permute(0, 2, 3, 1))
        joint_mask_full = torch.zeros(num_joints, device=device, dtype=torch.bool)
        joint_mask_full[joint_ids] = True
        delta_full = delta_full * joint_mask_full[None, :, None, None].float()

        refined = coarse_motion + delta_full

        if return_aux:
            aux = {
                "delta": delta,
                "active_mask": active_mask,
                "joint_mask": joint_mask,
                "scores": scores,
                "geom_feat": geom_feat,
            }
            if gt_motion is not None:
                aux["gt_motion"] = gt_motion
            return refined, aux
        return refined


class RNetV3(nn.Module):
    """
    Physics-first refinement model (v3).
    """

    def __init__(
        self,
        njoints=56,
        nfeats=6,
        body_model="smplx",
        pose_rep="rot6d",
        refine_joint_ids=None,
        top_k=5,
        window_size=7,
        train_window_size=10,
        vel_threshold=None,
        geom_sigma=0.1,
        selector_sigma=0.1,
        selector_alpha=1.0,
        selector_beta=0.5,
        selector_gamma=0.5,
        hidden_dim=256,
        num_temporal_blocks=2,
        dropout=0.1,
        pair_mode="semantic_nearest",
        topk_pairs=3,
        pair_reduce="mean",
        part_joint_ids=None,
        candidate_contact_pairs=None,
        use_contact_feature_aug=True,
        pair_feature_topk=3,
        use_closing_speed=True,
        use_part_contact_summary=True,
        tau_contact=0.1,
        tau_near=0.18,
        contact_error_margin=0.05,
        gate_level="joint",
        gate_init_bias=-2.0,
        bound_mode="tanh",
        delta_max=0.15,
    ):
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.version = "v3"
        self.refine_joint_ids = refine_joint_ids or _default_refine_joint_ids()
        self.train_window_size = int(train_window_size) if train_window_size is not None else None
        self.pair_mode = pair_mode
        self.topk_pairs = int(topk_pairs)
        self.pair_reduce = pair_reduce
        self.part_joint_ids = part_joint_ids or default_part_joint_ids()
        self.candidate_contact_pairs = candidate_contact_pairs or default_candidate_contact_pairs()
        self.tau_contact = float(tau_contact)
        self.tau_near = float(tau_near)
        self.contact_error_margin = float(contact_error_margin)
        self.gate_level = gate_level
        self.bound_mode = bound_mode
        self.delta_max = float(delta_max)

        self.active_selector = ActiveWindowSelectorV2(
            joint_ids=self.refine_joint_ids,
            top_k=top_k,
            window_size=window_size,
            vel_threshold=vel_threshold,
            sigma_contact=selector_sigma,
            alpha=selector_alpha,
            beta=selector_beta,
            gamma=selector_gamma,
        )
        self.surface_builder = SurfaceFeatureBuilder(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=True,
            glob=True,
            sigma=geom_sigma,
            use_contact_feature_aug=use_contact_feature_aug,
            part_joint_ids=self.part_joint_ids,
            candidate_pairs=self.candidate_contact_pairs,
            pair_feature_topk=pair_feature_topk,
            use_closing_speed=use_closing_speed,
            use_part_contact_summary=use_part_contact_summary,
        )

        input_dim = self.nfeats + self.surface_builder.feature_dim
        self.head = RNetV3Lite(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.nfeats,
            num_temporal_blocks=num_temporal_blocks,
            dropout=dropout,
            refine_joint_ids=self.refine_joint_ids,
            part_joint_ids=self.part_joint_ids,
            gate_init_bias=gate_init_bias,
        )

    def _bounded_delta(self, delta_raw):
        if self.bound_mode == "tanh":
            return self.delta_max * torch.tanh(delta_raw / max(self.delta_max, 1e-6))
        if self.bound_mode == "clip":
            return torch.clamp(delta_raw, -self.delta_max, self.delta_max)
        if self.bound_mode == "none":
            return delta_raw
        raise ValueError(f"Unsupported bound_mode: {self.bound_mode}")

    def forward(self, actor_motion, coarse_motion, gt_motion=None, lengths=None, return_aux=True):
        """
        actor_motion/coarse_motion: [B, J, 6, T]
        """
        device = coarse_motion.device
        batch_size, num_joints, _, num_frames = coarse_motion.shape
        if lengths is None:
            lengths = torch.full((batch_size,), num_frames, device=device, dtype=torch.long)

        actor_xyz = self.surface_builder.to_xyz(actor_motion)
        reactor_xyz = self.surface_builder.to_xyz(coarse_motion)

        overlap_metrics = None
        train_mask = None
        if gt_motion is not None:
            gt_xyz = self.surface_builder.to_xyz(gt_motion)
            train_mask, oracle_info = build_oracle_active_mask(
                actor_xyz,
                reactor_xyz,
                gt_xyz,
                self.active_selector,
                lengths=lengths,
                tau_contact=self.tau_contact,
                tau_near=self.tau_near,
                contact_error_margin=self.contact_error_margin,
                train_window_size=self.train_window_size,
            )
            active_mask = train_mask
            joint_mask = oracle_info["joint_mask"]
            scores = oracle_info["scores"]
            overlap_metrics = compute_overlap_metrics(
                oracle_info["coarse_mask"], oracle_info["gt_contact_mask"]
            )
            near_metrics = compute_overlap_metrics(
                oracle_info["coarse_mask"], oracle_info["gt_near_mask"]
            )
            overlap_metrics["overlap_iou_near"] = near_metrics["overlap_iou"]
            overlap_metrics["gt_near_recall_by_coarse_risk"] = near_metrics[
                "gt_contact_recall_by_coarse_risk"
            ]
            overlap_metrics["coarse_risk_precision_wrt_gt_near"] = near_metrics[
                "coarse_risk_precision_wrt_gt"
            ]
            if self.train_window_size is not None and self.train_window_size > 0:
                expanded = expand_time_mask(
                    oracle_info["coarse_mask"], self.train_window_size
                )
                expanded_metrics = compute_overlap_metrics(
                    expanded, oracle_info["gt_contact_mask"]
                )
                overlap_metrics["overlap_iou_expanded"] = expanded_metrics["overlap_iou"]
                overlap_metrics[
                    "gt_contact_recall_by_expanded_coarse_risk"
                ] = expanded_metrics["gt_contact_recall_by_coarse_risk"]
        else:
            active_mask, joint_mask, scores = self.active_selector.select(
                actor_xyz, reactor_xyz, lengths=lengths
            )

        geom_feat = self.surface_builder.build(
            actor_xyz,
            reactor_xyz,
            joint_ids=self.refine_joint_ids,
            lengths=lengths,
            active_mask=active_mask,
        )

        joint_ids = torch.as_tensor(self.refine_joint_ids, device=device, dtype=torch.long)
        coarse_local = coarse_motion.index_select(1, joint_ids)
        coarse_local = coarse_local.permute(0, 3, 1, 2)

        delta_raw, gate_logits = self.head(coarse_local, geom_feat)
        gate = torch.sigmoid(gate_logits)
        delta_bounded = self._bounded_delta(delta_raw)
        delta = delta_bounded * gate

        delta = delta * active_mask[:, :, None, None].float()
        delta_full = torch.zeros_like(coarse_motion)
        delta_full.index_copy_(1, joint_ids, delta.permute(0, 2, 3, 1))
        joint_mask_full = torch.zeros(num_joints, device=device, dtype=torch.bool)
        joint_mask_full[joint_ids] = True
        delta_full = delta_full * joint_mask_full[None, :, None, None].float()

        refined = coarse_motion + delta_full

        if return_aux:
            aux = {
                "delta": delta,
                "active_mask": active_mask,
                "joint_mask": joint_mask,
                "scores": scores,
                "geom_feat": geom_feat,
                "delta_raw": delta_raw,
                "delta_bounded": delta_bounded,
                "gate": gate,
            }
            if train_mask is not None:
                aux["train_mask"] = train_mask
            if overlap_metrics is not None:
                aux.update(overlap_metrics)
            if gt_motion is not None:
                aux["gt_motion"] = gt_motion
            return refined, aux
        return refined
