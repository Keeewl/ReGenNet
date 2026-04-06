import torch
import torch.nn as nn

from model.refine.active_window import ActiveWindowSelector, ActiveWindowSelectorV2, _default_refine_joint_ids
from model.refine.surface_features import SurfaceFeatureBuilder
from model.refine.refine_head import RNetV1Head
from model.refine.refine_head_v2 import RNetV2Lite


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
