import math

import torch

from model.contact.contact_defs import (
    ACTOR_PART_JOINT_IDS,
    ACTOR_PART_NAMES,
    FINGER_BASE_IDS,
    FINGER_TIP_IDS,
    HAND_JOINT_IDS,
    HAND_SIDES,
    WRIST_JOINT_IDS,
)
from model.contact.contact_geometry import ContactGeometry, safe_normalize, temporal_diff, topk_pairwise_distance
from model.crefine.mesh_regions import get_mesh_region_provider
from model.crefine.restored_body_model import RestoredBodyModelForward
from model.crefine.restored_space import restore_motion_batch, validate_restoration_metadata


def _softmin_pairwise_distance(a_xyz, b_xyz, beta=30.0):
    if a_xyz is None or b_xyz is None or a_xyz.numel() == 0 or b_xyz.numel() == 0:
        if a_xyz is not None:
            shape = a_xyz.shape[:2]
            device = a_xyz.device
            dtype = a_xyz.dtype
        elif b_xyz is not None:
            shape = b_xyz.shape[:2]
            device = b_xyz.device
            dtype = b_xyz.dtype
        else:
            raise ValueError("At least one input patch must be present.")
        return torch.full(shape, 1e6, device=device, dtype=dtype)
    dist = torch.linalg.norm(a_xyz[:, :, :, None, :] - b_xyz[:, :, None, :, :], dim=-1)
    dist = dist.reshape(dist.shape[0], dist.shape[1], -1)
    beta = float(beta)
    softmin = -torch.logsumexp(-beta * dist, dim=-1) / max(beta, 1e-6)
    count = max(int(dist.shape[-1]), 1)
    return (softmin + math.log(count) / max(beta, 1e-6)).clamp(min=0.0)


class HandContactFeatureBuilder:
    """
    Restored-shape-aware proposal feature builder.

    The proposal stays lightweight, but features are computed in restored pair space
    and explicitly encode mesh proximity, target/nontarget context, temporal phase
    cues, and shape metadata.
    """

    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        topk=3,
        sigma=0.1,
        density="small",
        softmin_beta=30.0,
        device="cpu",
    ):
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.geometry = ContactGeometry(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device=device,
        )
        self.body_forward = RestoredBodyModelForward(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device=device,
        )
        self.mesh_provider = get_mesh_region_provider(
            density=density,
            body_model=body_model,
            pose_rep=pose_rep,
        )
        self.topk = int(topk)
        self.sigma = float(sigma)
        self.softmin_beta = float(softmin_beta)

        self.shape_summary_dim = 8
        self.temporal_dim = 5
        self.hand_dim = 42
        self.part_dim = 27
        self.relation_dim = 14

    def _ensure_device(self, device):
        self.body_forward.to(device)

    def _to_vertices(self, motion, betas=None, gender_id=None, body_model_type=None):
        self._ensure_device(motion.device)
        num_frames = motion.shape[-1]
        mask = torch.ones(motion.shape[0], num_frames, device=motion.device, dtype=torch.bool)
        return self.body_forward.motion_to_xyz(
            motion,
            jointstype="vertices",
            betas=betas,
            gender_id=gender_id,
            mask=mask,
            body_model_type=body_model_type,
        )

    def _gather_patch(self, vertices, ids):
        if not ids:
            return None
        ids_t = torch.as_tensor(ids, device=vertices.device, dtype=torch.long)
        return vertices.index_select(1, ids_t).permute(0, 3, 1, 2).contiguous()

    def _build_shape_summary(self, metadata, num_frames, device, dtype):
        actor_betas = metadata["actor_betas"].to(device=device, dtype=dtype)
        reactor_betas = metadata["reactor_betas"].to(device=device, dtype=dtype)
        actor_gender = metadata["actor_gender_id"].to(device=device, dtype=dtype).view(-1, 1)
        reactor_gender = metadata["reactor_gender_id"].to(device=device, dtype=dtype).view(-1, 1)
        delta = actor_betas - reactor_betas
        summary = torch.stack(
            [
                actor_betas.mean(dim=-1),
                actor_betas.std(dim=-1, unbiased=False),
                reactor_betas.mean(dim=-1),
                reactor_betas.std(dim=-1, unbiased=False),
                delta.abs().mean(dim=-1),
                torch.linalg.norm(delta, dim=-1),
                actor_gender.squeeze(-1) / 2.0,
                reactor_gender.squeeze(-1) / 2.0,
            ],
            dim=-1,
        )
        return summary[:, None, None, :].expand(-1, num_frames, len(HAND_SIDES), -1)

    def _build_temporal_cues(self, lengths, metadata, num_frames, device, dtype):
        if lengths is None:
            lengths = torch.full((metadata["actor_betas"].shape[0],), num_frames, device=device, dtype=torch.long)
        else:
            lengths = torch.as_tensor(lengths, device=device, dtype=torch.long).view(-1)

        frame_ids = torch.arange(num_frames, device=device, dtype=dtype).view(1, num_frames)
        denom = lengths.to(dtype=dtype).view(-1, 1).clamp(min=1.0) - 1.0
        progress = torch.where(denom > 0, frame_ids / denom, torch.zeros_like(frame_ids))
        progress = progress.clamp(0.0, 1.0)

        if "raw_frame_ix" in metadata:
            raw_ix = metadata["raw_frame_ix"].to(device=device, dtype=dtype)
            raw_total = metadata["raw_nframes"].to(device=device, dtype=dtype).view(-1, 1).clamp(min=1.0)
            raw_progress = (raw_ix + 1.0) / raw_total
        else:
            raw_progress = progress

        if "processed_frame_ix" in metadata:
            proc_ix = metadata["processed_frame_ix"].to(device=device, dtype=dtype)
            frame_step = temporal_diff(proc_ix.unsqueeze(-1)).squeeze(-1)
        else:
            frame_step = temporal_diff(frame_ids.unsqueeze(-1)).squeeze(-1)
        frame_step = frame_step / frame_step.abs().amax(dim=1, keepdim=True).clamp(min=1.0)

        cues = torch.stack(
            [
                progress,
                torch.sin(progress * math.pi),
                torch.cos(progress * math.pi),
                raw_progress.clamp(0.0, 1.0),
                frame_step,
            ],
            dim=-1,
        )
        return cues

    def _build_hand_features(self, reactor_xyz, shape_summary, temporal_cues):
        batch_size, _, _, num_frames = reactor_xyz.shape
        device = reactor_xyz.device
        hand_features = []
        for side in HAND_SIDES:
            wrist_id = WRIST_JOINT_IDS[side]
            wrist_xyz = reactor_xyz[:, wrist_id].permute(0, 2, 1)
            wrist_vel = temporal_diff(wrist_xyz)
            wrist_acc = temporal_diff(wrist_vel)

            hand_ids = torch.as_tensor(HAND_JOINT_IDS[side], device=device, dtype=torch.long)
            hand_pos = reactor_xyz.index_select(1, hand_ids).permute(0, 3, 1, 2)
            hand_center = hand_pos.mean(dim=2)

            palm_ids = torch.as_tensor([wrist_id] + FINGER_BASE_IDS[side], device=device, dtype=torch.long)
            palm_pos = reactor_xyz.index_select(1, palm_ids).permute(0, 3, 1, 2)
            palm_center = palm_pos.mean(dim=2)

            tip_ids = torch.as_tensor(FINGER_TIP_IDS[side], device=device, dtype=torch.long)
            tip_pos = reactor_xyz.index_select(1, tip_ids).permute(0, 3, 1, 2)
            tip_center = tip_pos.mean(dim=2)

            hand_dir = safe_normalize(hand_center - wrist_xyz)
            tip_dir = safe_normalize(tip_center - wrist_xyz)
            hand_speed = torch.linalg.norm(wrist_vel, dim=-1, keepdim=True)

            tip_to_palm = torch.linalg.norm(tip_pos - palm_center[:, :, None, :], dim=-1)
            openness = tip_to_palm.mean(dim=-1, keepdim=True)

            pairwise = torch.linalg.norm(
                tip_pos[:, :, :, None, :] - tip_pos[:, :, None, :, :],
                dim=-1,
            )
            eye = torch.eye(tip_pos.shape[2], device=device, dtype=torch.bool)
            spread = pairwise.masked_fill(eye, 0.0).sum(dim=(-1, -2))
            spread = spread / float(tip_pos.shape[2] * max(tip_pos.shape[2] - 1, 1))
            spread = spread.unsqueeze(-1)

            base_ids = torch.as_tensor(FINGER_BASE_IDS[side], device=device, dtype=torch.long)
            base_pos = reactor_xyz.index_select(1, base_ids).permute(0, 3, 1, 2)
            base_dist = torch.linalg.norm(base_pos - wrist_xyz[:, :, None, :], dim=-1)
            tip_dist_wrist = torch.linalg.norm(tip_pos - wrist_xyz[:, :, None, :], dim=-1)
            flexion = ((base_dist - tip_dist_wrist) / base_dist.clamp(min=1e-6)).mean(dim=-1, keepdim=True)

            tip_var = tip_pos.var(dim=2, unbiased=False)
            tip_disp = torch.sqrt(torch.sum(tip_var, dim=-1, keepdim=True))

            side_idx = 0 if side == "left" else 1
            feat = torch.cat(
                [
                    wrist_xyz,
                    wrist_vel,
                    wrist_acc,
                    hand_center,
                    palm_center,
                    tip_center,
                    hand_dir,
                    tip_dir,
                    openness,
                    spread,
                    flexion,
                    tip_disp,
                    hand_speed,
                    temporal_cues,
                    shape_summary[:, :, side_idx],
                ],
                dim=-1,
            )
            hand_features.append(feat)

        return torch.stack(hand_features, dim=2)

    def _build_part_features(self, actor_xyz, actor_vertices, shape_summary, temporal_cues):
        device = actor_xyz.device
        part_features = []
        for part_name in ACTOR_PART_NAMES:
            part_ids = torch.as_tensor(ACTOR_PART_JOINT_IDS[part_name], device=device, dtype=torch.long)
            part_pos = actor_xyz.index_select(1, part_ids).permute(0, 3, 1, 2)
            part_center = part_pos.mean(dim=2)
            part_vel = temporal_diff(part_center)
            part_acc = temporal_diff(part_vel)
            spread = torch.linalg.norm(part_pos - part_center[:, :, None, :], dim=-1).mean(dim=2, keepdim=True)
            part_speed = torch.linalg.norm(part_vel, dim=-1, keepdim=True)

            target_patch_ids = []
            for ids in self.mesh_provider.actor_target_patch_ids(part_name).values():
                target_patch_ids.extend(ids)
            target_patch = self._gather_patch(actor_vertices, sorted(set(target_patch_ids)))
            if target_patch is None:
                target_center = actor_xyz.new_zeros(part_center.shape)
                target_vel = target_center.clone()
                target_dir = target_center.clone()
            else:
                target_center = target_patch.mean(dim=2)
                target_vel = temporal_diff(target_center)
                target_dir = safe_normalize(target_center - part_center)

            feat = torch.cat(
                [
                    part_center,
                    part_vel,
                    part_acc,
                    spread,
                    part_speed,
                    target_dir,
                    temporal_cues,
                    shape_summary[:, :, 0],
                ],
                dim=-1,
            )
            part_features.append(feat)
        return torch.stack(part_features, dim=2)

    def _build_relation_features(self, actor_xyz, reactor_xyz, actor_vertices, reactor_vertices, hand_feat, part_feat, metadata):
        batch_size, num_frames, _, _ = hand_feat.shape
        device = actor_xyz.device
        dtype = actor_xyz.dtype

        top1 = torch.zeros(batch_size, num_frames, 2, len(ACTOR_PART_NAMES), device=device, dtype=dtype)
        topk_mean = torch.zeros_like(top1)
        mesh_target_dist = torch.zeros_like(top1)
        mesh_nontarget_dist = torch.zeros_like(top1)
        target_to_nontarget = torch.zeros_like(top1)
        closing_joint = torch.zeros_like(top1)
        closing_mesh = torch.zeros_like(top1)
        rel_speed = torch.zeros_like(top1)
        approach_alignment = torch.zeros_like(top1)
        target_speed = torch.zeros_like(top1)
        target_approach = torch.zeros_like(top1)

        delta_betas = metadata["actor_betas"].to(device=device, dtype=dtype) - metadata["reactor_betas"].to(
            device=device, dtype=dtype
        )
        shape_delta_norm = torch.linalg.norm(delta_betas, dim=-1).view(-1, 1, 1, 1).expand_as(top1)

        hand_unions = {}
        for side in HAND_SIDES:
            union = []
            for ids in self.mesh_provider.reactor_hand_patch_ids(side).values():
                union.extend(ids)
            hand_unions[side] = sorted(set(union))

        for h_idx, side in enumerate(HAND_SIDES):
            hand_ids = HAND_JOINT_IDS[side]
            hand_patch = self._gather_patch(reactor_vertices, hand_unions[side])
            wrist_vel = hand_feat[:, :, h_idx, 3:6]
            wrist_xyz = hand_feat[:, :, h_idx, 0:3]

            for p_idx, part_name in enumerate(ACTOR_PART_NAMES):
                actor_ids = ACTOR_PART_JOINT_IDS[part_name]
                dist_top1, dist_topk = topk_pairwise_distance(
                    actor_xyz,
                    reactor_xyz,
                    actor_ids,
                    hand_ids,
                    self.topk,
                )
                top1[:, :, h_idx, p_idx] = dist_top1
                topk_mean[:, :, h_idx, p_idx] = dist_topk

                target_ids = []
                for ids in self.mesh_provider.actor_target_patch_ids(part_name).values():
                    target_ids.extend(ids)
                target_ids = sorted(set(target_ids))
                nontarget_ids = self.mesh_provider.actor_nontarget_patch_ids(part_name)

                target_patch = self._gather_patch(actor_vertices, target_ids)
                nontarget_patch = self._gather_patch(actor_vertices, nontarget_ids)

                mesh_target_dist[:, :, h_idx, p_idx] = _softmin_pairwise_distance(
                    hand_patch,
                    target_patch,
                    beta=self.softmin_beta,
                )
                mesh_nontarget_dist[:, :, h_idx, p_idx] = _softmin_pairwise_distance(
                    hand_patch,
                    nontarget_patch,
                    beta=self.softmin_beta,
                )
                target_to_nontarget[:, :, h_idx, p_idx] = _softmin_pairwise_distance(
                    target_patch,
                    nontarget_patch,
                    beta=self.softmin_beta,
                )

                target_center = part_feat[:, :, p_idx, 0:3]
                target_vel_vec = part_feat[:, :, p_idx, 3:6]
                vec_to_target = target_center - wrist_xyz
                target_speed[:, :, h_idx, p_idx] = torch.linalg.norm(target_vel_vec, dim=-1)
                rel_speed[:, :, h_idx, p_idx] = torch.linalg.norm(wrist_vel - target_vel_vec, dim=-1)
                approach_alignment[:, :, h_idx, p_idx] = (
                    safe_normalize(wrist_vel) * safe_normalize(vec_to_target)
                ).sum(dim=-1)

        closing_joint[:, 1:] = -(top1[:, 1:] - top1[:, :-1])
        closing_mesh[:, 1:] = -(mesh_target_dist[:, 1:] - mesh_target_dist[:, :-1])
        target_approach[:, 1:] = -(mesh_target_dist[:, 1:] - mesh_target_dist[:, :-1])

        soft_contact_target = torch.exp(-mesh_target_dist / max(self.sigma, 1e-6))
        soft_clearance = torch.exp(-mesh_nontarget_dist / max(self.sigma, 1e-6))

        return torch.stack(
            [
                top1,
                topk_mean,
                mesh_target_dist,
                mesh_nontarget_dist,
                target_to_nontarget,
                closing_joint,
                closing_mesh,
                rel_speed,
                approach_alignment,
                target_speed,
                target_approach,
                soft_contact_target,
                soft_clearance,
                shape_delta_norm,
            ],
            dim=-1,
        )

    def build(self, actor_motion, coarse_reactor_motion, lengths=None, restoration_meta=None, return_xyz=False):
        if restoration_meta is None:
            raise ValueError(
                "HandContactFeatureBuilder requires restoration_meta so proposal runs in restored pair space."
            )
        validate_restoration_metadata(restoration_meta, context="proposal feature builder")

        actor_motion, coarse_reactor_motion = restore_motion_batch(
            actor_motion,
            coarse_reactor_motion,
            restoration_meta,
        )
        actor_xyz = self.geometry.to_xyz(
            actor_motion,
            betas=restoration_meta["actor_betas"],
            gender_id=restoration_meta["actor_gender_id"],
            body_model_type=restoration_meta["body_model_type"],
            preserve_pair_space=True,
        )
        reactor_xyz = self.geometry.to_xyz(
            coarse_reactor_motion,
            betas=restoration_meta["reactor_betas"],
            gender_id=restoration_meta["reactor_gender_id"],
            body_model_type=restoration_meta["body_model_type"],
            preserve_pair_space=True,
        )
        actor_vertices = self._to_vertices(
            actor_motion,
            betas=restoration_meta["actor_betas"],
            gender_id=restoration_meta["actor_gender_id"],
            body_model_type=restoration_meta["body_model_type"],
        )
        reactor_vertices = self._to_vertices(
            coarse_reactor_motion,
            betas=restoration_meta["reactor_betas"],
            gender_id=restoration_meta["reactor_gender_id"],
            body_model_type=restoration_meta["body_model_type"],
        )

        temporal_cues = self._build_temporal_cues(
            lengths,
            restoration_meta,
            num_frames=actor_motion.shape[-1],
            device=actor_motion.device,
            dtype=actor_motion.dtype,
        )
        shape_summary = self._build_shape_summary(
            restoration_meta,
            num_frames=actor_motion.shape[-1],
            device=actor_motion.device,
            dtype=actor_motion.dtype,
        )

        hand_feat = self._build_hand_features(reactor_xyz, shape_summary, temporal_cues)
        part_feat = self._build_part_features(actor_xyz, actor_vertices, shape_summary, temporal_cues)
        rel_feat = self._build_relation_features(
            actor_xyz,
            reactor_xyz,
            actor_vertices,
            reactor_vertices,
            hand_feat,
            part_feat,
            restoration_meta,
        )

        if lengths is not None:
            frame_ids = torch.arange(hand_feat.shape[1], device=hand_feat.device).view(1, -1)
            lengths = torch.as_tensor(lengths, device=hand_feat.device, dtype=torch.long)
            mask = frame_ids < lengths.view(-1, 1)
            hand_feat = hand_feat * mask[:, :, None, None].float()
            part_feat = part_feat * mask[:, :, None, None].float()
            rel_feat = rel_feat * mask[:, :, None, None, None].float()

        if return_xyz:
            return hand_feat, part_feat, rel_feat, actor_xyz, reactor_xyz
        return hand_feat, part_feat, rel_feat
