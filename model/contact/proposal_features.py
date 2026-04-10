import torch

from model.contact.contact_defs import (
    HAND_SIDES,
    WRIST_JOINT_IDS,
    HAND_JOINT_IDS,
    FINGER_BASE_IDS,
    FINGER_TIP_IDS,
    ACTOR_PART_NAMES,
    ACTOR_PART_JOINT_IDS,
)
from model.contact.contact_geometry import ContactGeometry, temporal_diff, safe_normalize, topk_pairwise_distance


class HandContactFeatureBuilder:
    """
    Build hand-level, actor-part, and relation features for contact proposal.
    """

    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        topk=3,
        sigma=0.1,
        device="cpu",
    ):
        self.geometry = ContactGeometry(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device=device,
        )
        self.topk = int(topk)
        self.sigma = float(sigma)

    def _build_hand_features(self, reactor_xyz):
        """
        reactor_xyz: [B, J, 3, T]
        returns hand_feat: [B, T, 2, Fh]
        """
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

            palm_ids = torch.as_tensor(
                [wrist_id] + FINGER_BASE_IDS[side], device=device, dtype=torch.long
            )
            palm_pos = reactor_xyz.index_select(1, palm_ids).permute(0, 3, 1, 2)
            palm_center = palm_pos.mean(dim=2)

            tip_ids = torch.as_tensor(FINGER_TIP_IDS[side], device=device, dtype=torch.long)
            tip_pos = reactor_xyz.index_select(1, tip_ids).permute(0, 3, 1, 2)
            tip_center = tip_pos.mean(dim=2)

            hand_dir = safe_normalize(hand_center - wrist_xyz)
            tip_dir = safe_normalize(tip_center - wrist_xyz)

            tip_to_palm = torch.linalg.norm(tip_pos - palm_center[:, :, None, :], dim=-1)
            openness = tip_to_palm.mean(dim=-1, keepdim=True)

            pairwise = torch.linalg.norm(
                tip_pos[:, :, :, None, :] - tip_pos[:, :, None, :, :], dim=-1
            )
            eye = torch.eye(tip_pos.shape[2], device=device, dtype=torch.bool)
            spread = pairwise.masked_fill(eye, 0.0).sum(dim=(-1, -2)) / float(tip_pos.shape[2] * (tip_pos.shape[2] - 1))
            spread = spread.unsqueeze(-1)

            base_ids = torch.as_tensor(FINGER_BASE_IDS[side], device=device, dtype=torch.long)
            base_pos = reactor_xyz.index_select(1, base_ids).permute(0, 3, 1, 2)
            base_dist = torch.linalg.norm(base_pos - wrist_xyz[:, :, None, :], dim=-1)
            tip_dist_wrist = torch.linalg.norm(tip_pos - wrist_xyz[:, :, None, :], dim=-1)
            flexion = ((base_dist - tip_dist_wrist) / base_dist.clamp(min=1e-6)).mean(dim=-1, keepdim=True)

            tip_var = tip_pos.var(dim=2, unbiased=False)
            tip_disp = torch.sqrt(torch.sum(tip_var, dim=-1, keepdim=True))

            hand_rel = hand_pos - wrist_xyz[:, :, None, :]
            hand_mean = hand_rel.mean(dim=2)
            hand_std = hand_rel.std(dim=2, unbiased=False)

            feat = torch.cat(
                [
                    wrist_xyz,
                    wrist_vel,
                    wrist_acc,
                    hand_center,
                    palm_center,
                    hand_dir,
                    tip_dir,
                    openness,
                    spread,
                    flexion,
                    tip_disp,
                    hand_mean,
                    hand_std,
                ],
                dim=-1,
            )
            hand_features.append(feat)

        hand_feat = torch.stack(hand_features, dim=2)
        return hand_feat

    def _build_part_features(self, actor_xyz):
        """
        actor_xyz: [B, J, 3, T]
        returns part_feat: [B, T, 5, Fp]
        """
        part_features = []
        device = actor_xyz.device
        for part_name in ACTOR_PART_NAMES:
            part_ids = torch.as_tensor(ACTOR_PART_JOINT_IDS[part_name], device=device, dtype=torch.long)
            part_pos = actor_xyz.index_select(1, part_ids).permute(0, 3, 1, 2)
            part_center = part_pos.mean(dim=2)
            part_vel = temporal_diff(part_center)
            spread = torch.linalg.norm(part_pos - part_center[:, :, None, :], dim=-1).mean(dim=2, keepdim=True)
            part_rel = part_pos - part_center[:, :, None, :]
            part_mean = part_rel.mean(dim=2)
            part_std = part_rel.std(dim=2, unbiased=False)
            feat = torch.cat([part_center, part_vel, spread, part_mean, part_std], dim=-1)
            part_features.append(feat)
        part_feat = torch.stack(part_features, dim=2)
        return part_feat

    def _build_relation_features(self, actor_xyz, reactor_xyz, hand_feat, part_feat):
        """
        hand_feat: [B, T, 2, Fh]
        part_feat: [B, T, 5, Fp]
        returns rel_feat: [B, T, 2, 5, 8]
        """
        batch_size, num_frames, _, _ = hand_feat.shape
        device = actor_xyz.device
        top1 = torch.zeros(batch_size, num_frames, 2, 5, device=device, dtype=actor_xyz.dtype)
        topk_mean = torch.zeros_like(top1)
        for h_idx, side in enumerate(HAND_SIDES):
            hand_ids = HAND_JOINT_IDS[side]
            for p_idx, part_name in enumerate(ACTOR_PART_NAMES):
                actor_ids = ACTOR_PART_JOINT_IDS[part_name]
                dist_top1, dist_topk = topk_pairwise_distance(
                    actor_xyz, reactor_xyz, actor_ids, hand_ids, self.topk
                )
                top1[:, :, h_idx, p_idx] = dist_top1
                topk_mean[:, :, h_idx, p_idx] = dist_topk

        margin = topk_mean - top1
        delta = top1[:, 1:] - top1[:, :-1]
        closing = torch.cat([delta[:, :1] * 0.0, -delta], dim=1)

        wrist_vel = hand_feat[:, :, :, 3:6]
        part_vel = part_feat[:, :, :, 3:6]
        rel_speed = torch.linalg.norm(wrist_vel[:, :, :, None, :] - part_vel[:, :, None, :, :], dim=-1)

        soft_contact_top1 = torch.exp(-top1 / max(self.sigma, 1e-6))
        soft_contact_topk = torch.exp(-topk_mean / max(self.sigma, 1e-6))

        wrist_xyz = hand_feat[:, :, :, 0:3]
        part_center = part_feat[:, :, :, 0:3]
        vec_to_part = part_center[:, :, None, :, :] - wrist_xyz[:, :, :, None, :]
        vel_norm = safe_normalize(wrist_vel[:, :, :, None, :])
        dir_norm = safe_normalize(vec_to_part)
        alignment = (vel_norm * dir_norm).sum(dim=-1)
        vel_mag = torch.linalg.norm(wrist_vel, dim=-1, keepdim=True)
        alignment = torch.where(vel_mag > 1e-6, alignment, torch.zeros_like(alignment))

        rel_feat = torch.stack(
            [
                top1,
                topk_mean,
                margin,
                closing,
                rel_speed,
                soft_contact_top1,
                soft_contact_topk,
                alignment,
            ],
            dim=-1,
        )
        return rel_feat

    def build(self, actor_motion, coarse_reactor_motion, lengths=None, return_xyz=False):
        """
        actor_motion/coarse_reactor_motion: [B, J, 6, T]
        returns:
            hand_feat: [B, T, 2, Fh_raw]
            part_feat: [B, T, 5, Fp_raw]
            rel_feat: [B, T, 2, 5, 8]
        """
        actor_xyz = self.geometry.to_xyz(actor_motion)
        reactor_xyz = self.geometry.to_xyz(coarse_reactor_motion)

        hand_feat = self._build_hand_features(reactor_xyz)
        part_feat = self._build_part_features(actor_xyz)
        rel_feat = self._build_relation_features(actor_xyz, reactor_xyz, hand_feat, part_feat)

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
