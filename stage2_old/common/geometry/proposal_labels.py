import torch

from stage2_old.common.geometry.contact_defs import (
    HAND_SIDES,
    HAND_JOINT_IDS,
    TARGET_PARTS,
    ACTOR_PART_JOINT_IDS,
    PHASE_IDS,
)
from stage2_old.common.geometry.contact_geometry import ContactGeometry, build_time_mask, topk_pairwise_distance


def _compute_hand_part_distances(actor_xyz, reactor_xyz, topk=3):
    """
    returns top1/topk_mean: [B, T, 2, 5]
    """
    batch_size, _, _, num_frames = actor_xyz.shape
    device = actor_xyz.device
    top1 = torch.zeros(batch_size, num_frames, 2, 5, device=device, dtype=actor_xyz.dtype)
    topk_mean = torch.zeros_like(top1)
    for h_idx, side in enumerate(HAND_SIDES):
        hand_ids = HAND_JOINT_IDS[side]
        for p_idx, part_name in enumerate(TARGET_PARTS[1:]):
            actor_ids = ACTOR_PART_JOINT_IDS[part_name]
            dist_top1, dist_topk = topk_pairwise_distance(
                actor_xyz, reactor_xyz, actor_ids, hand_ids, topk
            )
            top1[:, :, h_idx, p_idx] = dist_top1
            topk_mean[:, :, h_idx, p_idx] = dist_topk
    return top1, topk_mean


def _apply_hysteresis(dist_top1, tau_near, delta_target):
    """
    dist_top1: [B, T, 2, 5]
    returns target_part: [B, T, 2] (0 = none)
    """
    batch_size, num_frames, num_hands, _ = dist_top1.shape
    target = torch.zeros(batch_size, num_frames, num_hands, device=dist_top1.device, dtype=torch.long)
    for b in range(batch_size):
        for h in range(num_hands):
            prev = 0
            for t in range(num_frames):
                dist = dist_top1[b, t, h]
                min_dist, min_idx = dist.min(dim=-1)
                if min_dist >= float(tau_near):
                    best = 0
                else:
                    best = int(min_idx.item()) + 1
                    if prev > 0:
                        prev_idx = prev - 1
                        prev_dist = dist[prev_idx]
                        if prev_dist <= min_dist + float(delta_target):
                            best = prev
                target[b, t, h] = best
                prev = best
    return target


def _build_phase_labels(dist_min, band, epsilon_move, epsilon_hold, recent_window):
    """
    dist_min: [B, T, 2]
    band: [B, T, 2]
    returns phase: [B, T, 2]
    """
    batch_size, num_frames, num_hands = dist_min.shape
    phase = torch.zeros_like(band)
    for b in range(batch_size):
        for h in range(num_hands):
            for t in range(num_frames):
                delta = 0.0
                if t > 0:
                    delta = float(dist_min[b, t, h] - dist_min[b, t - 1, h])
                start = max(0, t - int(recent_window) + 1)
                recent_band = band[b, start : t + 1, h]
                recent_contact = (recent_band == 2).any().item()
                recent_near = (recent_band >= 1).any().item()

                band_id = int(band[b, t, h])
                if band_id == 2:
                    phase_id = PHASE_IDS["hold"]
                elif band_id == 1 and abs(delta) <= float(epsilon_hold) and recent_contact:
                    phase_id = PHASE_IDS["hold"]
                elif band_id in (0, 1) and delta > float(epsilon_move) and recent_contact:
                    phase_id = PHASE_IDS["release"]
                elif band_id in (0, 1) and delta < -float(epsilon_move):
                    phase_id = PHASE_IDS["approach"]
                elif band_id == 0 and not recent_near:
                    phase_id = PHASE_IDS["idle"]
                else:
                    phase_id = PHASE_IDS["approach"] if band_id in (0, 1) else PHASE_IDS["idle"]
                phase[b, t, h] = phase_id
    return phase


class HandContactLabelBuilder:
    """
    Build frame-wise contact proposal labels from actor/gt reactor motions.
    """

    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        tau_contact=0.10,
        tau_near=0.18,
        delta_target=0.02,
        epsilon_move=0.01,
        epsilon_hold=0.005,
        recent_window=3,
        topk=3,
        device="cpu",
    ):
        self.geometry = ContactGeometry(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device=device,
        )
        self.tau_contact = float(tau_contact)
        self.tau_near = float(tau_near)
        self.delta_target = float(delta_target)
        self.epsilon_move = float(epsilon_move)
        self.epsilon_hold = float(epsilon_hold)
        self.recent_window = int(recent_window)
        self.topk = int(topk)

    def build(
        self,
        actor_motion,
        gt_reactor_motion,
        lengths=None,
        return_aux=False,
        actor_betas=None,
        reactor_betas=None,
        actor_gender_id=None,
        reactor_gender_id=None,
        body_model_type=None,
        preserve_pair_space=False,
    ):
        """
        actor_motion/gt_reactor_motion: [B, J, 6, T]
        returns labels dict with:
            active: [B, T, 2]
            target_part: [B, T, 2]
            band: [B, T, 2]
            phase: [B, T, 2]
        """
        actor_xyz = self.geometry.to_xyz(
            actor_motion,
            betas=actor_betas,
            gender_id=actor_gender_id,
            body_model_type=body_model_type,
            preserve_pair_space=preserve_pair_space,
        )
        reactor_xyz = self.geometry.to_xyz(
            gt_reactor_motion,
            betas=reactor_betas,
            gender_id=reactor_gender_id,
            body_model_type=body_model_type,
            preserve_pair_space=preserve_pair_space,
        )

        top1, topk_mean = _compute_hand_part_distances(
            actor_xyz, reactor_xyz, topk=self.topk
        )
        dist_min, _ = top1.min(dim=-1)

        target_part = _apply_hysteresis(top1, self.tau_near, self.delta_target)

        band = torch.zeros_like(target_part)
        band = torch.where(dist_min < self.tau_near, torch.ones_like(band), band)
        band = torch.where(dist_min < self.tau_contact, torch.full_like(band, 2), band)

        phase = _build_phase_labels(
            dist_min,
            band,
            epsilon_move=self.epsilon_move,
            epsilon_hold=self.epsilon_hold,
            recent_window=self.recent_window,
        )

        active = (phase != PHASE_IDS["idle"]).float()

        if lengths is not None:
            mask = build_time_mask(lengths, dist_min.shape[1], device=dist_min.device)
            if mask is not None:
                mask = mask[:, :, None]
                active = active * mask.float()
                target_part = torch.where(mask, target_part, torch.zeros_like(target_part))
                band = torch.where(mask, band, torch.zeros_like(band))
                phase = torch.where(mask, phase, torch.zeros_like(phase))

        labels = {
            "active": active,
            "target_part": target_part,
            "band": band,
            "phase": phase,
        }
        if return_aux:
            labels["dist_min"] = dist_min
            labels["dist_top1"] = top1
            labels["dist_topk_mean"] = topk_mean
        return labels
