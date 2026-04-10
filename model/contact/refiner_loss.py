import torch
import torch.nn as nn

from model.contact.contact_defs import (
    HAND_JOINT_IDS,
    WRIST_JOINT_IDS,
    FINGER_TIP_IDS,
    ACTOR_PART_JOINT_IDS,
    TARGET_PARTS,
    PHASE_IDS,
    BUFFER_JOINT_IDS,
)
from model.contact.contact_geometry import ContactGeometry, temporal_diff


def _masked_mse(diff, mask):
    if diff.numel() == 0:
        return diff.sum() * 0.0
    mask = mask.float()
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    extra = diff[0, 0].numel() if diff.dim() > 2 else 1
    denom = (mask.sum() * extra).clamp(min=1.0)
    return (diff * diff * mask).sum() / denom


def _build_phase_weight(phase_seq, weights):
    weight = torch.zeros_like(phase_seq, dtype=torch.float)
    for name, idx in PHASE_IDS.items():
        w = float(weights.get(name, 0.0))
        weight = torch.where(phase_seq == idx, torch.full_like(weight, w), weight)
    return weight


def _joint_mask_from_side(joint_ids, side):
    if side == 0:
        target_ids = set([WRIST_JOINT_IDS["left"]] + HAND_JOINT_IDS["left"] + [18])
    else:
        target_ids = set([WRIST_JOINT_IDS["right"]] + HAND_JOINT_IDS["right"] + [19])
    return torch.as_tensor([jid in target_ids for jid in joint_ids], dtype=torch.bool)


def _wrist_mask(joint_ids):
    target_ids = set([WRIST_JOINT_IDS["left"], WRIST_JOINT_IDS["right"]] + BUFFER_JOINT_IDS)
    return torch.as_tensor([jid in target_ids for jid in joint_ids], dtype=torch.bool)


def _hand_mask(joint_ids):
    target_ids = set(HAND_JOINT_IDS["left"] + HAND_JOINT_IDS["right"])
    return torch.as_tensor([jid in target_ids for jid in joint_ids], dtype=torch.bool)


class HandContactRefinerLoss(nn.Module):
    """
    Loss for HandContactRefiner.
    """

    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        phase_weights=None,
        lambda_wrist_res=1.0,
        lambda_hand_res=1.0,
        lambda_contact_align=0.5,
        lambda_smooth=0.1,
        lambda_identity=0.1,
        lambda_delta_reg=0.01,
        lambda_buffer=0.05,
    ):
        super().__init__()
        self.geometry = ContactGeometry(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device="cpu",
        )
        self.phase_weights = phase_weights or {
            "idle": 0.05,
            "approach": 0.6,
            "hold": 1.0,
            "release": 0.3,
        }
        self.lambda_wrist_res = float(lambda_wrist_res)
        self.lambda_hand_res = float(lambda_hand_res)
        self.lambda_contact_align = float(lambda_contact_align)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_identity = float(lambda_identity)
        self.lambda_delta_reg = float(lambda_delta_reg)
        self.lambda_buffer = float(lambda_buffer)

    def forward(self, refined_full, coarse_full, gt_full, actor_full, window_batch):
        device = refined_full.device
        joint_ids = window_batch["joint_ids"]
        joint_ids_t = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
        time_mask = window_batch["time_mask"].to(device)
        hand_side_idx = window_batch["hand_side_idx"].to(device)
        target_part_id = window_batch["target_part_id"].to(device)

        refined_local = refined_full.index_select(1, joint_ids_t).permute(0, 3, 1, 2)
        coarse_local = coarse_full.index_select(1, joint_ids_t).permute(0, 3, 1, 2)
        gt_local = gt_full.index_select(1, joint_ids_t).permute(0, 3, 1, 2)
        delta = refined_local - coarse_local

        wrist_mask = _wrist_mask(joint_ids).to(device)
        hand_mask = _hand_mask(joint_ids).to(device)

        wrist_loss = 0.0
        hand_loss = 0.0
        smooth_wrist = 0.0
        smooth_hand = 0.0
        delta_smooth = 0.0
        identity_loss = 0.0
        buffer_loss = 0.0

        for i in range(refined_local.shape[0]):
            side = int(hand_side_idx[i].item())
            joint_mask = _joint_mask_from_side(joint_ids, side).to(device)
            phase_seq = window_batch["cond_feat"][i, :, -4:].argmax(dim=-1)
            phase_weight = _build_phase_weight(phase_seq, self.phase_weights).to(device)
            frame_mask = time_mask[i].float() * phase_weight

            wrist_joint_mask = joint_mask & wrist_mask
            hand_joint_mask = joint_mask & hand_mask
            non_target_mask = ~joint_mask
            buffer_joint_mask = torch.as_tensor([jid in BUFFER_JOINT_IDS for jid in joint_ids], device=device)

            if wrist_joint_mask.any():
                wrist_idx = torch.nonzero(wrist_joint_mask, as_tuple=False).flatten()
                wrist_loss = wrist_loss + _masked_mse(
                    refined_local[i, :, wrist_idx, :] - gt_local[i, :, wrist_idx, :],
                    frame_mask,
                )
                smooth_wrist = smooth_wrist + _masked_mse(
                    temporal_diff(refined_local[i, :, wrist_idx, :]),
                    frame_mask,
                )

            if hand_joint_mask.any():
                hand_idx = torch.nonzero(hand_joint_mask, as_tuple=False).flatten()
                hand_loss = hand_loss + _masked_mse(
                    refined_local[i, :, hand_idx, :] - gt_local[i, :, hand_idx, :],
                    frame_mask,
                )
                smooth_hand = smooth_hand + _masked_mse(
                    temporal_diff(refined_local[i, :, hand_idx, :]),
                    frame_mask,
                )

            delta_smooth = delta_smooth + _masked_mse(
                temporal_diff(delta[i]),
                frame_mask,
            )

            if non_target_mask.any():
                non_idx = torch.nonzero(non_target_mask, as_tuple=False).flatten()
                identity_loss = identity_loss + _masked_mse(
                    delta[i, :, non_idx, :],
                    time_mask[i].float(),
                )

            if buffer_joint_mask.any():
                buffer_idx = torch.nonzero(buffer_joint_mask, as_tuple=False).flatten()
                buffer_loss = buffer_loss + _masked_mse(
                    delta[i, :, buffer_idx, :],
                    time_mask[i].float(),
                )

        num_windows = max(refined_local.shape[0], 1)
        wrist_loss = wrist_loss / num_windows
        hand_loss = hand_loss / num_windows
        smooth_wrist = smooth_wrist / num_windows
        smooth_hand = smooth_hand / num_windows
        delta_smooth = delta_smooth / num_windows
        identity_loss = identity_loss / num_windows
        buffer_loss = buffer_loss / num_windows

        contact_align = self._contact_alignment_loss(
            refined_full, actor_full, hand_side_idx, target_part_id, time_mask, window_batch["cond_feat"]
        )

        total = (
            self.lambda_wrist_res * wrist_loss
            + self.lambda_hand_res * hand_loss
            + self.lambda_contact_align * contact_align
            + self.lambda_smooth * (smooth_wrist + smooth_hand + delta_smooth)
            + self.lambda_identity * identity_loss
            + self.lambda_delta_reg * _masked_mse(delta, time_mask)
            + self.lambda_buffer * buffer_loss
        )

        return total, {
            "loss_wrist_res": wrist_loss,
            "loss_hand_res": hand_loss,
            "loss_contact_align": contact_align,
            "loss_smooth": smooth_wrist + smooth_hand + delta_smooth,
            "loss_identity": identity_loss,
            "loss_buffer": buffer_loss,
            "loss_total": total,
        }

    def _contact_alignment_loss(self, refined_full, actor_full, hand_side_idx, target_part_id, time_mask, cond_feat):
        device = refined_full.device
        self.geometry._ensure_device(device)
        actor_xyz = self.geometry.to_xyz(actor_full)
        refined_xyz = self.geometry.to_xyz(refined_full)

        total = 0.0
        count = 0
        for i in range(refined_full.shape[0]):
            side = int(hand_side_idx[i].item())
            target_id = int(target_part_id[i].item())
            if target_id == 0:
                continue
            target_name = TARGET_PARTS[target_id]
            patch_ids = ACTOR_PART_JOINT_IDS.get(target_name, [])
            if not patch_ids:
                continue

            patch_ids_t = torch.as_tensor(patch_ids, device=device, dtype=torch.long)
            patch_xyz = actor_xyz[i].index_select(0, patch_ids_t).permute(2, 0, 1)
            patch_center = patch_xyz.mean(dim=1)

            wrist_id = WRIST_JOINT_IDS["left"] if side == 0 else WRIST_JOINT_IDS["right"]
            wrist_xyz = refined_xyz[i, wrist_id].permute(1, 0)

            hand_ids = HAND_JOINT_IDS["left"] if side == 0 else HAND_JOINT_IDS["right"]
            hand_ids_t = torch.as_tensor(hand_ids, device=device, dtype=torch.long)
            hand_xyz = refined_xyz[i].index_select(0, hand_ids_t).permute(2, 0, 1)
            hand_center = hand_xyz.mean(dim=1)

            tip_ids = FINGER_TIP_IDS["left"] if side == 0 else FINGER_TIP_IDS["right"]
            tip_ids_t = torch.as_tensor(tip_ids, device=device, dtype=torch.long)
            tip_xyz = refined_xyz[i].index_select(0, tip_ids_t).permute(2, 0, 1)
            tip_center = tip_xyz.mean(dim=1)

            wrist_dist = torch.linalg.norm(wrist_xyz - patch_center, dim=-1)
            hand_dist = torch.linalg.norm(hand_center - patch_center, dim=-1)
            tip_dist = torch.linalg.norm(tip_center - patch_center, dim=-1)

            phase_seq = cond_feat[i, :, -4:].argmax(dim=-1)
            phase_weight = _build_phase_weight(phase_seq, self.phase_weights).to(device)
            dist = (wrist_dist + hand_dist + tip_dist) / 3.0
            weight = time_mask[i].float() * phase_weight
            total = total + (dist * weight).sum() / weight.sum().clamp(min=1.0)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)
        return total / count
