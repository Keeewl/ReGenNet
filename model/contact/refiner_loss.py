import torch
import torch.nn as nn

from model.contact.contact_defs import (
    HAND_JOINT_IDS,
    WRIST_JOINT_IDS,
    default_refiner_joint_ids,
)
from model.contact.contact_geometry import build_time_mask


def _masked_mse(diff, mask):
    if diff.numel() == 0:
        return diff.sum() * 0.0
    mask = mask.float()
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    extra = diff[0, 0].numel() if diff.dim() > 2 else 1
    denom = (mask.sum() * extra).clamp(min=1.0)
    return (diff * diff * mask).sum() / denom


def _joint_to_hand_index(joint_ids):
    left = set(HAND_JOINT_IDS["left"] + [WRIST_JOINT_IDS["left"]])
    right = set(HAND_JOINT_IDS["right"] + [WRIST_JOINT_IDS["right"]])
    mapping = []
    for jid in joint_ids:
        if jid in left:
            mapping.append(0)
        elif jid in right:
            mapping.append(1)
        else:
            mapping.append(0)
    return torch.as_tensor(mapping, dtype=torch.long)


class HandContactRefinerLoss(nn.Module):
    """
    Baseline loss for HCR refiner.
    """

    def __init__(self, joint_ids=None):
        super().__init__()
        self.joint_ids = joint_ids or default_refiner_joint_ids()
        self.register_buffer("joint_to_hand", _joint_to_hand_index(self.joint_ids))

    def forward(self, refined_motion, gt_motion, lengths=None, active_mask=None):
        """
        refined_motion/gt_motion: [B, J, 6, T]
        active_mask: [B, T, 2] or None
        """
        device = refined_motion.device
        joint_ids = torch.as_tensor(self.joint_ids, device=device, dtype=torch.long)
        refined_local = refined_motion.index_select(1, joint_ids).permute(0, 3, 1, 2)
        gt_local = gt_motion.index_select(1, joint_ids).permute(0, 3, 1, 2)
        diff = refined_local - gt_local

        num_frames = refined_local.shape[1]
        time_mask = build_time_mask(lengths, num_frames, device=device)
        if time_mask is None:
            mask = torch.ones(refined_local.shape[0], num_frames, device=device)
        else:
            mask = time_mask.float()

        if active_mask is not None:
            hand_idx = self.joint_to_hand.to(active_mask.device)
            active_joint = active_mask.index_select(2, hand_idx)
            mask = mask[:, :, None] * active_joint.float()
        else:
            mask = mask[:, :, None]

        loss = _masked_mse(diff, mask)
        return loss, {"loss_refiner": loss}
