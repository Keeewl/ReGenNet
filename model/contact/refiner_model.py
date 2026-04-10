import torch
import torch.nn as nn

from model.contact.contact_defs import (
    HAND_JOINT_IDS,
    WRIST_JOINT_IDS,
    HAND_SIDES,
    default_refiner_joint_ids,
)
from model.contact.contact_geometry import build_time_mask


class TemporalConvBlock(nn.Module):
    """
    Lightweight temporal block per joint.

    Inputs:
        x: [B, T, J, H]
    Outputs:
        out: [B, T, J, H]
    """

    def __init__(self, hidden_dim, dropout=0.1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, groups=hidden_dim
        )
        self.pw = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, num_frames, num_joints, hidden = x.shape
        y = x.permute(0, 2, 3, 1).reshape(bsz * num_joints, hidden, num_frames)
        y = self.pw(self.dw(y))
        y = self.act(y)
        y = self.dropout(y)
        y = y.reshape(bsz, num_joints, hidden, num_frames).permute(0, 3, 1, 2)
        return x + y


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


class HandContactRefiner(nn.Module):
    """
    Hand Contact Refinement (HCR) refiner.

    Inputs:
        coarse_motion: [B, J, 6, T]
        proposal_active: [B, T, 2] or None
    Outputs:
        refined_motion: [B, J, 6, T]
    """

    def __init__(
        self,
        input_dim=6,
        hidden_dim=128,
        num_temporal_blocks=2,
        dropout=0.1,
        joint_ids=None,
        use_gate=False,
        gate_init_bias=-1.0,
    ):
        super().__init__()
        self.joint_ids = joint_ids or default_refiner_joint_ids()
        self.use_gate = bool(use_gate)

        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_blocks = nn.ModuleList(
            [TemporalConvBlock(hidden_dim, dropout=dropout) for _ in range(num_temporal_blocks)]
        )
        self.delta_head = nn.Linear(hidden_dim, input_dim)
        self.gate_head = nn.Linear(hidden_dim, 1) if self.use_gate else None
        if self.gate_head is not None:
            nn.init.constant_(self.gate_head.bias, float(gate_init_bias))

        self.register_buffer("joint_to_hand", _joint_to_hand_index(self.joint_ids))

    def _apply_active_gate(self, delta, proposal_active):
        if proposal_active is None:
            return delta
        hand_idx = self.joint_to_hand.to(proposal_active.device)
        active_joint = proposal_active.index_select(2, hand_idx)
        return delta * active_joint.unsqueeze(-1).float()

    def forward(self, coarse_motion, proposal_active=None, lengths=None, return_aux=True):
        if coarse_motion.dim() != 4:
            raise ValueError("coarse_motion must be [B, J, 6, T]")
        device = coarse_motion.device
        batch_size, num_joints, _, num_frames = coarse_motion.shape
        joint_ids = torch.as_tensor(self.joint_ids, device=device, dtype=torch.long)

        coarse_local = coarse_motion.index_select(1, joint_ids).permute(0, 3, 1, 2)
        x = self.embed(coarse_local)
        for block in self.temporal_blocks:
            x = block(x)
        delta = self.delta_head(x)

        gate = None
        if self.gate_head is not None:
            gate = torch.sigmoid(self.gate_head(x))
            delta = delta * gate

        delta = self._apply_active_gate(delta, proposal_active)

        if lengths is not None:
            mask = build_time_mask(lengths, num_frames, device=device)
            if mask is not None:
                delta = delta * mask[:, :, None, None].float()

        delta_full = torch.zeros_like(coarse_motion)
        delta_full.index_copy_(1, joint_ids, delta.permute(0, 2, 3, 1))
        refined = coarse_motion + delta_full

        if return_aux:
            return refined, {
                "delta": delta,
                "delta_full": delta_full,
                "gate": gate,
                "joint_ids": joint_ids,
            }
        return refined
