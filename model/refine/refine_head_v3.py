import torch
import torch.nn as nn

from model.refine.active_window import PART_JOINT_IDS
from model.refine.refine_head_v2 import (
    JointFeatureEmbed,
    TemporalConvBlock,
    PartPoolingBlock,
    PartInteractionBlock,
    PartFusionBlock,
)


class RNetV3Lite(nn.Module):
    """
    Lightweight v3 refine head (joint embed + temporal + part interaction).

    Inputs:
        x_coarse: [B, T, J, 6]
        geom_feat: [B, T, J, F]
    Outputs:
        delta_raw: [B, T, J, 6]
        gate_logits: [B, T, J, 1]
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=6,
        num_temporal_blocks=2,
        dropout=0.1,
        refine_joint_ids=None,
        part_names=None,
        part_joint_ids=None,
        gate_init_bias=-2.0,
    ):
        super().__init__()
        if refine_joint_ids is None:
            raise ValueError("refine_joint_ids is required for RNetV3Lite")

        if part_names is None:
            part_names = ["left_arm", "right_arm", "left_hand", "right_hand", "coord"]

        if part_joint_ids is None:
            part_joint_ids = {
                "left_arm": PART_JOINT_IDS["left_arm"],
                "right_arm": PART_JOINT_IDS["right_arm"],
                "left_hand": PART_JOINT_IDS["left_hand"],
                "right_hand": PART_JOINT_IDS["right_hand"],
                "coord": [12, 15],
            }

        part_joint_indices = []
        joint_to_part = [-1 for _ in refine_joint_ids]
        for part_idx, name in enumerate(part_names):
            ids = set(part_joint_ids.get(name, []))
            indices = [i for i, jid in enumerate(refine_joint_ids) if jid in ids]
            part_joint_indices.append(indices)
            for idx in indices:
                joint_to_part[idx] = part_idx

        if any(idx < 0 for idx in joint_to_part):
            raise ValueError("refine_joint_ids contain joints not covered by part_joint_ids")

        self.embed = JointFeatureEmbed(input_dim, hidden_dim, dropout=dropout)
        self.temporal_blocks = nn.ModuleList(
            [TemporalConvBlock(hidden_dim, dropout=dropout) for _ in range(num_temporal_blocks)]
        )
        self.part_pool = PartPoolingBlock(part_joint_indices)
        self.part_interact = PartInteractionBlock(hidden_dim, dropout=dropout)
        self.part_fuse = PartFusionBlock(hidden_dim, torch.as_tensor(joint_to_part, dtype=torch.long))
        self.delta_head = nn.Linear(hidden_dim, output_dim)
        self.gate_head = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.gate_head.bias, float(gate_init_bias))

    def forward(self, x_coarse, geom_feat):
        """
        x_coarse: [B, T, J, 6]
        geom_feat: [B, T, J, F]
        returns delta_raw: [B, T, J, 6], gate_logits: [B, T, J, 1]
        """
        if x_coarse.dim() != 4 or geom_feat.dim() != 4:
            raise ValueError("x_coarse and geom_feat must be 4D tensors")
        feat = torch.cat([x_coarse, geom_feat], dim=-1)
        x = self.embed(feat)
        for block in self.temporal_blocks:
            x = block(x)
        part_feat = self.part_pool(x)
        part_feat = self.part_interact(part_feat)
        x = self.part_fuse(x, part_feat)
        delta_raw = self.delta_head(x)
        gate_logits = self.gate_head(x)
        return delta_raw, gate_logits
