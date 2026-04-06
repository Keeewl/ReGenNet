import math
import torch
import torch.nn as nn

from model.refine.active_window import PART_JOINT_IDS


class JointFeatureEmbed(nn.Module):
    """
    Joint-wise feature embedding.

    Inputs:
        x: [B, T, J, F]
    Outputs:
        out: [B, T, J, H]
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)


class TemporalConvBlock(nn.Module):
    """
    Lightweight temporal modeling per joint with depthwise conv.

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


class PartPoolingBlock(nn.Module):
    """
    Pool joint features into part-level tokens.

    Inputs:
        x: [B, T, J, H]
    Outputs:
        part_feat: [B, T, P, H]
    """

    def __init__(self, part_joint_indices):
        super().__init__()
        self.part_joint_indices = part_joint_indices

    def forward(self, x):
        bsz, num_frames, _, hidden = x.shape
        parts = []
        for indices in self.part_joint_indices:
            if not indices:
                part_feat = torch.zeros(bsz, num_frames, 1, hidden, device=x.device, dtype=x.dtype)
            else:
                idx = torch.as_tensor(indices, device=x.device, dtype=torch.long)
                part_feat = x.index_select(2, idx).mean(dim=2, keepdim=True)
            parts.append(part_feat)
        return torch.cat(parts, dim=2)


class PartInteractionBlock(nn.Module):
    """
    Sparse part interaction with light attention and MLP.

    Inputs:
        part_feat: [B, T, P, H]
    Outputs:
        out: [B, T, P, H]
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, part_feat):
        hidden = part_feat.shape[-1]
        q = self.q(part_feat)
        k = self.k(part_feat)
        v = self.v(part_feat)
        scores = torch.einsum("btph,btqh->btpq", q, k) / math.sqrt(hidden)
        attn = torch.softmax(scores, dim=-1)
        mix = torch.einsum("btpq,btqh->btph", attn, v)
        mix = self.dropout(self.proj(mix))
        out = part_feat + mix
        out = out + self.dropout(self.mlp(out))
        return out


class PartFusionBlock(nn.Module):
    """
    Fuse part features back to joints.

    Inputs:
        joint_feat: [B, T, J, H]
        part_feat: [B, T, P, H]
    Outputs:
        out: [B, T, J, H]
    """

    def __init__(self, hidden_dim, joint_to_part):
        super().__init__()
        self.register_buffer("joint_to_part", joint_to_part)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, joint_feat, part_feat):
        joint_to_part = self.joint_to_part.to(joint_feat.device)
        part_context = part_feat.index_select(2, joint_to_part)
        return joint_feat + self.proj(part_context)


class RNetV2Lite(nn.Module):
    """
    Lightweight v2 refine head (joint embed + temporal + part interaction).

    Inputs:
        x_coarse: [B, T, J, 6]
        geom_feat: [B, T, J, F]
    Outputs:
        delta: [B, T, J, 6]
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
    ):
        super().__init__()
        if refine_joint_ids is None:
            raise ValueError("refine_joint_ids is required for RNetV2Lite")

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
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_coarse, geom_feat):
        """
        x_coarse: [B, T, J, 6]
        geom_feat: [B, T, J, F]
        returns delta: [B, T, J, 6]
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
        return self.out(x)
