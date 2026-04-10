import math
import torch
import torch.nn as nn

from model.contact.contact_defs import (
    HAND_JOINT_IDS,
    WRIST_JOINT_IDS,
    BUFFER_JOINT_IDS,
    default_refiner_joint_ids,
)


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class TemporalSelfAttentionBlock(nn.Module):
    """
    Temporal self-attention per token.

    Inputs:
        x: [B, T, N, D]
        time_mask: [B, T] or None
    Outputs:
        out: [B, T, N, D]
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, time_mask=None):
        bsz, num_frames, num_tokens, hidden = x.shape
        y = self.ln1(x).permute(0, 2, 1, 3).reshape(bsz * num_tokens, num_frames, hidden)
        key_padding_mask = None
        if time_mask is not None:
            key_padding_mask = (~time_mask).repeat_interleave(num_tokens, dim=0)
        attn_out, _ = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        attn_out = attn_out.reshape(bsz, num_tokens, num_frames, hidden).permute(0, 2, 1, 3)
        x = x + attn_out
        z = self.ln2(x).permute(0, 2, 1, 3).reshape(bsz * num_tokens, num_frames, hidden)
        mlp_out = self.mlp(z)
        mlp_out = mlp_out.reshape(bsz, num_tokens, num_frames, hidden).permute(0, 2, 1, 3)
        return x + mlp_out


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention from reactor tokens to actor patch tokens.

    Inputs:
        reactor: [B, T, J, D]
        actor_tokens: [B, T, P, D]
        actor_mask: [B, T, P] or None
    Outputs:
        out: [B, T, J, D]
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, reactor, actor_tokens, actor_mask=None):
        bsz, num_frames, num_joints, hidden = reactor.shape
        q = reactor.reshape(bsz * num_frames, num_joints, hidden)
        kv = actor_tokens.reshape(bsz * num_frames, actor_tokens.shape[2], hidden)
        key_padding_mask = None
        if actor_mask is not None:
            key_padding_mask = (~actor_mask).reshape(bsz * num_frames, actor_mask.shape[2])
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        attn_out = attn_out.reshape(bsz, num_frames, num_joints, hidden)
        out = reactor + attn_out
        out = out + self.mlp(self.ln(out))
        return out


class SpatialSelfAttentionBlock(nn.Module):
    """
    Spatial self-attention across joint tokens per frame.

    Inputs:
        x: [B, T, J, D]
    Outputs:
        out: [B, T, J, D]
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        bsz, num_frames, num_joints, hidden = x.shape
        q = x.reshape(bsz * num_frames, num_joints, hidden)
        attn_out, _ = self.attn(q, q, q, need_weights=False)
        attn_out = attn_out.reshape(bsz, num_frames, num_joints, hidden)
        out = x + attn_out
        out = out + self.mlp(self.ln(out))
        return out


class HandContactRefiner(nn.Module):
    """
    Proposal-conditioned hand contact refiner.

    Inputs:
        coarse_local: [B, T, J, 6]
        actor_patch: [B, T, P, Fa]
        relation_feat: [B, T, 8]
        cond_feat: [B, T, C]
        time_mask: [B, T] or None
        actor_patch_mask: [B, T, P] or None
    Outputs:
        delta_local: [B, T, J, 6]
    """

    def __init__(
        self,
        joint_ids=None,
        hidden_dim=128,
        num_temporal_blocks=2,
        num_cross_blocks=2,
        num_spatial_blocks=1,
        dropout=0.1,
        delta_max=0.15,
        use_spatial_attn=True,
    ):
        super().__init__()
        self.joint_ids = joint_ids or default_refiner_joint_ids(include_buffer=True)
        self.hidden_dim = hidden_dim
        self.delta_max = float(delta_max)
        self.use_spatial_attn = bool(use_spatial_attn)

        self.reactor_encoder = FeatureEncoder(6, hidden_dim, dropout=dropout)
        self.actor_encoder = FeatureEncoder(9, hidden_dim, dropout=dropout)
        self.relation_encoder = FeatureEncoder(8, hidden_dim, dropout=dropout)
        self.cond_encoder = FeatureEncoder(15, hidden_dim, dropout=dropout)

        self.temporal_blocks = nn.ModuleList(
            [TemporalSelfAttentionBlock(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_temporal_blocks)]
        )
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_cross_blocks)]
        )
        self.spatial_blocks = nn.ModuleList(
            [SpatialSelfAttentionBlock(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_spatial_blocks)]
        )

        self.wrist_head = nn.Linear(hidden_dim, 6)
        self.finger_head = nn.Linear(hidden_dim, 6)

        wrist_ids = {WRIST_JOINT_IDS["left"], WRIST_JOINT_IDS["right"]}
        buffer_ids = set(BUFFER_JOINT_IDS)
        self.register_buffer(
            "wrist_mask",
            torch.as_tensor([jid in wrist_ids or jid in buffer_ids for jid in self.joint_ids], dtype=torch.bool),
        )

    def _bounded_delta(self, delta):
        return self.delta_max * torch.tanh(delta / max(self.delta_max, 1e-6))

    def forward(self, coarse_local, actor_patch, relation_feat, cond_feat, time_mask=None, actor_patch_mask=None):
        if coarse_local.dim() != 4:
            raise ValueError("coarse_local must be [B,T,J,6]")

        reactor = self.reactor_encoder(coarse_local)
        cond = self.cond_encoder(cond_feat)
        reactor = reactor + cond.unsqueeze(2)

        actor_tokens = self.actor_encoder(actor_patch)
        rel_token = self.relation_encoder(relation_feat).unsqueeze(2)
        actor_tokens = torch.cat([actor_tokens, rel_token], dim=2)

        if actor_patch_mask is not None:
            rel_mask = torch.ones(actor_patch_mask.shape[:2] + (1,), device=actor_patch_mask.device, dtype=torch.bool)
            actor_mask = torch.cat([actor_patch_mask, rel_mask], dim=2)
        else:
            actor_mask = None

        for block in self.temporal_blocks:
            reactor = block(reactor, time_mask=time_mask)
        for block in self.cross_blocks:
            reactor = block(reactor, actor_tokens, actor_mask=actor_mask)
        if self.use_spatial_attn:
            for block in self.spatial_blocks:
                reactor = block(reactor)

        wrist_logits = self.wrist_head(reactor)
        finger_logits = self.finger_head(reactor)
        wrist_mask = self.wrist_mask.to(reactor.device).view(1, 1, -1, 1)
        delta = torch.where(wrist_mask, wrist_logits, finger_logits)
        delta = self._bounded_delta(delta)
        return delta
