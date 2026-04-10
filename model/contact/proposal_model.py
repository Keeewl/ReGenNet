import math
import torch
import torch.nn as nn


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

    def forward(self, x):
        bsz, num_frames, num_tokens, hidden = x.shape
        y = self.ln1(x).permute(0, 2, 1, 3).reshape(bsz * num_tokens, num_frames, hidden)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        attn_out = attn_out.reshape(bsz, num_tokens, num_frames, hidden).permute(0, 2, 1, 3)
        x = x + attn_out
        z = self.ln2(x).permute(0, 2, 1, 3).reshape(bsz * num_tokens, num_frames, hidden)
        mlp_out = self.mlp(z)
        mlp_out = mlp_out.reshape(bsz, num_tokens, num_frames, hidden).permute(0, 2, 1, 3)
        return x + mlp_out


class HandPartCrossAttentionBlock(nn.Module):
    """
    Hand-to-part cross-attention using relation-conditioned keys/values.

    Inputs:
        hand_feat: [B, T, 2, D]
        part_feat: [B, T, 5, D]
        rel_feat: [B, T, 2, 5, D]
    Outputs:
        out: [B, T, 2, D]
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.r = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hand_feat, part_feat, rel_feat):
        hidden = hand_feat.shape[-1]
        q = self.q(hand_feat)
        k = self.k(part_feat)
        v = self.v(part_feat)
        r = self.r(rel_feat)

        k = k[:, :, None, :, :] + r
        v = v[:, :, None, :, :] + r

        scores = torch.einsum("bthd,bthkd->bthk", q, k) / math.sqrt(hidden)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bthk,bthkd->bthd", attn, v)
        out = hand_feat + self.proj(ctx)
        out = out + self.mlp(self.ln(out))
        return out


class HandContactProposal(nn.Module):
    """
    Hand-level contact proposal model.

    Inputs:
        hand_feat: [B, T, 2, Fh]
        part_feat: [B, T, 5, Fp]
        rel_feat: [B, T, 2, 5, 8]
    Outputs:
        active_logits: [B, T, 2, 1]
        target_logits: [B, T, 2, 6]
        band_logits: [B, T, 2, 3]
        phase_logits: [B, T, 2, 4]
    """

    def __init__(
        self,
        hand_dim,
        part_dim,
        relation_dim=8,
        hidden_dim=64,
        num_temporal_blocks=2,
        dropout=0.1,
    ):
        super().__init__()
        self.hand_encoder = FeatureEncoder(hand_dim, hidden_dim, dropout=dropout)
        self.part_encoder = FeatureEncoder(part_dim, hidden_dim, dropout=dropout)
        self.rel_encoder = FeatureEncoder(relation_dim, hidden_dim, dropout=dropout)

        self.temporal_hand = nn.ModuleList(
            [TemporalSelfAttentionBlock(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_temporal_blocks)]
        )
        self.temporal_part = nn.ModuleList(
            [TemporalSelfAttentionBlock(hidden_dim, num_heads=4, dropout=dropout) for _ in range(num_temporal_blocks)]
        )
        self.cross_blocks = nn.ModuleList(
            [HandPartCrossAttentionBlock(hidden_dim, dropout=dropout) for _ in range(num_temporal_blocks)]
        )

        self.active_head = nn.Linear(hidden_dim, 1)
        self.target_head = nn.Linear(hidden_dim, 6)
        self.band_head = nn.Linear(hidden_dim, 3)
        self.phase_head = nn.Linear(hidden_dim, 4)

    def forward(self, hand_feat, part_feat, rel_feat):
        if hand_feat.dim() != 4 or part_feat.dim() != 4 or rel_feat.dim() != 5:
            raise ValueError("hand_feat [B,T,2,F], part_feat [B,T,5,F], rel_feat [B,T,2,5,F]")

        hand = self.hand_encoder(hand_feat)
        part = self.part_encoder(part_feat)
        rel = self.rel_encoder(rel_feat)

        for block in self.temporal_hand:
            hand = block(hand)
        for block in self.temporal_part:
            part = block(part)
        for block in self.cross_blocks:
            hand = block(hand, part, rel)

        active_logits = self.active_head(hand)
        target_logits = self.target_head(hand)
        band_logits = self.band_head(hand)
        phase_logits = self.phase_head(hand)

        return {
            "active": active_logits,
            "target": target_logits,
            "band": band_logits,
            "phase": phase_logits,
        }
