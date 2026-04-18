"""Transformer-style local residual refiner for Stage2-lite.

This module implements the current joint-based baseline network:

- coarse local reactor motion is the main stream
- actor target local motion is the conditioning stream
- target summary features provide lightweight relation modulation

It predicts a low-amplitude local residual `delta_local` instead of
re-generating the full sequence. The design intentionally stays lightweight and
hand-centric, and does not include diffusion or mesh-aware components.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class JointLocalRefinerConfig:
    hidden_dim: int = 256
    num_heads: int = 4
    num_blocks: int = 3
    dropout: float = 0.1
    mlp_ratio: float = 2.0
    delta_scale: float = 0.15
    use_time_pos_embed: bool = True
    use_hand_side_embed: bool = True
    max_window_size: int = 32
    motion_dim: int = 6
    summary_dim: int = 17
    num_joint_roles: int = 3
    num_target_parts: int = 6
    num_window_states: int = 2
    num_hand_sides: int = 2


def _safe_inverted_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert valid-mask to key_padding_mask and avoid all-masked rows."""

    key_padding_mask = ~mask.bool()
    all_masked = key_padding_mask.all(dim=-1)
    if bool(all_masked.any()):
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_masked, 0] = False
    return key_padding_mask


def _apply_time_mask(tokens: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
    """Zero invalid frames for tensors shaped [N, W, ..., D]."""

    return tokens * time_mask[:, :, None, None].to(dtype=tokens.dtype, device=tokens.device)


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, reactor_tokens: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        num_windows, window_size, num_joints, hidden_dim = reactor_tokens.shape
        tokens = reactor_tokens.permute(0, 2, 1, 3).reshape(num_windows * num_joints, window_size, hidden_dim)
        normed = self.norm(tokens)
        attn_mask = _safe_inverted_padding_mask(
            time_mask.unsqueeze(1).expand(num_windows, num_joints, window_size).reshape(num_windows * num_joints, window_size)
        )
        attended, _ = self.attn(normed, normed, normed, key_padding_mask=attn_mask, need_weights=False)
        tokens = tokens + self.dropout(attended)
        tokens = tokens.reshape(num_windows, num_joints, window_size, hidden_dim).permute(0, 2, 1, 3)
        return _apply_time_mask(tokens, time_mask)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        reactor_tokens: torch.Tensor,
        actor_tokens: torch.Tensor,
        actor_target_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_windows, window_size, num_joints, hidden_dim = reactor_tokens.shape
        num_targets = actor_tokens.shape[2]

        query = reactor_tokens.reshape(num_windows * window_size, num_joints, hidden_dim)
        residual = query
        context = actor_tokens.reshape(num_windows * window_size, num_targets, hidden_dim)
        query = self.query_norm(query)
        context = self.context_norm(context)

        actor_key_padding_mask = _safe_inverted_padding_mask(
            actor_target_mask.unsqueeze(1).expand(num_windows, window_size, num_targets).reshape(num_windows * window_size, num_targets)
        )
        attended, _ = self.attn(
            query,
            context,
            context,
            key_padding_mask=actor_key_padding_mask,
            need_weights=False,
        )
        query = residual + self.dropout(attended)
        query = query.reshape(num_windows, window_size, num_joints, hidden_dim)
        return _apply_time_mask(query, time_mask)


class SpatialSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, reactor_tokens: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        num_windows, window_size, num_joints, hidden_dim = reactor_tokens.shape
        tokens = reactor_tokens.reshape(num_windows * window_size, num_joints, hidden_dim)
        normed = self.norm(tokens)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + self.dropout(attended)
        tokens = tokens.reshape(num_windows, window_size, num_joints, hidden_dim)
        return _apply_time_mask(tokens, time_mask)


class RelationFiLM(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_scale_shift = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, reactor_tokens: torch.Tensor, summary_tokens: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_scale_shift(summary_tokens)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        modulated = self.norm(reactor_tokens)
        return modulated * (1.0 + gamma[:, :, None, :]) + beta[:, :, None, :]


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, reactor_tokens: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        updated = reactor_tokens + self.dropout(self.ffn(self.norm(reactor_tokens)))
        return _apply_time_mask(updated, time_mask)


class RefinerBlock(nn.Module):
    def __init__(self, config: JointLocalRefinerConfig):
        super().__init__()
        self.temporal = TemporalSelfAttentionBlock(config.hidden_dim, config.num_heads, config.dropout)
        self.cross = CrossAttentionBlock(config.hidden_dim, config.num_heads, config.dropout)
        self.spatial = SpatialSelfAttentionBlock(config.hidden_dim, config.num_heads, config.dropout)
        self.relation = RelationFiLM(config.hidden_dim)
        self.ffn = FeedForwardBlock(config.hidden_dim, config.mlp_ratio, config.dropout)

    def forward(
        self,
        reactor_tokens: torch.Tensor,
        actor_tokens: torch.Tensor,
        summary_tokens: torch.Tensor,
        time_mask: torch.Tensor,
        actor_target_mask: torch.Tensor,
    ) -> torch.Tensor:
        reactor_tokens = self.temporal(reactor_tokens, time_mask)
        reactor_tokens = self.cross(reactor_tokens, actor_tokens, actor_target_mask, time_mask)
        reactor_tokens = self.spatial(reactor_tokens, time_mask)
        reactor_tokens = self.relation(reactor_tokens, summary_tokens)
        reactor_tokens = self.ffn(reactor_tokens, time_mask)
        return reactor_tokens


class JointLocalRefiner(nn.Module):
    """Hand-centric temporal cross-transformer residual refiner."""

    def __init__(self, config: JointLocalRefinerConfig | None = None):
        super().__init__()
        self.config = config or JointLocalRefinerConfig()

        self.reactor_input_proj = nn.Linear(self.config.motion_dim, self.config.hidden_dim)
        self.actor_input_proj = nn.Linear(self.config.motion_dim, self.config.hidden_dim)
        self.summary_proj = nn.Linear(self.config.summary_dim, self.config.hidden_dim)

        self.joint_role_embed = nn.Embedding(self.config.num_joint_roles, self.config.hidden_dim)
        self.target_part_embed = nn.Embedding(self.config.num_target_parts, self.config.hidden_dim)
        self.window_state_embed = nn.Embedding(self.config.num_window_states, self.config.hidden_dim)
        self.hand_side_embed = (
            nn.Embedding(self.config.num_hand_sides, self.config.hidden_dim)
            if self.config.use_hand_side_embed
            else None
        )
        self.time_pos_embed = (
            nn.Embedding(self.config.max_window_size, self.config.hidden_dim)
            if self.config.use_time_pos_embed
            else None
        )

        self.blocks = nn.ModuleList(
            [RefinerBlock(self.config) for _ in range(self.config.num_blocks)]
        )
        self.output_norm = nn.LayerNorm(self.config.hidden_dim)
        self.delta_head = nn.Linear(self.config.hidden_dim, self.config.motion_dim)

    def _check_window_batch(self, window_batch) -> None:
        coarse_local = window_batch["coarse_local"]
        actor_target_local = window_batch["actor_target_local"]
        target_summary_feat = window_batch["target_summary_feat"]
        if coarse_local.dim() != 4:
            raise ValueError("coarse_local must have shape [Nw, J_ref, F, W].")
        if actor_target_local.dim() != 4:
            raise ValueError("actor_target_local must have shape [Nw, J_t, F, W].")
        if target_summary_feat.dim() != 3:
            raise ValueError("target_summary_feat must have shape [Nw, W, C].")
        if coarse_local.shape[2] != self.config.motion_dim:
            raise ValueError(
                f"JointLocalRefiner expects coarse_local feature dim {self.config.motion_dim}, "
                f"got {coarse_local.shape[2]}."
            )
        if actor_target_local.shape[2] != self.config.motion_dim:
            raise ValueError(
                f"JointLocalRefiner expects actor_target_local feature dim {self.config.motion_dim}, "
                f"got {actor_target_local.shape[2]}."
            )
        if target_summary_feat.shape[-1] != self.config.summary_dim:
            raise ValueError(
                f"JointLocalRefiner expects target_summary_feat dim {self.config.summary_dim}, "
                f"got {target_summary_feat.shape[-1]}."
            )
        if coarse_local.shape[-1] > self.config.max_window_size:
            raise ValueError(
                f"Window size {coarse_local.shape[-1]} exceeds max_window_size={self.config.max_window_size}."
            )

    def _build_condition_context(self, window_batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        target_part = window_batch["target_part_id"].long().to(device)
        window_state = window_batch["window_state_id"].long().to(device)
        part_embed = self.target_part_embed(target_part)
        state_embed = self.window_state_embed(window_state)
        summary_bias = part_embed + state_embed

        token_bias = summary_bias[:, None, None, :]
        if self.hand_side_embed is not None and "hand_side_id" in window_batch:
            hand_side = window_batch["hand_side_id"].long().to(device)
            hand_embed = self.hand_side_embed(hand_side)
            token_bias = token_bias + hand_embed[:, None, None, :]
            summary_bias = summary_bias + hand_embed
        return token_bias, summary_bias[:, None, :]

    def forward(self, window_batch):
        self._check_window_batch(window_batch)

        coarse_local = window_batch["coarse_local"]
        actor_target_local = window_batch["actor_target_local"]
        time_mask = window_batch["time_mask"].bool().to(coarse_local.device)
        actor_target_mask = window_batch["actor_target_mask"].bool().to(coarse_local.device)

        if coarse_local.shape[0] == 0:
            delta_local = torch.zeros_like(coarse_local)
            return {
                "delta_local": delta_local,
                "refined_local": coarse_local + delta_local,
            }

        reactor_tokens = coarse_local.permute(0, 3, 1, 2).contiguous()
        actor_tokens = actor_target_local.permute(0, 3, 1, 2).contiguous()

        reactor_tokens = self.reactor_input_proj(reactor_tokens)
        actor_tokens = self.actor_input_proj(actor_tokens)
        summary_tokens = self.summary_proj(window_batch["target_summary_feat"].to(coarse_local.device))

        joint_role_embed = self.joint_role_embed(
            window_batch["joint_role_id"].long().to(coarse_local.device)
        )[None, None, :, :]
        reactor_tokens = reactor_tokens + joint_role_embed

        token_bias, summary_bias = self._build_condition_context(window_batch, coarse_local.device)
        reactor_tokens = reactor_tokens + token_bias
        actor_tokens = actor_tokens + token_bias
        summary_tokens = summary_tokens + summary_bias

        if self.time_pos_embed is not None:
            time_index = torch.arange(reactor_tokens.shape[1], device=coarse_local.device)
            time_embed = self.time_pos_embed(time_index)[None, :, None, :]
            reactor_tokens = reactor_tokens + time_embed
            actor_tokens = actor_tokens + time_embed
            summary_tokens = summary_tokens + time_embed[:, :, 0, :]

        reactor_tokens = _apply_time_mask(reactor_tokens, time_mask)
        actor_tokens = _apply_time_mask(actor_tokens, time_mask)
        actor_tokens = actor_tokens * actor_target_mask[:, None, :, None].to(dtype=actor_tokens.dtype, device=actor_tokens.device)
        summary_tokens = summary_tokens * time_mask[:, :, None].to(dtype=summary_tokens.dtype, device=summary_tokens.device)

        for block in self.blocks:
            reactor_tokens = block(
                reactor_tokens,
                actor_tokens,
                summary_tokens,
                time_mask,
                actor_target_mask,
            )

        reactor_tokens = self.output_norm(reactor_tokens)
        raw_delta = self.delta_head(reactor_tokens)
        raw_delta = raw_delta * time_mask[:, :, None, None].to(dtype=raw_delta.dtype, device=raw_delta.device)
        delta_local = self.config.delta_scale * torch.tanh(raw_delta)
        delta_local = delta_local.permute(0, 2, 3, 1).contiguous()
        refined_local = coarse_local + delta_local
        return {
            "delta_local": delta_local,
            "refined_local": refined_local,
        }
