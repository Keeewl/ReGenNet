"""First trainable window-level residual refiner for refine_v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from refine_v2.model.condition_encoder import RefineV2ConditionEncoder, RefineV2ConditionEncoderConfig
from refine_v2.model.joint_groups import (
    LEFT_ARM_IDS,
    LEFT_HAND_IDS,
    RIGHT_ARM_IDS,
    RIGHT_HAND_IDS,
    TRANSL_INDEX,
    ResidualGroupScales,
    residual_scale_tensor,
)


@dataclass
class RefineV2WindowRefinerConfig:
    motion_num_joints: int
    motion_num_channels: int
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    max_window_size: int = 256
    num_hands: int = 2
    num_regions: int = 6
    top_k_regions: int = 3
    delta_scale: float = 1.0
    use_geometry_features: bool = False
    use_geometry_v2_features: bool = False
    use_separate_residual_heads: bool = False
    use_group_gated_residual: bool = False
    hand_delta_scale: float = 1.0
    arm_delta_scale: float = 1.0
    torso_delta_scale: float = 0.5
    root_delta_scale: float = 0.2
    transl_delta_scale: float = 0.2
    lower_body_delta_scale: float = 0.1


class RefineV2RefinerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cond = nn.LayerNorm(hidden_dim)
        self.cond_mod = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.cond_mod.weight)
        nn.init.zeros_(self.cond_mod.bias)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        ffn_dim = int(round(hidden_dim * float(mlp_ratio)))
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        actor_tokens: torch.Tensor,
        global_condition: torch.Tensor,
        per_frame_condition: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y, _ = self.self_attn(
            self.norm_self(x),
            self.norm_self(x),
            self.norm_self(x),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + y
        y, _ = self.cross_attn(
            self.norm_cross(x),
            actor_tokens,
            actor_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + y
        scale, shift = self.cond_mod(global_condition).chunk(2, dim=-1)
        cond = self.norm_cond(x + per_frame_condition)
        x = x + cond * scale.unsqueeze(1) + shift.unsqueeze(1) + per_frame_condition
        x = x + self.ffn(self.norm_ffn(x))
        return x


class RefineV2WindowRefiner(nn.Module):
    """Temporal residual refiner with actor and mesh-aware conditioning."""

    def __init__(self, config: RefineV2WindowRefinerConfig):
        super().__init__()
        self.config = config
        self.motion_dim = int(config.motion_num_joints) * int(config.motion_num_channels)
        d = int(config.hidden_dim)
        self.coarse_motion_proj = nn.Linear(self.motion_dim, d)
        self.actor_motion_proj = nn.Linear(self.motion_dim, d)
        self.position_embedding = nn.Parameter(torch.zeros(1, int(config.max_window_size), d))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.condition_encoder = RefineV2ConditionEncoder(
            RefineV2ConditionEncoderConfig(
                hidden_dim=d,
                num_hands=int(config.num_hands),
                num_regions=int(config.num_regions),
                top_k_regions=int(config.top_k_regions),
                dropout=float(config.dropout),
                use_geometry_features=bool(config.use_geometry_features),
                use_geometry_v2_features=bool(config.use_geometry_v2_features),
            )
        )
        self.blocks = nn.ModuleList(
            [
                RefineV2RefinerBlock(
                    hidden_dim=d,
                    num_heads=int(config.num_heads),
                    dropout=float(config.dropout),
                    mlp_ratio=float(config.mlp_ratio),
                )
                for _ in range(int(config.num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(d)
        self.output_head = nn.Linear(d, self.motion_dim)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        self.hand_output_head = None
        self.arm_output_head = None
        self.body_output_head = None
        self.transl_output_head = None
        if bool(config.use_separate_residual_heads):
            self.hand_output_head = nn.Linear(d, self.motion_dim)
            self.arm_output_head = nn.Linear(d, self.motion_dim)
            self.body_output_head = nn.Linear(d, self.motion_dim)
            self.transl_output_head = nn.Linear(d, self.motion_dim)
            for head in (
                self.hand_output_head,
                self.arm_output_head,
                self.body_output_head,
                self.transl_output_head,
            ):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def _motion_to_tokens(self, value: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"motion tensor must be [B,J,F,T], got shape={tuple(value.shape)}")
        b, j, f, t = value.shape
        if j != self.config.motion_num_joints or f != self.config.motion_num_channels:
            raise ValueError(
                "motion shape does not match model config: "
                f"got J={j}, F={f}, expected J={self.config.motion_num_joints}, F={self.config.motion_num_channels}"
            )
        return proj(value.permute(0, 3, 1, 2).reshape(b, t, j * f).float())

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coarse_motion = batch["coarse_motion_window"].float()
        actor_motion = batch["actor_motion_window"].float()
        b, j, f, t = coarse_motion.shape
        if t > self.position_embedding.shape[1]:
            raise ValueError(f"window length {t} exceeds max_window_size={self.position_embedding.shape[1]}")

        coarse_tokens = self._motion_to_tokens(coarse_motion, self.coarse_motion_proj)
        actor_tokens = self._motion_to_tokens(actor_motion, self.actor_motion_proj)
        x = coarse_tokens + self.position_embedding[:, :t, :]
        actor_tokens = actor_tokens + self.position_embedding[:, :t, :]

        cond = self.condition_encoder(
            hand_side_id=batch["hand_side_id"],
            primary_target_region_id=batch["primary_target_region_id"],
            topk_target_region_ids=batch["topk_target_region_ids"],
            topk_region_scores_numeric=batch["topk_region_scores_numeric"],
            coarse_region_contact_mask_window=batch["coarse_region_contact_mask_window"],
            coarse_min_region_dist_window=batch["coarse_min_region_dist_window"],
            primary_relative_vector_window=batch.get("primary_relative_vector_window"),
            primary_relative_dist_window=batch.get("primary_relative_dist_window"),
            topk_relative_vectors_window=batch.get("topk_relative_vectors_window"),
            topk_relative_dists_window=batch.get("topk_relative_dists_window"),
            topk_relative_dist_velocity_window=batch.get("topk_relative_dist_velocity_window"),
            coarse_topk_nearest_vectors_window=batch.get("coarse_topk_nearest_vectors_window"),
            coarse_topk_nearest_dists_window=batch.get("coarse_topk_nearest_dists_window"),
        )

        valid_mask = batch.get("valid_mask")
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()

        for block in self.blocks:
            x = block(
                x,
                actor_tokens,
                cond["global_condition"],
                cond["per_frame_condition"],
                key_padding_mask=key_padding_mask,
            )

        x_norm = self.output_norm(x)
        if bool(self.config.use_separate_residual_heads):
            head_tokens = (
                self.hand_output_head(x_norm),
                self.arm_output_head(x_norm),
                self.body_output_head(x_norm),
                self.transl_output_head(x_norm),
            )
            head_deltas = [
                tokens.reshape(b, t, j, f).permute(0, 2, 3, 1).contiguous()
                for tokens in head_tokens
            ]
            masks = self._residual_head_masks(j, f, device=coarse_motion.device, dtype=coarse_motion.dtype)
            pred_delta = (
                head_deltas[0] * masks["hand"]
                + head_deltas[1] * masks["arm"]
                + head_deltas[2] * masks["body"]
                + head_deltas[3] * masks["transl"]
            ) * float(self.config.delta_scale)
        else:
            delta_tokens = self.output_head(x_norm) * float(self.config.delta_scale)
            pred_delta = delta_tokens.reshape(b, t, j, f).permute(0, 2, 3, 1).contiguous()
        if bool(self.config.use_group_gated_residual):
            scales = residual_scale_tensor(
                num_joints=j,
                num_channels=f,
                device=pred_delta.device,
                dtype=pred_delta.dtype,
                scales=ResidualGroupScales(
                    hand=float(self.config.hand_delta_scale),
                    arm=float(self.config.arm_delta_scale),
                    torso=float(self.config.torso_delta_scale),
                    root=float(self.config.root_delta_scale),
                    transl=float(self.config.transl_delta_scale),
                    lower_body=float(self.config.lower_body_delta_scale),
                ),
            )
            pred_delta = pred_delta * scales
        pred_motion = coarse_motion + pred_delta
        return {
            "pred_delta_motion_window": pred_delta,
            "pred_motion_window": pred_motion,
            "coarse_motion_window": coarse_motion,
        }

    def _residual_head_masks(self, num_joints: int, num_channels: int, *, device, dtype) -> dict[str, torch.Tensor]:
        hand_ids = [idx for idx in LEFT_HAND_IDS + RIGHT_HAND_IDS if 0 <= int(idx) < int(num_joints)]
        arm_ids = [idx for idx in LEFT_ARM_IDS + RIGHT_ARM_IDS if 0 <= int(idx) < int(num_joints)]
        transl_ids = [TRANSL_INDEX] if 0 <= int(TRANSL_INDEX) < int(num_joints) else []
        hand = torch.zeros((1, num_joints, num_channels, 1), device=device, dtype=dtype)
        arm = torch.zeros_like(hand)
        transl = torch.zeros_like(hand)
        if hand_ids:
            hand[:, hand_ids, :, :] = 1.0
        if arm_ids:
            arm[:, arm_ids, :, :] = 1.0
        if transl_ids:
            transl[:, transl_ids, :, :] = 1.0
        body = (1.0 - hand - arm - transl).clamp_min(0.0)
        return {"hand": hand, "arm": arm, "body": body, "transl": transl}
