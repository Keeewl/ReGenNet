"""Mesh-aware condition encoder for refine_v2 window refiner."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class RefineV2ConditionEncoderConfig:
    hidden_dim: int = 256
    num_hands: int = 2
    num_regions: int = 6
    top_k_regions: int = 3
    dropout: float = 0.0
    use_geometry_features: bool = False
    use_geometry_v2_features: bool = False
    use_hand_target_interaction: bool = False
    use_hand_target_spatial_attention: bool = False
    interaction_num_layers: int = 1
    interaction_num_heads: int = 4


class RefineV2SpatialInteractionBlock(nn.Module):
    """Lightweight per-frame hand/arm-to-target spatial interaction block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.region_norm = nn.LayerNorm(hidden_dim)
        self.region_self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.hand_query_norm = nn.LayerNorm(hidden_dim)
        self.arm_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.hand_ffn_norm = nn.LayerNorm(hidden_dim)
        self.arm_ffn_norm = nn.LayerNorm(hidden_dim)
        self.region_ffn_norm = nn.LayerNorm(hidden_dim)
        self.hand_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.arm_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.region_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hand_query: torch.Tensor,
        arm_query: torch.Tensor,
        region_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        region_y, _ = self.region_self_attn(
            self.region_norm(region_tokens),
            self.region_norm(region_tokens),
            self.region_norm(region_tokens),
            need_weights=False,
        )
        region_tokens = region_tokens + region_y

        hand_y, _ = self.cross_attn(
            self.hand_query_norm(hand_query),
            region_tokens,
            region_tokens,
            need_weights=False,
        )
        hand_query = hand_query + hand_y

        arm_y, _ = self.cross_attn(
            self.arm_query_norm(arm_query),
            region_tokens,
            region_tokens,
            need_weights=False,
        )
        arm_query = arm_query + arm_y

        hand_query = hand_query + self.hand_ffn(self.hand_ffn_norm(hand_query))
        arm_query = arm_query + self.arm_ffn(self.arm_ffn_norm(arm_query))
        region_tokens = region_tokens + self.region_ffn(self.region_ffn_norm(region_tokens))
        return hand_query, arm_query, region_tokens


class RefineV2ConditionEncoder(nn.Module):
    """Encode hand/region/top-k/contact conditions into refiner tokens."""

    def __init__(self, config: RefineV2ConditionEncoderConfig):
        super().__init__()
        self.config = config
        d = int(config.hidden_dim)
        self.hand_embedding = nn.Embedding(int(config.num_hands), d)
        self.region_embedding = nn.Embedding(int(config.num_regions), d)
        self.topk_score_mlp = nn.Sequential(
            nn.Linear(3, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(d * 3),
            nn.Linear(d * 3, d),
            nn.SiLU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(d, d),
        )
        self.frame_contact_mlp = nn.Sequential(
            nn.Linear(int(config.num_regions) * 2, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.geometry_mlp = None
        if bool(config.use_geometry_features):
            geom_dim = 4 + int(config.top_k_regions) * 4
            if bool(config.use_geometry_v2_features):
                geom_dim += int(config.top_k_regions) * 5
            self.geometry_mlp = nn.Sequential(
                nn.Linear(geom_dim, d),
                nn.SiLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(d, d),
            )
        self.interaction_query_mlp = None
        self.interaction_arm_query_mlp = None
        self.interaction_region_mlp = None
        self.interaction_region_fuse = None
        self.hand_interaction_mlp = None
        self.arm_interaction_mlp = None
        self.interaction_blocks = None
        if bool(config.use_hand_target_interaction):
            region_dim = 4 + (5 if bool(config.use_geometry_v2_features) else 0)
            self.interaction_query_mlp = nn.Sequential(
                nn.LayerNorm(d + 4),
                nn.Linear(d + 4, d),
                nn.SiLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(d, d),
            )
            self.interaction_region_mlp = nn.Sequential(
                nn.LayerNorm(region_dim),
                nn.Linear(region_dim, d),
                nn.SiLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(d, d),
            )
            self.hand_interaction_mlp = nn.Sequential(
                nn.LayerNorm(d * 2),
                nn.Linear(d * 2, d),
                nn.SiLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(d, d),
            )
            self.arm_interaction_mlp = nn.Sequential(
                nn.LayerNorm(d * 2),
                nn.Linear(d * 2, d),
                nn.SiLU(),
                nn.Dropout(float(config.dropout)),
                nn.Linear(d, d),
            )
            if bool(config.use_hand_target_spatial_attention):
                self.interaction_arm_query_mlp = nn.Sequential(
                    nn.LayerNorm(d * 2 + 4),
                    nn.Linear(d * 2 + 4, d),
                    nn.SiLU(),
                    nn.Dropout(float(config.dropout)),
                    nn.Linear(d, d),
                )
                self.interaction_region_fuse = nn.Sequential(
                    nn.LayerNorm(d * 2),
                    nn.Linear(d * 2, d),
                    nn.SiLU(),
                    nn.Dropout(float(config.dropout)),
                    nn.Linear(d, d),
                )
                self.interaction_blocks = nn.ModuleList(
                    [
                        RefineV2SpatialInteractionBlock(
                            hidden_dim=d,
                            num_heads=int(config.interaction_num_heads),
                            dropout=float(config.dropout),
                        )
                        for _ in range(max(1, int(config.interaction_num_layers)))
                    ]
                )
        self.frame_fuse = nn.Sequential(
            nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d),
            nn.SiLU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(d, d),
        )

    def forward(
        self,
        *,
        hand_side_id: torch.Tensor,
        primary_target_region_id: torch.Tensor,
        topk_target_region_ids: torch.Tensor,
        topk_region_scores_numeric: torch.Tensor,
        coarse_region_contact_mask_window: torch.Tensor,
        coarse_min_region_dist_window: torch.Tensor,
        primary_relative_vector_window: torch.Tensor | None = None,
        primary_relative_dist_window: torch.Tensor | None = None,
        topk_relative_vectors_window: torch.Tensor | None = None,
        topk_relative_dists_window: torch.Tensor | None = None,
        topk_relative_dist_velocity_window: torch.Tensor | None = None,
        coarse_topk_nearest_vectors_window: torch.Tensor | None = None,
        coarse_topk_nearest_dists_window: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hand_side_id = hand_side_id.long().clamp(0, self.config.num_hands - 1)
        primary_target_region_id = primary_target_region_id.long().clamp(0, self.config.num_regions - 1)
        topk_target_region_ids = topk_target_region_ids.long().clamp(0, self.config.num_regions - 1)

        hand_emb = self.hand_embedding(hand_side_id)
        primary_emb = self.region_embedding(primary_target_region_id)

        topk_region_emb = self.region_embedding(topk_target_region_ids)
        topk_scores = topk_region_scores_numeric.float().clone()
        if topk_scores.numel():
            topk_scores[..., 0] = torch.log1p(topk_scores[..., 0].clamp_min(0.0)) / 4.0
            topk_scores[..., 1:] = topk_scores[..., 1:].clamp(0.0, 2.0)
        topk_score_emb = self.topk_score_mlp(topk_scores)
        topk_emb = (topk_region_emb + topk_score_emb).mean(dim=1)

        global_condition = self.global_mlp(torch.cat([hand_emb, primary_emb, topk_emb], dim=-1))

        contact = coarse_region_contact_mask_window.float().transpose(1, 2)
        dist = coarse_min_region_dist_window.float().transpose(1, 2)
        dist = torch.nan_to_num(dist, nan=0.0, posinf=10.0, neginf=0.0).clamp(0.0, 10.0)
        frame_contact = self.frame_contact_mlp(torch.cat([contact, dist], dim=-1))
        hand_interaction_condition = None
        arm_interaction_condition = None
        if self.geometry_mlp is not None:
            if (
                primary_relative_vector_window is None
                or primary_relative_dist_window is None
                or topk_relative_vectors_window is None
                or topk_relative_dists_window is None
            ):
                raise KeyError(
                    "use_geometry_features=True requires geometry cache fields in the batch: "
                    "primary_relative_vector_window, primary_relative_dist_window, "
                    "topk_relative_vectors_window, topk_relative_dists_window."
                )
            primary_vec = primary_relative_vector_window.float().transpose(1, 2)
            primary_dist = primary_relative_dist_window.float().unsqueeze(-1)
            topk_vec = topk_relative_vectors_window.float().permute(0, 3, 1, 2).flatten(2)
            topk_dist = topk_relative_dists_window.float().transpose(1, 2)
            geom_parts = [primary_vec, primary_dist, topk_vec, topk_dist]
            if bool(self.config.use_geometry_v2_features):
                if (
                    topk_relative_dist_velocity_window is None
                    or coarse_topk_nearest_vectors_window is None
                    or coarse_topk_nearest_dists_window is None
                ):
                    raise KeyError(
                        "use_geometry_v2_features=True requires v2 geometry cache fields: "
                        "topk_relative_dist_velocity_window, coarse_topk_nearest_vectors_window, "
                        "coarse_topk_nearest_dists_window."
                    )
                topk_vel = topk_relative_dist_velocity_window.float().transpose(1, 2)
                nearest_vec = coarse_topk_nearest_vectors_window.float().permute(0, 3, 1, 2).flatten(2)
                nearest_dist = coarse_topk_nearest_dists_window.float().transpose(1, 2)
                geom_parts.extend([topk_vel, nearest_vec, nearest_dist])
            geom = torch.cat(geom_parts, dim=-1)
            geom = torch.nan_to_num(geom, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
            frame_contact = frame_contact + self.geometry_mlp(geom)
        if bool(self.config.use_hand_target_interaction):
            if (
                primary_relative_vector_window is None
                or primary_relative_dist_window is None
                or topk_relative_vectors_window is None
                or topk_relative_dists_window is None
            ):
                raise KeyError(
                    "use_hand_target_interaction=True requires geometry fields: "
                    "primary_relative_vector_window, primary_relative_dist_window, "
                    "topk_relative_vectors_window, topk_relative_dists_window."
                )
            primary_vec = primary_relative_vector_window.float().transpose(1, 2)
            primary_dist = primary_relative_dist_window.float().unsqueeze(-1)
            hand_query = self.interaction_query_mlp(torch.cat([frame_contact, primary_vec, primary_dist], dim=-1))
            region_parts = [
                topk_relative_vectors_window.float().permute(0, 3, 1, 2),
                topk_relative_dists_window.float().transpose(1, 2).unsqueeze(-1),
            ]
            if bool(self.config.use_geometry_v2_features):
                if coarse_topk_nearest_vectors_window is None or coarse_topk_nearest_dists_window is None:
                    raise KeyError(
                        "use_hand_target_interaction with use_geometry_v2_features=True requires "
                        "coarse_topk_nearest_vectors_window and coarse_topk_nearest_dists_window."
                    )
                if topk_relative_dist_velocity_window is None:
                    raise KeyError(
                        "use_hand_target_interaction with use_geometry_v2_features=True requires "
                        "topk_relative_dist_velocity_window."
                    )
                region_parts.extend(
                    [
                        topk_relative_dist_velocity_window.float().transpose(1, 2).unsqueeze(-1),
                        coarse_topk_nearest_vectors_window.float().permute(0, 3, 1, 2),
                        coarse_topk_nearest_dists_window.float().transpose(1, 2).unsqueeze(-1),
                    ]
                )
            region_tokens = self.interaction_region_mlp(torch.cat(region_parts, dim=-1))
            if self.interaction_blocks is not None:
                arm_query = self.interaction_arm_query_mlp(
                    torch.cat(
                        [
                            frame_contact,
                            global_condition.unsqueeze(1).expand_as(frame_contact),
                            primary_vec,
                            primary_dist,
                        ],
                        dim=-1,
                    )
                )
                region_cond = (topk_region_emb + topk_score_emb).unsqueeze(1).expand(
                    -1,
                    region_tokens.shape[1],
                    -1,
                    -1,
                )
                region_tokens = self.interaction_region_fuse(torch.cat([region_tokens, region_cond], dim=-1))
                bt = int(region_tokens.shape[0] * region_tokens.shape[1])
                num_regions = int(region_tokens.shape[2])
                hand_query_bt = hand_query.reshape(bt, 1, -1)
                arm_query_bt = arm_query.reshape(bt, 1, -1)
                region_tokens_bt = region_tokens.reshape(bt, num_regions, -1)
                for block in self.interaction_blocks:
                    hand_query_bt, arm_query_bt, region_tokens_bt = block(
                        hand_query_bt,
                        arm_query_bt,
                        region_tokens_bt,
                    )
                hand_query = hand_query_bt.reshape_as(hand_query)
                arm_query = arm_query_bt.reshape_as(arm_query)
                region_tokens = region_tokens_bt.reshape_as(region_tokens)
                pooled_regions = region_tokens.mean(dim=2)
                interaction = 0.5 * (hand_query + arm_query)
                arm_condition_src = arm_query
            else:
                scores = (hand_query.unsqueeze(2) * region_tokens).sum(dim=-1) / math.sqrt(float(hand_query.shape[-1]))
                attn = torch.softmax(scores, dim=2)
                interaction = (attn.unsqueeze(-1) * region_tokens).sum(dim=2)
                pooled_regions = interaction
                arm_condition_src = frame_contact
            frame_contact = frame_contact + interaction
            hand_interaction_condition = self.hand_interaction_mlp(torch.cat([pooled_regions, hand_query], dim=-1))
            arm_interaction_condition = self.arm_interaction_mlp(torch.cat([pooled_regions, arm_condition_src], dim=-1))
        per_frame_condition = self.frame_fuse(
            torch.cat([frame_contact, global_condition.unsqueeze(1).expand_as(frame_contact)], dim=-1)
        )
        out = {
            "global_condition": global_condition,
            "per_frame_condition": per_frame_condition,
        }
        if hand_interaction_condition is not None:
            out["hand_interaction_condition"] = hand_interaction_condition
        if arm_interaction_condition is not None:
            out["arm_interaction_condition"] = arm_interaction_condition
        return out
