"""Mesh-aware condition encoder for refine_v2 window refiner."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RefineV2ConditionEncoderConfig:
    hidden_dim: int = 256
    num_hands: int = 2
    num_regions: int = 6
    top_k_regions: int = 3
    dropout: float = 0.0


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
        per_frame_condition = self.frame_fuse(
            torch.cat([frame_contact, global_condition.unsqueeze(1).expand_as(frame_contact)], dim=-1)
        )
        return {
            "global_condition": global_condition,
            "per_frame_condition": per_frame_condition,
        }
