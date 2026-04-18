"""Deterministic hand-centric window selection for Stage2-lite.

Current positioning:

- This module is the baseline joint-based window selector for the new `refine/` stack.
- It operates on restored-pair-space motion using joint-level hand/target geometry and
  lightweight motion cues to produce deterministic contact-critical windows.
- It is intentionally simple, stable, and auditable.
- It is not yet the final region-aware or mesh-aware window selector.
  A stronger region/mesh-aware version may replace or extend this later, but the current
  implementation is the baseline that later Stage2-lite modules should integrate against first.

This module turns `reaction_data` batches into:

- raw contact-critical segments: variable-length semantic regions
- model windows: fixed-length time crops for the later local refiner

All frame intervals use Python slicing semantics: `[start_frame, end_frame)`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from refine.data.restored_space import (
    REQUIRED_RESTORATION_METADATA_FIELDS,
    RESTORED_PAIR_SPACE,
    extract_restoration_metadata,
    restore_pair_batch,
    validate_restoration_metadata,
)
from refine.data.schema import normalize_space_definition


WINDOW_STATE_NAMES = ("strict", "near")
WINDOW_STATE_IDS = {"strict": 0, "near": 1}

HAND_SIDE_NAMES = ("left", "right")
HAND_SIDE_IDS = {"left": 0, "right": 1}

TARGET_PART_TO_JOINT_IDS = {
    "torso_head": (0, 3, 6, 9, 12, 15, 22, 23, 24),
    "lower_body": (1, 2, 4, 5, 7, 8, 10, 11),
    "left_arm": (13, 16, 18, 20),
    "right_arm": (14, 17, 19, 21),
    "left_hand": (25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39),
    "right_hand": (40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54),
}
TARGET_PART_NAMES = tuple(TARGET_PART_TO_JOINT_IDS.keys())
TARGET_PART_IDS = {name: idx for idx, name in enumerate(TARGET_PART_NAMES)}

HAND_TO_JOINT_IDS = {
    "left": TARGET_PART_TO_JOINT_IDS["left_hand"],
    "right": TARGET_PART_TO_JOINT_IDS["right_hand"],
}


@dataclass
class WindowConfig:
    """Default hyperparameters for deterministic time ROI selection.

    Thresholds are intentionally expressed in normalized score space `[0, 1]`.
    The distance scales below remain in meters because the scoring is built from
    restored-pair-space geometry.
    """

    strict_score_threshold: float = 0.62
    near_score_threshold_pre: float = 0.42
    near_score_threshold_post: float = 0.34
    min_anchor_len: int = 2
    target_smooth_k: int = 5
    raw_L_min: int = 6
    raw_L_max: int = 24
    model_W: int = 16
    gap_merge: int = 2
    pre_max: int = 8
    post_max: int = 6
    per_hand_max_windows: int = 3
    per_seq_max_windows: int = 6
    max_target_switch_within_window: int = 0
    strict_contact_distance: float = 0.08
    near_contact_distance: float = 0.18
    margin_scale: float = 0.08
    approaching_scale: float = 0.04
    relative_speed_scale: float = 0.35
    target_motion_scale: float = 0.20
    near_only_len_min: int = 8
    near_state_priority_bias: float = 0.85


@dataclass
class RawSegment:
    batch_index: int
    dataset_key: str
    hand_side: str
    hand_side_id: int
    target_part: str
    target_part_id: int
    window_state: str
    window_state_id: int
    anchor_start_frame: int
    anchor_end_frame: int
    raw_start_frame: int
    raw_end_frame: int
    center_frame: int
    raw_length: int
    strict_peak_score: float
    near_mean_score: float
    merge_count: int = 0
    smoothed_target_consistency: float = 1.0


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return (
        torch.arange(max_len, device=lengths.device).unsqueeze(0)
        < lengths.view(-1, 1)
    )


def _to_numpy(value: Any):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_meta_scalar(value: Any, default: str = "") -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    return normalize_space_definition(value, default=default)


def _mode_with_tiebreak(ids: list[int], scores: list[float], fallback: int) -> int:
    if not ids:
        return int(fallback)
    counts = Counter(int(x) for x in ids)
    max_count = max(counts.values())
    candidates = [idx for idx, count in counts.items() if count == max_count]
    if len(candidates) == 1:
        return int(candidates[0])
    score_by_id = {idx: 0.0 for idx in candidates}
    for idx, score in zip(ids, scores):
        idx = int(idx)
        if idx in score_by_id:
            score_by_id[idx] += float(score)
    best = max(candidates, key=lambda idx: (score_by_id[idx], idx == fallback))
    return int(best)


class DeterministicWindowSelector:
    """Anchor-first deterministic selector for hand-centric local refinement.

    Notes:

    - This class is the current baseline joint-based implementation.
    - It does not depend on blueprint caches or learned proposal networks.
    - Its job is limited to deterministic time-ROI selection for later local refinement.
    """

    def __init__(
        self,
        config: WindowConfig | None = None,
        *,
        body_model: str = "smplx",
        pose_rep: str = "rot6d",
    ):
        self.config = config or WindowConfig()
        self.body_model = str(body_model)
        self.pose_rep = str(pose_rep)
        self._rot2xyz_cache: dict[str, Any] = {}

    def _get_rot2xyz(self, device: torch.device):
        key = f"{self.body_model}:{device.type}:{device.index}"
        if key in self._rot2xyz_cache:
            return self._rot2xyz_cache[key]
        if self.body_model != "smplx":
            raise ValueError(
                f"DeterministicWindowSelector currently expects body_model='smplx', got {self.body_model}."
            )
        from model.rotation2xyz import Rotation2xyz_x

        rot2xyz = Rotation2xyz_x(device=str(device), dataset="interx")
        self._rot2xyz_cache[key] = rot2xyz
        return rot2xyz

    def _restore_pair_if_needed(
        self,
        actor_motion: torch.Tensor,
        coarse_motion: torch.Tensor,
        restoration_meta: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if restoration_meta is None:
            raise ValueError(
                "Window selection requires restoration metadata and restored pair space."
            )
        if not all(key in restoration_meta for key in REQUIRED_RESTORATION_METADATA_FIELDS):
            restoration_meta = extract_restoration_metadata(restoration_meta, device=actor_motion.device)
        else:
            validate_restoration_metadata(restoration_meta, context="window selector restoration metadata")
        actor_restored, coarse_restored = restore_pair_batch(
            actor_motion,
            coarse_motion,
            restoration_meta,
        )
        return actor_restored, coarse_restored, restoration_meta

    def _motions_to_xyz(
        self,
        motion: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if motion.dim() != 4:
            raise ValueError("motion must have shape [B, J, F, T].")
        if motion.shape[2] == 3:
            return motion
        rot2xyz = self._get_rot2xyz(motion.device)
        mask = _lengths_to_mask(lengths.to(motion.device), motion.shape[-1]).bool()
        return rot2xyz(
            x=motion,
            mask=mask,
            pose_rep=self.pose_rep,
            glob=True,
            translation=True,
            jointstype=self.body_model,
            vertstrans=True,
            num_person=1,
            betas=None,
            beta=0,
            glob_rot=None,
        )

    def _compute_candidate_target_scores(
        self,
        actor_xyz: torch.Tensor,
        coarse_xyz: torch.Tensor,
        valid_len: int,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        num_parts = len(TARGET_PART_NAMES)
        target_dists = torch.zeros((2, num_parts, valid_len), device=actor_xyz.device, dtype=actor_xyz.dtype)
        other_dists = torch.zeros_like(target_dists)
        contact_ratio = torch.zeros_like(target_dists)
        rel_speed = torch.zeros_like(target_dists)
        target_motion = torch.zeros_like(target_dists)

        for hand_side, hand_id in HAND_SIDE_IDS.items():
            hand_xyz = coarse_xyz[list(HAND_TO_JOINT_IDS[hand_side]), :, :valid_len]
            hand_bt = hand_xyz.permute(2, 0, 1).contiguous()
            hand_center = hand_xyz.mean(dim=0)
            hand_vel = torch.zeros_like(hand_center)
            hand_vel[:, 1:] = hand_center[:, 1:] - hand_center[:, :-1]

            for part_name, part_id in TARGET_PART_IDS.items():
                target_xyz = actor_xyz[list(TARGET_PART_TO_JOINT_IDS[part_name]), :, :valid_len]
                target_bt = target_xyz.permute(2, 0, 1).contiguous()
                pair_dist = torch.cdist(hand_bt, target_bt)
                min_per_hand = pair_dist.min(dim=-1).values
                target_dists[hand_id, part_id] = min_per_hand.mean(dim=-1)
                contact_ratio[hand_id, part_id] = (
                    pair_dist < cfg.near_contact_distance
                ).float().mean(dim=(-1, -2))

                target_center = target_xyz.mean(dim=0)
                target_vel = torch.zeros_like(target_center)
                target_vel[:, 1:] = target_center[:, 1:] - target_center[:, :-1]
                rel_speed[hand_id, part_id] = (hand_vel - target_vel).norm(dim=0)
                target_motion[hand_id, part_id] = target_vel.norm(dim=0)

            for part_id in range(num_parts):
                if num_parts == 1:
                    other_dists[hand_id, part_id] = target_dists[hand_id, part_id]
                    continue
                mask = torch.ones(num_parts, dtype=torch.bool, device=actor_xyz.device)
                mask[part_id] = False
                other_dists[hand_id, part_id] = target_dists[hand_id, mask].min(dim=0).values

        near_closeness = (
            (cfg.near_contact_distance - target_dists) / cfg.near_contact_distance
        ).clamp(0.0, 1.0)
        strict_closeness = (
            (cfg.near_contact_distance - target_dists)
            / max(cfg.near_contact_distance - cfg.strict_contact_distance, 1e-6)
        ).clamp(0.0, 1.0)
        margin = ((other_dists - target_dists) / max(cfg.margin_scale, 1e-6)).clamp(0.0, 1.0)
        approach = torch.zeros_like(target_dists)
        approach[..., 1:] = target_dists[..., :-1] - target_dists[..., 1:]
        approaching_score = (approach / max(cfg.approaching_scale, 1e-6)).clamp(0.0, 1.0)
        relative_stability = (1.0 - rel_speed / max(cfg.relative_speed_scale, 1e-6)).clamp(0.0, 1.0)
        target_motion_score = (target_motion / max(cfg.target_motion_scale, 1e-6)).clamp(0.0, 1.0)
        selection_score = (
            0.55 * near_closeness
            + 0.20 * margin
            + 0.15 * contact_ratio
            + 0.10 * relative_stability
        )

        return {
            "target_dists": target_dists,
            "other_dists": other_dists,
            "contact_ratio": contact_ratio,
            "near_closeness": near_closeness,
            "strict_closeness": strict_closeness,
            "margin": margin,
            "approaching_score": approaching_score,
            "relative_stability": relative_stability,
            "target_motion_score": target_motion_score,
            "selection_score": selection_score,
        }

    def _smooth_target_ids(
        self,
        raw_target_ids: torch.Tensor,
        selection_score: torch.Tensor,
        valid_len: int,
    ) -> torch.Tensor:
        if self.config.target_smooth_k <= 1:
            return raw_target_ids.clone()
        smoothed = raw_target_ids.clone()
        half = self.config.target_smooth_k // 2
        for hand_id in range(raw_target_ids.shape[0]):
            for frame_idx in range(valid_len):
                start = max(0, frame_idx - half)
                end = min(valid_len, frame_idx + half + 1)
                ids = raw_target_ids[hand_id, start:end].tolist()
                scores = [
                    float(selection_score[hand_id, int(ids[i]), start + i].item())
                    for i in range(len(ids))
                ]
                smoothed[hand_id, frame_idx] = _mode_with_tiebreak(
                    ids,
                    scores,
                    fallback=int(raw_target_ids[hand_id, frame_idx].item()),
                )
        return smoothed

    def _compute_strict_near_scores(
        self,
        candidate_scores: dict[str, torch.Tensor],
        target_ids: torch.Tensor,
        valid_len: int,
    ) -> dict[str, torch.Tensor]:
        strict_scores = torch.zeros((2, valid_len), device=target_ids.device)
        near_scores = torch.zeros_like(strict_scores)
        gathered = {}
        for key, value in candidate_scores.items():
            gathered[key] = torch.zeros((2, valid_len), device=value.device, dtype=value.dtype)
            for hand_id in range(2):
                frame_index = torch.arange(valid_len, device=value.device)
                gathered[key][hand_id] = value[hand_id, target_ids[hand_id, :valid_len], frame_index]

        strict_scores = (
            0.60 * gathered["strict_closeness"]
            + 0.15 * gathered["margin"]
            + 0.10 * gathered["contact_ratio"]
            + 0.10 * gathered["relative_stability"]
            + 0.05 * gathered["approaching_score"]
        ).clamp(0.0, 1.0)
        near_scores = (
            0.55 * gathered["near_closeness"]
            + 0.15 * gathered["margin"]
            + 0.15 * gathered["approaching_score"]
            + 0.10 * gathered["relative_stability"]
            + 0.05 * gathered["target_motion_score"]
        ).clamp(0.0, 1.0)
        gathered["strict_scores"] = strict_scores
        gathered["near_scores"] = near_scores
        return gathered

    def _extract_strict_anchors(
        self,
        strict_scores: torch.Tensor,
        target_ids: torch.Tensor,
        valid_len: int,
    ) -> list[dict[str, int]]:
        anchors: list[dict[str, int]] = []
        threshold = self.config.strict_score_threshold
        min_anchor_len = self.config.min_anchor_len
        for hand_id in range(strict_scores.shape[0]):
            mask = strict_scores[hand_id, :valid_len] >= threshold
            start = 0
            while start < valid_len:
                if not bool(mask[start].item()):
                    start += 1
                    continue
                end = start + 1
                while end < valid_len and bool(mask[end].item()):
                    end += 1
                sub_start = start
                while sub_start < end:
                    target_id = int(target_ids[hand_id, sub_start].item())
                    sub_end = sub_start + 1
                    while sub_end < end and int(target_ids[hand_id, sub_end].item()) == target_id:
                        sub_end += 1
                    if sub_end - sub_start >= min_anchor_len:
                        anchors.append(
                            {
                                "hand_side_id": hand_id,
                                "target_part_id": target_id,
                                "start_frame": sub_start,
                                "end_frame": sub_end,
                            }
                        )
                    sub_start = sub_end
                start = end
        return anchors

    def _grow_segments_from_anchors(
        self,
        strict_anchors: list[dict[str, int]],
        strict_scores: torch.Tensor,
        near_scores: torch.Tensor,
        target_ids: torch.Tensor,
        valid_len: int,
        *,
        batch_index: int,
        dataset_key: str,
    ) -> list[RawSegment]:
        segments: list[RawSegment] = []
        cfg = self.config
        for anchor in strict_anchors:
            hand_id = int(anchor["hand_side_id"])
            target_id = int(anchor["target_part_id"])
            left = int(anchor["start_frame"])
            right = int(anchor["end_frame"])
            pre_steps = 0
            while left > 0 and pre_steps < cfg.pre_max:
                frame_idx = left - 1
                if int(target_ids[hand_id, frame_idx].item()) != target_id:
                    break
                if float(near_scores[hand_id, frame_idx].item()) < cfg.near_score_threshold_pre:
                    break
                left -= 1
                pre_steps += 1
            post_steps = 0
            while right < valid_len and post_steps < cfg.post_max:
                if int(target_ids[hand_id, right].item()) != target_id:
                    break
                if float(near_scores[hand_id, right].item()) < cfg.near_score_threshold_post:
                    break
                right += 1
                post_steps += 1
            strict_slice = strict_scores[hand_id, left:right]
            local_peak = int(torch.argmax(strict_slice).item()) if strict_slice.numel() > 0 else 0
            center = left + local_peak
            consistency = float(
                (target_ids[hand_id, left:right] == target_id).float().mean().item()
            ) if right > left else 1.0
            segments.append(
                RawSegment(
                    batch_index=batch_index,
                    dataset_key=dataset_key,
                    hand_side=HAND_SIDE_NAMES[hand_id],
                    hand_side_id=hand_id,
                    target_part=TARGET_PART_NAMES[target_id],
                    target_part_id=target_id,
                    window_state="strict",
                    window_state_id=WINDOW_STATE_IDS["strict"],
                    anchor_start_frame=int(anchor["start_frame"]),
                    anchor_end_frame=int(anchor["end_frame"]),
                    raw_start_frame=left,
                    raw_end_frame=right,
                    center_frame=center,
                    raw_length=max(0, right - left),
                    strict_peak_score=float(strict_scores[hand_id, center].item()),
                    near_mean_score=float(near_scores[hand_id, left:right].mean().item()) if right > left else 0.0,
                    merge_count=0,
                    smoothed_target_consistency=consistency,
                )
            )
        return segments

    def _extract_near_only_segments(
        self,
        strict_scores: torch.Tensor,
        near_scores: torch.Tensor,
        target_ids: torch.Tensor,
        valid_len: int,
        *,
        batch_index: int,
        dataset_key: str,
        strict_segments: list[RawSegment],
    ) -> list[RawSegment]:
        segments: list[RawSegment] = []
        occupied = {
            (segment.hand_side_id, frame_idx)
            for segment in strict_segments
            for frame_idx in range(segment.raw_start_frame, segment.raw_end_frame)
        }
        threshold = max(self.config.near_score_threshold_pre, self.config.near_score_threshold_post)
        for hand_id in range(strict_scores.shape[0]):
            frame_idx = 0
            while frame_idx < valid_len:
                if (
                    float(near_scores[hand_id, frame_idx].item()) < threshold
                    or float(strict_scores[hand_id, frame_idx].item()) >= self.config.strict_score_threshold
                    or (hand_id, frame_idx) in occupied
                ):
                    frame_idx += 1
                    continue
                target_id = int(target_ids[hand_id, frame_idx].item())
                start = frame_idx
                end = frame_idx + 1
                while end < valid_len:
                    if (hand_id, end) in occupied:
                        break
                    if int(target_ids[hand_id, end].item()) != target_id:
                        break
                    if float(near_scores[hand_id, end].item()) < threshold:
                        break
                    if float(strict_scores[hand_id, end].item()) >= self.config.strict_score_threshold:
                        break
                    end += 1
                if end - start >= self.config.near_only_len_min:
                    near_slice = near_scores[hand_id, start:end]
                    local_peak = int(torch.argmax(near_slice).item()) if near_slice.numel() > 0 else 0
                    center = start + local_peak
                    consistency = float(
                        (target_ids[hand_id, start:end] == target_id).float().mean().item()
                    )
                    segments.append(
                        RawSegment(
                            batch_index=batch_index,
                            dataset_key=dataset_key,
                            hand_side=HAND_SIDE_NAMES[hand_id],
                            hand_side_id=hand_id,
                            target_part=TARGET_PART_NAMES[target_id],
                            target_part_id=target_id,
                            window_state="near",
                            window_state_id=WINDOW_STATE_IDS["near"],
                            anchor_start_frame=start,
                            anchor_end_frame=end,
                            raw_start_frame=start,
                            raw_end_frame=end,
                            center_frame=center,
                            raw_length=end - start,
                            strict_peak_score=float(strict_scores[hand_id, start:end].max().item()),
                            near_mean_score=float(near_slice.mean().item()),
                            merge_count=0,
                            smoothed_target_consistency=consistency,
                        )
                    )
                frame_idx = end
        return segments

    def _merge_segments(self, segments: list[RawSegment]) -> list[RawSegment]:
        if not segments:
            return []
        segments = sorted(
            segments,
            key=lambda item: (
                item.hand_side_id,
                item.target_part_id,
                item.window_state_id,
                item.raw_start_frame,
                item.raw_end_frame,
            ),
        )
        merged: list[RawSegment] = [segments[0]]
        for current in segments[1:]:
            previous = merged[-1]
            same_group = (
                previous.hand_side_id == current.hand_side_id
                and previous.target_part_id == current.target_part_id
                and previous.window_state_id == current.window_state_id
            )
            gap = current.raw_start_frame - previous.raw_end_frame
            if same_group and gap <= self.config.gap_merge:
                merged[-1] = RawSegment(
                    batch_index=previous.batch_index,
                    dataset_key=previous.dataset_key,
                    hand_side=previous.hand_side,
                    hand_side_id=previous.hand_side_id,
                    target_part=previous.target_part,
                    target_part_id=previous.target_part_id,
                    window_state=previous.window_state,
                    window_state_id=previous.window_state_id,
                    anchor_start_frame=min(previous.anchor_start_frame, current.anchor_start_frame),
                    anchor_end_frame=max(previous.anchor_end_frame, current.anchor_end_frame),
                    raw_start_frame=min(previous.raw_start_frame, current.raw_start_frame),
                    raw_end_frame=max(previous.raw_end_frame, current.raw_end_frame),
                    center_frame=previous.center_frame
                    if previous.strict_peak_score >= current.strict_peak_score
                    else current.center_frame,
                    raw_length=max(previous.raw_end_frame, current.raw_end_frame)
                    - min(previous.raw_start_frame, current.raw_start_frame),
                    strict_peak_score=max(previous.strict_peak_score, current.strict_peak_score),
                    near_mean_score=max(previous.near_mean_score, current.near_mean_score),
                    merge_count=previous.merge_count + current.merge_count + 1,
                    smoothed_target_consistency=max(
                        previous.smoothed_target_consistency,
                        current.smoothed_target_consistency,
                    ),
                )
            else:
                merged.append(current)
        return merged

    def _filter_segments(self, segments: list[RawSegment]) -> list[RawSegment]:
        filtered = []
        for segment in segments:
            if segment.smoothed_target_consistency < 0.5:
                continue
            if segment.raw_length < self.config.raw_L_min:
                continue
            filtered.append(segment)
        return filtered

    def _segment_to_model_window(
        self,
        segment: RawSegment,
        valid_len: int,
    ) -> dict[str, Any]:
        center = max(0, min(int(segment.center_frame), max(valid_len - 1, 0)))
        if valid_len >= self.config.model_W:
            start = center - self.config.model_W // 2
            start = max(0, min(start, valid_len - self.config.model_W))
            end = start + self.config.model_W
        else:
            start = 0
            end = valid_len
        window = asdict(segment)
        window.update(
            {
                "start_frame": int(start),
                "end_frame": int(end),
                "center_frame": int(center),
                "model_window_size": int(self.config.model_W),
            }
        )
        return window

    def _limit_windows_per_hand_and_seq(
        self,
        window_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        def sort_key(item: dict[str, Any]):
            state_priority = 1.0 if item["window_state"] == "strict" else self.config.near_state_priority_bias
            return (
                state_priority,
                float(item.get("strict_peak_score", 0.0)),
                float(item.get("smoothed_target_consistency", 0.0)),
                float(item.get("near_mean_score", 0.0)),
                float(item.get("raw_length", 0)),
            )

        by_batch: dict[int, list[dict[str, Any]]] = {}
        for item in window_items:
            by_batch.setdefault(int(item["batch_index"]), []).append(item)

        limited: list[dict[str, Any]] = []
        for batch_index, items in by_batch.items():
            hand_kept: list[dict[str, Any]] = []
            for hand_id in range(len(HAND_SIDE_NAMES)):
                hand_items = [item for item in items if int(item["hand_side_id"]) == hand_id]
                hand_items = sorted(hand_items, key=sort_key, reverse=True)[: self.config.per_hand_max_windows]
                hand_kept.extend(hand_items)
            hand_kept = sorted(hand_kept, key=sort_key, reverse=True)[: self.config.per_seq_max_windows]
            limited.extend(sorted(hand_kept, key=lambda item: (item["batch_index"], item["start_frame"], item["hand_side_id"])))
        return limited

    def build_windows_for_batch(
        self,
        actor_motion: torch.Tensor,
        coarse_motion: torch.Tensor,
        lengths: torch.Tensor,
        restoration_meta,
        dataset_keys=None,
    ) -> dict[str, Any]:
        lengths = lengths.long()
        actor_motion, coarse_motion, restoration_meta = self._restore_pair_if_needed(
            actor_motion,
            coarse_motion,
            restoration_meta,
        )
        meta_space = _normalize_meta_scalar(
            restoration_meta.get("space_definition", RESTORED_PAIR_SPACE),
            default=RESTORED_PAIR_SPACE,
        )
        if meta_space and meta_space != RESTORED_PAIR_SPACE:
            raise ValueError(
                f"Window selector requires restored pair space, got '{meta_space}'."
            )

        actor_xyz = self._motions_to_xyz(actor_motion, lengths)
        coarse_xyz = self._motions_to_xyz(coarse_motion, lengths)

        meta_dataset_keys = restoration_meta.get("dataset_key", dataset_keys)
        if meta_dataset_keys is None:
            meta_dataset_keys = [f"sample_{idx}" for idx in range(actor_motion.shape[0])]

        window_items: list[dict[str, Any]] = []
        debug_items: list[dict[str, Any]] = []

        for batch_index in range(actor_motion.shape[0]):
            valid_len = int(lengths[batch_index].item())
            if valid_len <= 0:
                debug_items.append(
                    {
                        "batch_index": batch_index,
                        "dataset_key": f"sample_{batch_index}",
                        "window_items": [],
                    }
                )
                continue

            dataset_key = meta_dataset_keys[batch_index]
            if isinstance(dataset_key, bytes):
                dataset_key = dataset_key.decode("utf-8")
            dataset_key = str(dataset_key)

            sample_actor_xyz = actor_xyz[batch_index, :55, :, :valid_len]
            sample_coarse_xyz = coarse_xyz[batch_index, :55, :, :valid_len]

            candidate_scores = self._compute_candidate_target_scores(
                sample_actor_xyz,
                sample_coarse_xyz,
                valid_len,
            )
            raw_target_ids = torch.argmax(candidate_scores["selection_score"], dim=1)
            smoothed_target_ids = self._smooth_target_ids(
                raw_target_ids,
                candidate_scores["selection_score"],
                valid_len,
            )
            frame_scores = self._compute_strict_near_scores(
                candidate_scores,
                smoothed_target_ids,
                valid_len,
            )
            strict_anchors = self._extract_strict_anchors(
                frame_scores["strict_scores"],
                smoothed_target_ids,
                valid_len,
            )
            strict_segments = self._grow_segments_from_anchors(
                strict_anchors,
                frame_scores["strict_scores"],
                frame_scores["near_scores"],
                smoothed_target_ids,
                valid_len,
                batch_index=batch_index,
                dataset_key=dataset_key,
            )
            near_segments = self._extract_near_only_segments(
                frame_scores["strict_scores"],
                frame_scores["near_scores"],
                smoothed_target_ids,
                valid_len,
                batch_index=batch_index,
                dataset_key=dataset_key,
                strict_segments=strict_segments,
            )
            raw_segments = self._merge_segments(strict_segments + near_segments)
            raw_segments = self._filter_segments(raw_segments)
            model_windows = [
                self._segment_to_model_window(segment, valid_len)
                for segment in raw_segments
            ]
            model_windows = self._limit_windows_per_hand_and_seq(model_windows)

            window_items.extend(model_windows)
            debug_items.append(
                {
                    "batch_index": batch_index,
                    "dataset_key": dataset_key,
                    "raw_target_ids": _to_numpy(raw_target_ids),
                    "smoothed_target_ids": _to_numpy(smoothed_target_ids),
                    "strict_scores": _to_numpy(frame_scores["strict_scores"]),
                    "near_scores": _to_numpy(frame_scores["near_scores"]),
                    "target_dists": _to_numpy(candidate_scores["target_dists"]),
                    "selection_score": _to_numpy(candidate_scores["selection_score"]),
                    "strict_anchors": strict_anchors,
                    "raw_segments": [asdict(segment) for segment in raw_segments],
                    "window_items": model_windows,
                }
            )

        return {
            "window_items": window_items,
            "debug": debug_items,
        }
