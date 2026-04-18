"""Joint-based local feature builder for Stage2-lite.

This module is the current joint-based baseline between:

- deterministic window selection
- future local refiner network / losses

It does not build mesh tokens or proposal-style feature stacks. Its job is
limited to cropping hand-centric local motion windows and packaging the minimum
joint-based relation summary required by later Stage2-lite modules.

Field naming convention:

- `local_joint_ids`: contiguous indices inside the local cropped tensor
- `source_joint_ids`: original SMPL-X joint ids used to form that local crop
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from refine.data.restored_space import (
    REQUIRED_RESTORATION_METADATA_FIELDS,
    RESTORED_PAIR_SPACE,
    extract_restoration_metadata,
    restore_pair_batch,
    validate_restoration_metadata,
)
from refine.data.schema import normalize_space_definition
from refine.model.joints import (
    JOINT_ROLE_IDS,
    MAX_TARGET_JOINTS,
    TARGET_PART_NAMES,
    get_hand_joint_scope,
    get_target_joint_ids,
)


@dataclass(frozen=True)
class FeatureBuilderConfig:
    model_window_size: int = 16
    target_summary_feature_dim: int = 17


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return (
        torch.arange(max_len, device=lengths.device).unsqueeze(0)
        < lengths.view(-1, 1)
    )


def _normalize_meta_scalar(value: Any, default: str = "") -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    return normalize_space_definition(value, default=default)


class JointFeatureBuilder:
    """Build hand-centric joint-based local windows for later Stage2-lite modules.

    The output keeps both:

    - `local_joint_ids`: local tensor indexing used after cropping
    - `source_joint_ids`: original global SMPL-X joint ids per window
    """

    def __init__(
        self,
        config: FeatureBuilderConfig | None = None,
        *,
        body_model: str = "smplx",
        pose_rep: str = "rot6d",
    ):
        self.config = config or FeatureBuilderConfig()
        self.body_model = str(body_model)
        self.pose_rep = str(pose_rep)
        self._rot2xyz_cache: dict[str, Any] = {}

    def _get_rot2xyz(self, device: torch.device):
        key = f"{self.body_model}:{device.type}:{device.index}"
        if key in self._rot2xyz_cache:
            return self._rot2xyz_cache[key]
        if self.body_model != "smplx":
            raise ValueError(
                f"JointFeatureBuilder currently expects body_model='smplx', got {self.body_model}."
            )
        from model.rotation2xyz import Rotation2xyz_x

        rot2xyz = Rotation2xyz_x(device=str(device), dataset="interx")
        self._rot2xyz_cache[key] = rot2xyz
        return rot2xyz

    def _ensure_restored_pair(
        self,
        actor_motion: torch.Tensor,
        coarse_motion: torch.Tensor,
        gt_motion: torch.Tensor | None,
        restoration_meta,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        if restoration_meta is None:
            raise ValueError(
                "JointFeatureBuilder requires restoration metadata and restored pair space."
            )
        if not all(key in restoration_meta for key in REQUIRED_RESTORATION_METADATA_FIELDS):
            restoration_meta = extract_restoration_metadata(
                restoration_meta,
                device=actor_motion.device,
            )
        else:
            validate_restoration_metadata(restoration_meta, context="feature builder restoration metadata")

        actor_motion, coarse_motion = restore_pair_batch(
            actor_motion,
            coarse_motion,
            restoration_meta,
        )
        if gt_motion is not None:
            _, gt_motion = restore_pair_batch(actor_motion, gt_motion, restoration_meta)

        meta_space = _normalize_meta_scalar(
            restoration_meta.get("space_definition", RESTORED_PAIR_SPACE),
            default=RESTORED_PAIR_SPACE,
        )
        if meta_space and meta_space != RESTORED_PAIR_SPACE:
            raise ValueError(
                f"JointFeatureBuilder requires restored pair space, got '{meta_space}'."
            )
        return actor_motion, coarse_motion, gt_motion, restoration_meta

    def _motions_to_xyz(
        self,
        motion: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        if motion is None:
            return None
        if motion.dim() != 4:
            raise ValueError("motion must have shape [B, J, F, T].")
        if motion.shape[2] == 3:
            return motion[:, :55]
        rot2xyz = self._get_rot2xyz(motion.device)
        mask = _lengths_to_mask(lengths.to(motion.device), motion.shape[-1]).bool()
        xyz = rot2xyz(
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
        return xyz[:, :55]

    def _get_joint_scope(self, hand_side: str):
        return get_hand_joint_scope(hand_side)

    def _get_target_joint_ids(self, target_part: str) -> tuple[int, ...]:
        return get_target_joint_ids(target_part)

    def _build_time_mask(self, actual_len: int, window_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(window_size, dtype=torch.bool, device=device)
        mask[: max(0, min(actual_len, window_size))] = True
        return mask

    def _crop_motion_window(
        self,
        motion: torch.Tensor,
        batch_index: int,
        joint_ids: tuple[int, ...],
        start_frame: int,
        end_frame: int,
        window_size: int,
    ) -> tuple[torch.Tensor, int]:
        clipped = motion[batch_index, list(joint_ids), :, start_frame:end_frame]
        actual_len = int(clipped.shape[-1])
        output = torch.zeros(
            (len(joint_ids), motion.shape[2], window_size),
            dtype=motion.dtype,
            device=motion.device,
        )
        if actual_len > 0:
            output[:, :, :actual_len] = clipped[:, :, :window_size]
        return output, min(actual_len, window_size)

    def _crop_xyz_window(
        self,
        xyz: torch.Tensor,
        batch_index: int,
        joint_ids: tuple[int, ...],
        start_frame: int,
        end_frame: int,
        window_size: int,
    ) -> tuple[torch.Tensor, int]:
        clipped = xyz[batch_index, list(joint_ids), :, start_frame:end_frame]
        actual_len = int(clipped.shape[-1])
        output = torch.zeros(
            (len(joint_ids), 3, window_size),
            dtype=xyz.dtype,
            device=xyz.device,
        )
        if actual_len > 0:
            output[:, :, :actual_len] = clipped[:, :, :window_size]
        return output, min(actual_len, window_size)

    def _pad_target_joints(
        self,
        actor_motion: torch.Tensor,
        actor_xyz: torch.Tensor,
        batch_index: int,
        target_joint_ids: tuple[int, ...],
        start_frame: int,
        end_frame: int,
        window_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_target_local, actual_len = self._crop_motion_window(
            actor_motion,
            batch_index,
            target_joint_ids,
            start_frame,
            end_frame,
            window_size,
        )
        actor_target_xyz, _ = self._crop_xyz_window(
            actor_xyz,
            batch_index,
            target_joint_ids,
            start_frame,
            end_frame,
            window_size,
        )
        padded_motion = torch.zeros(
            (MAX_TARGET_JOINTS, actor_motion.shape[2], window_size),
            dtype=actor_motion.dtype,
            device=actor_motion.device,
        )
        padded_xyz = torch.zeros(
            (MAX_TARGET_JOINTS, 3, window_size),
            dtype=actor_xyz.dtype,
            device=actor_xyz.device,
        )
        target_mask = torch.zeros(MAX_TARGET_JOINTS, dtype=torch.bool, device=actor_motion.device)
        target_joint_id_tensor = torch.full(
            (MAX_TARGET_JOINTS,),
            -1,
            dtype=torch.long,
            device=actor_motion.device,
        )
        count = len(target_joint_ids)
        padded_motion[:count] = actor_target_local
        padded_xyz[:count] = actor_target_xyz
        target_mask[:count] = True
        target_joint_id_tensor[:count] = torch.as_tensor(target_joint_ids, dtype=torch.long, device=actor_motion.device)
        return padded_motion, padded_xyz, target_mask, target_joint_id_tensor

    def _compute_hand_center(
        self,
        coarse_local_xyz: torch.Tensor,
        core_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(core_mask.any()):
            raise ValueError("JointFeatureBuilder requires at least one core joint.")
        return coarse_local_xyz[core_mask].mean(dim=0)

    def _compute_target_center(
        self,
        actor_target_xyz: torch.Tensor,
        actor_target_mask: torch.Tensor,
    ) -> torch.Tensor:
        return actor_target_xyz[actor_target_mask].mean(dim=0)

    def _build_target_summary_feat(
        self,
        hand_center: torch.Tensor,
        target_center: torch.Tensor,
        coarse_local_xyz: torch.Tensor,
        core_mask: torch.Tensor,
        actor_target_xyz: torch.Tensor,
        actor_target_mask: torch.Tensor,
        time_mask: torch.Tensor,
        target_part_id: int,
        window_state_id: int,
    ) -> torch.Tensor:
        window_size = int(time_mask.shape[0])
        feature_dim = self.config.target_summary_feature_dim
        feat = torch.zeros(
            (window_size, feature_dim),
            dtype=hand_center.dtype,
            device=hand_center.device,
        )
        valid_len = int(time_mask.sum().item())
        if valid_len <= 0:
            return feat

        rel = hand_center[:, :valid_len] - target_center[:, :valid_len]
        dist = rel.norm(dim=0, keepdim=True)

        hand_vel = torch.zeros_like(hand_center[:, :valid_len])
        target_vel = torch.zeros_like(target_center[:, :valid_len])
        if valid_len > 1:
            hand_vel[:, 1:] = hand_center[:, 1:valid_len] - hand_center[:, : valid_len - 1]
            target_vel[:, 1:] = target_center[:, 1:valid_len] - target_center[:, : valid_len - 1]
        rel_vel = hand_vel - target_vel
        rel_speed_norm = rel_vel.norm(dim=0, keepdim=True)

        hand_xyz = coarse_local_xyz[core_mask, :, :valid_len].permute(2, 0, 1).contiguous()
        target_xyz = actor_target_xyz[actor_target_mask, :, :valid_len].permute(2, 0, 1).contiguous()
        min_joint_distance = torch.cdist(hand_xyz, target_xyz).amin(dim=(-1, -2)).view(1, valid_len)

        frame_feat = torch.cat(
            [
                rel.transpose(0, 1),
                dist.transpose(0, 1),
                rel_vel.transpose(0, 1),
                rel_speed_norm.transpose(0, 1),
                min_joint_distance.transpose(0, 1),
            ],
            dim=-1,
        )
        state_one_hot = torch.nn.functional.one_hot(
            torch.tensor(window_state_id, dtype=torch.long, device=feat.device),
            num_classes=2,
        ).to(feat.dtype)
        target_one_hot = torch.nn.functional.one_hot(
            torch.tensor(target_part_id, dtype=torch.long, device=feat.device),
            num_classes=len(TARGET_PART_NAMES),
        ).to(feat.dtype)
        context_feat = torch.cat([state_one_hot, target_one_hot], dim=0).view(1, -1).expand(valid_len, -1)
        feat[:valid_len] = torch.cat([frame_feat, context_feat], dim=-1)
        return feat

    def build_window_batch(
        self,
        actor_motion: torch.Tensor,
        coarse_motion: torch.Tensor,
        gt_motion: torch.Tensor | None,
        lengths: torch.Tensor,
        window_items,
        restoration_meta,
        sample_indices: torch.Tensor | None = None,
    ):
        actor_motion, coarse_motion, gt_motion, restoration_meta = self._ensure_restored_pair(
            actor_motion,
            coarse_motion,
            gt_motion,
            restoration_meta,
        )
        lengths = lengths.long()
        actor_xyz = self._motions_to_xyz(actor_motion, lengths)
        coarse_xyz = self._motions_to_xyz(coarse_motion, lengths)

        if sample_indices is None:
            sample_indices = torch.arange(actor_motion.shape[0], device=actor_motion.device, dtype=torch.long)
        else:
            sample_indices = sample_indices.long().to(actor_motion.device)

        if not window_items:
            scope = self._get_joint_scope("left")
            local_joint_ids = torch.arange(len(scope.source_joint_ids), dtype=torch.long, device=actor_motion.device)
            joint_role_id = torch.as_tensor(scope.joint_role_ids, dtype=torch.long, device=actor_motion.device)
            core_mask = joint_role_id == JOINT_ROLE_IDS["core"]
            support_mask = joint_role_id == JOINT_ROLE_IDS["support"]
            stabilize_mask = joint_role_id == JOINT_ROLE_IDS["stabilize"]
            empty_local = actor_motion.new_zeros((0, len(scope.source_joint_ids), actor_motion.shape[2], self.config.model_window_size))
            empty_target = actor_motion.new_zeros((0, MAX_TARGET_JOINTS, actor_motion.shape[2], self.config.model_window_size))
            empty_target_mask = torch.zeros((0, MAX_TARGET_JOINTS), dtype=torch.bool, device=actor_motion.device)
            empty_time_mask = torch.zeros((0, self.config.model_window_size), dtype=torch.bool, device=actor_motion.device)
            empty_meta = torch.zeros((0,), dtype=torch.long, device=actor_motion.device)
            empty_summary = actor_motion.new_zeros((0, self.config.model_window_size, self.config.target_summary_feature_dim))
            return {
                "coarse_local": empty_local,
                "gt_local": None if gt_motion is None else empty_local.clone(),
                "actor_target_local": empty_target,
                "actor_target_mask": empty_target_mask,
                "actor_target_joint_ids": torch.full((0, MAX_TARGET_JOINTS), -1, dtype=torch.long, device=actor_motion.device),
                "local_joint_ids": local_joint_ids,
                "joint_role_id": joint_role_id,
                "core_mask": core_mask,
                "support_mask": support_mask,
                "stabilize_mask": stabilize_mask,
                "source_joint_ids": torch.zeros((0, len(scope.source_joint_ids)), dtype=torch.long, device=actor_motion.device),
                "target_part_id": empty_meta,
                "window_state_id": empty_meta,
                "time_mask": empty_time_mask,
                "start_frame": empty_meta,
                "end_frame": empty_meta,
                "center_frame": empty_meta,
                "sample_index": empty_meta,
                "hand_side_id": empty_meta,
                "raw_start_frame": empty_meta,
                "raw_end_frame": empty_meta,
                "target_summary_feat": empty_summary,
            }

        first_scope = self._get_joint_scope(window_items[0]["hand_side"])
        local_joint_ids = torch.arange(len(first_scope.source_joint_ids), dtype=torch.long, device=actor_motion.device)
        joint_role_id = torch.as_tensor(first_scope.joint_role_ids, dtype=torch.long, device=actor_motion.device)
        core_mask = joint_role_id == JOINT_ROLE_IDS["core"]
        support_mask = joint_role_id == JOINT_ROLE_IDS["support"]
        stabilize_mask = joint_role_id == JOINT_ROLE_IDS["stabilize"]

        coarse_local_list = []
        gt_local_list = []
        actor_target_local_list = []
        actor_target_mask_list = []
        actor_target_joint_ids_list = []
        source_joint_ids_list = []
        time_mask_list = []
        target_summary_feat_list = []
        target_part_ids = []
        window_state_ids = []
        start_frames = []
        end_frames = []
        center_frames = []
        sample_index_list = []
        hand_side_ids = []
        raw_start_frames = []
        raw_end_frames = []

        for item in window_items:
            batch_index = int(item["batch_index"])
            hand_side = str(item["hand_side"])
            hand_scope = self._get_joint_scope(hand_side)
            target_joint_ids = self._get_target_joint_ids(item["target_part"])
            window_size = int(item.get("model_window_size", self.config.model_window_size))
            start_frame = int(item["start_frame"])
            end_frame = int(item["end_frame"])

            coarse_local, actual_len = self._crop_motion_window(
                coarse_motion,
                batch_index,
                hand_scope.source_joint_ids,
                start_frame,
                end_frame,
                window_size,
            )
            coarse_local_xyz, _ = self._crop_xyz_window(
                coarse_xyz,
                batch_index,
                hand_scope.source_joint_ids,
                start_frame,
                end_frame,
                window_size,
            )
            if gt_motion is not None:
                gt_local, _ = self._crop_motion_window(
                    gt_motion,
                    batch_index,
                    hand_scope.source_joint_ids,
                    start_frame,
                    end_frame,
                    window_size,
                )
            else:
                gt_local = None

            actor_target_local, actor_target_xyz, actor_target_mask, actor_target_joint_id_tensor = self._pad_target_joints(
                actor_motion,
                actor_xyz,
                batch_index,
                target_joint_ids,
                start_frame,
                end_frame,
                window_size,
            )
            time_mask = self._build_time_mask(actual_len, window_size, actor_motion.device)
            hand_center = self._compute_hand_center(coarse_local_xyz, core_mask)
            target_center = self._compute_target_center(actor_target_xyz, actor_target_mask)
            target_summary_feat = self._build_target_summary_feat(
                hand_center,
                target_center,
                coarse_local_xyz,
                core_mask,
                actor_target_xyz,
                actor_target_mask,
                time_mask,
                target_part_id=int(item["target_part_id"]),
                window_state_id=int(item["window_state_id"]),
            )

            coarse_local_list.append(coarse_local)
            if gt_local is not None:
                gt_local_list.append(gt_local)
            actor_target_local_list.append(actor_target_local)
            actor_target_mask_list.append(actor_target_mask)
            actor_target_joint_ids_list.append(actor_target_joint_id_tensor)
            source_joint_ids_list.append(torch.as_tensor(hand_scope.source_joint_ids, dtype=torch.long, device=actor_motion.device))
            time_mask_list.append(time_mask)
            target_summary_feat_list.append(target_summary_feat)
            target_part_ids.append(int(item["target_part_id"]))
            window_state_ids.append(int(item["window_state_id"]))
            start_frames.append(start_frame)
            end_frames.append(end_frame)
            center_frames.append(int(item["center_frame"]))
            sample_index_list.append(int(sample_indices[batch_index].item()))
            hand_side_ids.append(int(item["hand_side_id"]))
            raw_start_frames.append(int(item.get("raw_start_frame", start_frame)))
            raw_end_frames.append(int(item.get("raw_end_frame", end_frame)))

        window_batch = {
            "coarse_local": torch.stack(coarse_local_list, dim=0),
            "gt_local": None if gt_motion is None else torch.stack(gt_local_list, dim=0),
            "actor_target_local": torch.stack(actor_target_local_list, dim=0),
            "actor_target_mask": torch.stack(actor_target_mask_list, dim=0),
            "actor_target_joint_ids": torch.stack(actor_target_joint_ids_list, dim=0),
            "local_joint_ids": local_joint_ids,
            "joint_role_id": joint_role_id,
            "core_mask": core_mask,
            "support_mask": support_mask,
            "stabilize_mask": stabilize_mask,
            "source_joint_ids": torch.stack(source_joint_ids_list, dim=0),
            "target_part_id": torch.as_tensor(target_part_ids, dtype=torch.long, device=actor_motion.device),
            "window_state_id": torch.as_tensor(window_state_ids, dtype=torch.long, device=actor_motion.device),
            "time_mask": torch.stack(time_mask_list, dim=0),
            "start_frame": torch.as_tensor(start_frames, dtype=torch.long, device=actor_motion.device),
            "end_frame": torch.as_tensor(end_frames, dtype=torch.long, device=actor_motion.device),
            "center_frame": torch.as_tensor(center_frames, dtype=torch.long, device=actor_motion.device),
            "sample_index": torch.as_tensor(sample_index_list, dtype=torch.long, device=actor_motion.device),
            "hand_side_id": torch.as_tensor(hand_side_ids, dtype=torch.long, device=actor_motion.device),
            "raw_start_frame": torch.as_tensor(raw_start_frames, dtype=torch.long, device=actor_motion.device),
            "raw_end_frame": torch.as_tensor(raw_end_frames, dtype=torch.long, device=actor_motion.device),
            "target_summary_feat": torch.stack(target_summary_feat_list, dim=0),
        }
        return window_batch
