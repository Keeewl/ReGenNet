"""Joint-group utilities for scope-aware refine_v2 residuals and losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch


LOWER_BODY_IDS = (1, 2, 4, 5, 7, 8, 10, 11)
LEFT_ARM_IDS = (13, 16, 18, 20)
RIGHT_ARM_IDS = (14, 17, 19, 21)
LEFT_HAND_IDS = tuple(range(25, 40))
RIGHT_HAND_IDS = tuple(range(40, 55))
ROOT_IDS = (0,)
TRANSL_INDEX = 55
TORSO_ROOT_IDS = (0, 3, 6, 9, 12, 15, 22, 23, 24)


@dataclass
class ResidualGroupScales:
    hand: float = 1.0
    arm: float = 1.0
    torso: float = 0.5
    root: float = 0.2
    transl: float = 0.2
    lower_body: float = 0.1


@dataclass
class MotionGroupWeights:
    selected_hand: float = 3.0
    same_side_arm: float = 2.0
    other_hand_arm: float = 1.0
    torso_root: float = 0.75
    lower_body: float = 0.25
    transl: float = 0.25


@dataclass
class ContactGroupWeights:
    selected_hand: float = 4.0
    same_side_arm: float = 3.0
    other_upper: float = 1.0
    body: float = 0.5


@dataclass
class PhasePreserveGroupWeights:
    hand: float = 0.05
    arm: float = 0.1
    torso: float = 0.3
    root: float = 1.0
    transl: float = 2.0
    lower_body: float = 0.5


def _valid_ids(ids, num_joints: int) -> list[int]:
    return [int(idx) for idx in ids if 0 <= int(idx) < int(num_joints)]


def side_group_ids(hand_side_id: int) -> dict[str, list[int]]:
    hand_side_id = int(hand_side_id)
    if hand_side_id == 0:
        return {
            "selected_hand": list(LEFT_HAND_IDS),
            "same_side_arm": list(LEFT_ARM_IDS),
            "other_hand": list(RIGHT_HAND_IDS),
            "other_arm": list(RIGHT_ARM_IDS),
        }
    return {
        "selected_hand": list(RIGHT_HAND_IDS),
        "same_side_arm": list(RIGHT_ARM_IDS),
        "other_hand": list(LEFT_HAND_IDS),
        "other_arm": list(LEFT_ARM_IDS),
    }


def residual_scale_tensor(
    *,
    num_joints: int,
    num_channels: int,
    device,
    dtype,
    scales: ResidualGroupScales,
) -> torch.Tensor:
    scale = torch.full((num_joints, num_channels), float(scales.torso), device=device, dtype=dtype)
    for idx in _valid_ids(LOWER_BODY_IDS, num_joints):
        scale[idx, :] = float(scales.lower_body)
    for idx in _valid_ids(TORSO_ROOT_IDS, num_joints):
        scale[idx, :] = float(scales.torso)
    for idx in _valid_ids(ROOT_IDS, num_joints):
        scale[idx, :] = float(scales.root)
    for idx in _valid_ids(LEFT_ARM_IDS + RIGHT_ARM_IDS, num_joints):
        scale[idx, :] = float(scales.arm)
    for idx in _valid_ids(LEFT_HAND_IDS + RIGHT_HAND_IDS, num_joints):
        scale[idx, :] = float(scales.hand)
    if 0 <= TRANSL_INDEX < num_joints:
        scale[TRANSL_INDEX, :] = float(scales.transl)
    return scale.view(1, num_joints, num_channels, 1)


def group_weight_tensor(
    *,
    hand_side_id: torch.Tensor,
    num_joints: int,
    num_channels: int,
    num_frames: int,
    device,
    dtype,
    weights: MotionGroupWeights,
) -> torch.Tensor:
    batch_size = int(hand_side_id.shape[0])
    out = torch.full(
        (batch_size, num_joints, num_channels, num_frames),
        float(weights.torso_root),
        device=device,
        dtype=dtype,
    )
    lower_ids = _valid_ids(LOWER_BODY_IDS, num_joints)
    if lower_ids:
        out[:, lower_ids, :, :] = float(weights.lower_body)
    root_ids = _valid_ids(ROOT_IDS, num_joints)
    if root_ids:
        out[:, root_ids, :, :] = float(weights.torso_root)
    if 0 <= TRANSL_INDEX < num_joints:
        out[:, TRANSL_INDEX, :, :] = float(weights.transl)
    hand_side_cpu = hand_side_id.detach().cpu().long().view(-1).tolist()
    for b, side in enumerate(hand_side_cpu):
        groups = side_group_ids(int(side))
        selected_hand = _valid_ids(groups["selected_hand"], num_joints)
        same_arm = _valid_ids(groups["same_side_arm"], num_joints)
        other = _valid_ids(groups["other_hand"] + groups["other_arm"], num_joints)
        if other:
            out[b, other, :, :] = float(weights.other_hand_arm)
        if same_arm:
            out[b, same_arm, :, :] = float(weights.same_side_arm)
        if selected_hand:
            out[b, selected_hand, :, :] = float(weights.selected_hand)
    return out


def contact_group_weight_tensor(
    *,
    hand_side_id: torch.Tensor,
    num_joints: int,
    num_channels: int,
    num_frames: int,
    device,
    dtype,
    weights: ContactGroupWeights,
) -> torch.Tensor:
    batch_size = int(hand_side_id.shape[0])
    out = torch.full(
        (batch_size, num_joints, num_channels, num_frames),
        float(weights.body),
        device=device,
        dtype=dtype,
    )
    hand_side_cpu = hand_side_id.detach().cpu().long().view(-1).tolist()
    for b, side in enumerate(hand_side_cpu):
        groups = side_group_ids(int(side))
        selected_hand = _valid_ids(groups["selected_hand"], num_joints)
        same_arm = _valid_ids(groups["same_side_arm"], num_joints)
        other = _valid_ids(groups["other_hand"] + groups["other_arm"], num_joints)
        if other:
            out[b, other, :, :] = float(weights.other_upper)
        if same_arm:
            out[b, same_arm, :, :] = float(weights.same_side_arm)
        if selected_hand:
            out[b, selected_hand, :, :] = float(weights.selected_hand)
    return out


def phase_preserve_group_weight_tensor(
    *,
    num_joints: int,
    num_channels: int,
    num_frames: int,
    device,
    dtype,
    weights: PhasePreserveGroupWeights,
) -> torch.Tensor:
    out = torch.full(
        (1, num_joints, num_channels, num_frames),
        float(weights.torso),
        device=device,
        dtype=dtype,
    )
    lower_ids = _valid_ids(LOWER_BODY_IDS, num_joints)
    if lower_ids:
        out[:, lower_ids, :, :] = float(weights.lower_body)
    root_ids = _valid_ids(ROOT_IDS, num_joints)
    if root_ids:
        out[:, root_ids, :, :] = float(weights.root)
    arm_ids = _valid_ids(LEFT_ARM_IDS + RIGHT_ARM_IDS, num_joints)
    if arm_ids:
        out[:, arm_ids, :, :] = float(weights.arm)
    hand_ids = _valid_ids(LEFT_HAND_IDS + RIGHT_HAND_IDS, num_joints)
    if hand_ids:
        out[:, hand_ids, :, :] = float(weights.hand)
    if 0 <= TRANSL_INDEX < num_joints:
        out[:, TRANSL_INDEX, :, :] = float(weights.transl)
    return out


def _mean_abs_group(delta: torch.Tensor, valid_mask: torch.Tensor, ids: list[int], channels: slice | None = None) -> torch.Tensor:
    if not ids:
        return delta.new_zeros((delta.shape[0],))
    x = delta[:, ids, :, :] if channels is None else delta[:, ids, channels, :]
    weights = valid_mask.float().view(valid_mask.shape[0], 1, 1, valid_mask.shape[1]).expand_as(x)
    return (torch.abs(x) * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp_min(1e-8)


def delta_group_norms(delta: torch.Tensor, valid_mask: torch.Tensor, hand_side_id: torch.Tensor) -> dict[str, torch.Tensor]:
    num_joints = int(delta.shape[1])
    batch_size = int(delta.shape[0])
    selected_hand = []
    same_arm = []
    other_hand_arm = []
    for b, side in enumerate(hand_side_id.detach().cpu().long().view(-1).tolist()):
        groups = side_group_ids(int(side))
        selected_hand.append(_mean_abs_group(delta[b : b + 1], valid_mask[b : b + 1], _valid_ids(groups["selected_hand"], num_joints))[0])
        same_arm.append(_mean_abs_group(delta[b : b + 1], valid_mask[b : b + 1], _valid_ids(groups["same_side_arm"], num_joints))[0])
        other_ids = _valid_ids(groups["other_hand"] + groups["other_arm"], num_joints)
        other_hand_arm.append(_mean_abs_group(delta[b : b + 1], valid_mask[b : b + 1], other_ids)[0])
    transl = (
        _mean_abs_group(delta, valid_mask, [TRANSL_INDEX], slice(0, 3))
        if 0 <= TRANSL_INDEX < num_joints
        else delta.new_zeros((batch_size,))
    )
    return {
        "delta_norm_selected_hand": torch.stack(selected_hand) if selected_hand else delta.new_zeros((batch_size,)),
        "delta_norm_same_side_arm": torch.stack(same_arm) if same_arm else delta.new_zeros((batch_size,)),
        "delta_norm_other_hand_arm": torch.stack(other_hand_arm) if other_hand_arm else delta.new_zeros((batch_size,)),
        "delta_norm_torso_root": _mean_abs_group(delta, valid_mask, _valid_ids(TORSO_ROOT_IDS, num_joints)),
        "delta_norm_lower_body": _mean_abs_group(delta, valid_mask, _valid_ids(LOWER_BODY_IDS, num_joints)),
        "delta_norm_transl": transl,
    }
