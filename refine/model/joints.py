"""Lightweight joint-scope helpers for Stage2-lite joint-based modules.

This module only defines the hand-centric local joint scope used by the
joint-based baseline. The support scope is intentionally limited to:

- elbow / shoulder
- upper torso

It does not include translation-only entries or a root/transl slot.
"""

from __future__ import annotations

from dataclasses import dataclass


JOINT_ROLE_NAMES = ("core", "support", "stabilize")
JOINT_ROLE_IDS = {name: idx for idx, name in enumerate(JOINT_ROLE_NAMES)}

LEFT_WRIST_ID = 20
RIGHT_WRIST_ID = 21

LEFT_HAND_IDS = (25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39)
RIGHT_HAND_IDS = (40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54)

LEFT_SUPPORT_IDS = (18, 16, 3, 6, 9, 12)
RIGHT_SUPPORT_IDS = (19, 17, 3, 6, 9, 12)

TARGET_PART_TO_JOINT_IDS = {
    "torso_head": (0, 3, 6, 9, 12, 15, 22, 23, 24),
    "lower_body": (1, 2, 4, 5, 7, 8, 10, 11),
    "left_arm": (13, 16, 18, 20),
    "right_arm": (14, 17, 19, 21),
    "left_hand": LEFT_HAND_IDS,
    "right_hand": RIGHT_HAND_IDS,
}
TARGET_PART_NAMES = tuple(TARGET_PART_TO_JOINT_IDS.keys())
TARGET_PART_IDS = {name: idx for idx, name in enumerate(TARGET_PART_NAMES)}
MAX_TARGET_JOINTS = max(len(ids) for ids in TARGET_PART_TO_JOINT_IDS.values())


@dataclass(frozen=True)
class JointScope:
    hand_side: str
    source_joint_ids: tuple[int, ...]
    joint_role_ids: tuple[int, ...]

    @property
    def core_size(self) -> int:
        return sum(1 for role_id in self.joint_role_ids if role_id == JOINT_ROLE_IDS["core"])

    @property
    def support_size(self) -> int:
        return sum(1 for role_id in self.joint_role_ids if role_id == JOINT_ROLE_IDS["support"])

    @property
    def stabilize_size(self) -> int:
        return sum(1 for role_id in self.joint_role_ids if role_id == JOINT_ROLE_IDS["stabilize"])


def get_hand_joint_scope(hand_side: str) -> JointScope:
    """Return the hand-centric local joint scope for one hand.

    `source_joint_ids` are original SMPL-X joint ids in global indexing.
    Downstream modules may build `local_joint_ids = range(len(source_joint_ids))`
    for indexing inside the cropped local tensor.
    """

    hand_side = str(hand_side).strip().lower()
    if hand_side == "left":
        source_joint_ids = (LEFT_WRIST_ID,) + LEFT_HAND_IDS + LEFT_SUPPORT_IDS
    elif hand_side == "right":
        source_joint_ids = (RIGHT_WRIST_ID,) + RIGHT_HAND_IDS + RIGHT_SUPPORT_IDS
    else:
        raise ValueError(f"Unsupported hand_side: {hand_side}")

    joint_role_ids = (
        (JOINT_ROLE_IDS["core"],)
        + (JOINT_ROLE_IDS["core"],) * (len(LEFT_HAND_IDS))
        + (JOINT_ROLE_IDS["support"],) * len(LEFT_SUPPORT_IDS)
    )
    return JointScope(
        hand_side=hand_side,
        source_joint_ids=source_joint_ids,
        joint_role_ids=joint_role_ids,
    )


def get_target_joint_ids(target_part: str) -> tuple[int, ...]:
    target_part = str(target_part).strip()
    if target_part not in TARGET_PART_TO_JOINT_IDS:
        raise KeyError(f"Unsupported target_part: {target_part}")
    return TARGET_PART_TO_JOINT_IDS[target_part]
