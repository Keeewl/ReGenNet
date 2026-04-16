HAND_SIDES = ("left", "right")

PART_JOINT_IDS = {
    "torso_head": [0, 3, 6, 9, 12, 15, 22, 23, 24],
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "left_hand": [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    "right_hand": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
}

WRIST_JOINT_IDS = {"left": 20, "right": 21}
HAND_JOINT_IDS = {
    "left": PART_JOINT_IDS["left_hand"],
    "right": PART_JOINT_IDS["right_hand"],
}

FINGER_BASE_IDS = {
    "left": [25, 28, 31, 34, 37],
    "right": [40, 43, 46, 49, 52],
}
FINGER_TIP_IDS = {
    "left": [27, 30, 33, 36, 39],
    "right": [42, 45, 48, 51, 54],
}

ACTOR_PART_NAMES = [
    "actor_left_hand",
    "actor_right_hand",
    "actor_left_arm",
    "actor_right_arm",
    "actor_torso_head",
]

ACTOR_PART_JOINT_IDS = {
    "actor_left_hand": PART_JOINT_IDS["left_hand"],
    "actor_right_hand": PART_JOINT_IDS["right_hand"],
    "actor_left_arm": PART_JOINT_IDS["left_arm"],
    "actor_right_arm": PART_JOINT_IDS["right_arm"],
    "actor_torso_head": PART_JOINT_IDS["torso_head"],
}

TARGET_PARTS = [
    "none",
    "actor_left_hand",
    "actor_right_hand",
    "actor_left_arm",
    "actor_right_arm",
    "actor_torso_head",
]
TARGET_PART_IDS = {name: idx for idx, name in enumerate(TARGET_PARTS)}

BAND_IDS = {
    "far": 0,
    "near": 1,
    "contact": 2,
}

PHASE_IDS = {
    "idle": 0,
    "approach": 1,
    "hold": 2,
    "release": 3,
}

BUFFER_JOINT_IDS = [18, 19]

CORE_CONTACT_JOINT_IDS = {
    "left": [WRIST_JOINT_IDS["left"]] + PART_JOINT_IDS["left_hand"],
    "right": [WRIST_JOINT_IDS["right"]] + PART_JOINT_IDS["right_hand"],
}

# Hand-centric refinement remains local, but elbow/shoulder/upper-torso/root
# can move slightly to support contact formation when hand-only edits are not enough.
SUPPORT_CONTACT_JOINT_IDS = {
    "left": [16, 18, 3, 6, 9, 55],
    "right": [17, 19, 3, 6, 9, 55],
}

HAND_CENTRIC_SHARED_JOINT_IDS = [3, 6, 9, 55]


def default_refiner_joint_ids(include_buffer=False):
    return hand_centric_joint_ids(include_buffer=include_buffer)


def core_contact_joint_ids(side=None):
    if side is None:
        return {name: list(ids) for name, ids in CORE_CONTACT_JOINT_IDS.items()}
    return list(CORE_CONTACT_JOINT_IDS[str(side)])


def support_contact_joint_ids(side=None, include_buffer=False):
    if side is None:
        out = {name: list(ids) for name, ids in SUPPORT_CONTACT_JOINT_IDS.items()}
        if include_buffer:
            for name in out:
                out[name] = sorted(set(out[name] + BUFFER_JOINT_IDS))
        return out
    ids = list(SUPPORT_CONTACT_JOINT_IDS[str(side)])
    if include_buffer:
        ids = sorted(set(ids + BUFFER_JOINT_IDS))
    return ids


def hand_centric_joint_ids(include_buffer=False):
    joint_ids = []
    for side in HAND_SIDES:
        joint_ids.extend(core_contact_joint_ids(side))
        joint_ids.extend(support_contact_joint_ids(side, include_buffer=include_buffer))
    return sorted(set(int(jid) for jid in joint_ids))


def joint_scope_masks(joint_ids, side, include_buffer=False):
    joint_ids = [int(jid) for jid in joint_ids]
    core = set(core_contact_joint_ids(side))
    support = set(support_contact_joint_ids(side, include_buffer=include_buffer)) - core
    core_mask = [jid in core for jid in joint_ids]
    support_mask = [jid in support for jid in joint_ids]
    stabilize_mask = [not (c or s) for c, s in zip(core_mask, support_mask)]
    return {
        "core": core_mask,
        "support": support_mask,
        "stabilize": stabilize_mask,
    }
