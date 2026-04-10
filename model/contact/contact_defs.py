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


def default_refiner_joint_ids(include_buffer=False):
    joint_ids = [WRIST_JOINT_IDS["left"], WRIST_JOINT_IDS["right"]]
    joint_ids += PART_JOINT_IDS["left_hand"] + PART_JOINT_IDS["right_hand"]
    if include_buffer:
        joint_ids += BUFFER_JOINT_IDS
    return joint_ids
