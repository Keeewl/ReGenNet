import functools
import math

import torch

from model.contact.contact_defs import (
    FINGER_BASE_IDS,
    FINGER_TIP_IDS,
    HAND_SIDES,
    PART_JOINT_IDS,
)
from model.rotation2xyz import Rotation2xyz_x


REACTOR_PATCH_NAMES = [
    "fingertip_0",
    "fingertip_1",
    "fingertip_2",
    "fingertip_3",
    "fingertip_4",
    "palm",
    "knuckle",
]

ACTOR_TORSO_SUBPATCHES = ["torso_front_upper", "head_lower_front"]

WINDOW_STATE_NAMES = ("strict", "near")
WINDOW_STATE_IDS = {name: idx for idx, name in enumerate(WINDOW_STATE_NAMES)}


def _identity_rot6d(num_joints, device):
    rot6d = torch.zeros(num_joints, 6, device=device)
    rot6d[:, 0] = 1.0
    rot6d[:, 3] = 1.0
    return rot6d


def _build_rest_pose_vertices(body_model="smplx", pose_rep="rot6d", device="cpu"):
    if body_model != "smplx":
        raise ValueError("mesh regions currently support smplx only")
    num_joints = 56
    rot2xyz = Rotation2xyz_x(device=device)
    motion = torch.zeros(1, num_joints, 6, 1, device=device)
    motion[:, :-1, :, 0] = _identity_rot6d(num_joints - 1, device=device)
    motion[:, -1, :3, 0] = 0.0
    mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    verts = rot2xyz(
        x=motion,
        mask=mask,
        pose_rep=pose_rep,
        translation=True,
        glob=True,
        jointstype="vertices",
        vertstrans=True,
        num_person=1,
    )
    joints = rot2xyz(
        x=motion,
        mask=mask,
        pose_rep=pose_rep,
        translation=True,
        glob=True,
        jointstype=body_model,
        vertstrans=True,
        num_person=1,
    )
    verts = verts[0, :, :, 0].contiguous()
    joints = joints[0, :, :, 0].contiguous()
    return verts, joints


def _nearest_vertices(vertices, joint_xyz, k):
    dist = torch.linalg.norm(vertices - joint_xyz, dim=-1)
    k = min(int(k), dist.numel())
    return torch.topk(dist, k=k, largest=False).indices


def _select_vertices_for_joints(vertices, joints, joint_ids, k):
    if len(joint_ids) == 0:
        return []
    ids = []
    for jid in joint_ids:
        ids.append(_nearest_vertices(vertices, joints[int(jid)], k))
    ids = torch.cat(ids, dim=0).unique()
    return ids.detach().cpu().tolist()


def _density_k(density, base):
    if density == "small":
        return max(6, int(math.ceil(base * 0.6)))
    return int(base)


class MeshRegionProvider:
    """
    Build sparse mesh patch vertex ids for reactor/actor contact regions.
    """

    def __init__(self, density="medium", body_model="smplx", pose_rep="rot6d", device="cpu"):
        self.density = density
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.device = device
        self._build_regions()

    def _build_regions(self):
        verts, joints = _build_rest_pose_vertices(
            body_model=self.body_model,
            pose_rep=self.pose_rep,
            device=self.device,
        )
        self._vertices = verts
        self._joints = joints

        reactor_patches = {side: {} for side in HAND_SIDES}
        for side in HAND_SIDES:
            tip_ids = FINGER_TIP_IDS[side]
            base_ids = FINGER_BASE_IDS[side]
            for idx, jid in enumerate(tip_ids):
                k = _density_k(self.density, 18)
                reactor_patches[side][f"fingertip_{idx}"] = _select_vertices_for_joints(
                    verts, joints, [jid], k
                )
            palm_joints = [base_ids[0], base_ids[1], base_ids[2], base_ids[3], base_ids[4]]
            palm_k = _density_k(self.density, 28)
            reactor_patches[side]["palm"] = _select_vertices_for_joints(
                verts, joints, palm_joints, palm_k
            )
            knuckle_k = _density_k(self.density, 22)
            reactor_patches[side]["knuckle"] = _select_vertices_for_joints(
                verts, joints, base_ids, knuckle_k
            )
        self.reactor_patches = reactor_patches

        actor_parts = {}
        actor_parts["actor_left_hand"] = {
            "actor_left_hand": _select_vertices_for_joints(
                verts, joints, PART_JOINT_IDS["left_hand"], _density_k(self.density, 24)
            )
        }
        actor_parts["actor_right_hand"] = {
            "actor_right_hand": _select_vertices_for_joints(
                verts, joints, PART_JOINT_IDS["right_hand"], _density_k(self.density, 24)
            )
        }
        actor_parts["actor_left_arm"] = {
            "actor_left_arm": _select_vertices_for_joints(
                verts, joints, [18, 20], _density_k(self.density, 28)
            )
        }
        actor_parts["actor_right_arm"] = {
            "actor_right_arm": _select_vertices_for_joints(
                verts, joints, [19, 21], _density_k(self.density, 28)
            )
        }
        torso_ids = [3, 6, 9, 12, 15]
        head_ids = [22, 23, 24]
        actor_parts["actor_torso_head"] = {
            "torso_front_upper": _select_vertices_for_joints(
                verts, joints, torso_ids, _density_k(self.density, 32)
            ),
            "head_lower_front": _select_vertices_for_joints(
                verts, joints, head_ids, _density_k(self.density, 24)
            ),
        }
        self.actor_parts = actor_parts

    def reactor_hand_patch_ids(self, side):
        return self.reactor_patches.get(side, {})

    def actor_target_patch_ids(self, target_part):
        return self.actor_parts.get(target_part, {})

    def actor_nontarget_patch_ids(self, target_part):
        ids = []
        for part_name, subpatches in self.actor_parts.items():
            if part_name == target_part:
                continue
            for patch_ids in subpatches.values():
                ids.extend(patch_ids)
        if not ids:
            return []
        return sorted(set(ids))


@functools.lru_cache(maxsize=8)
def get_mesh_region_provider(density="medium", body_model="smplx", pose_rep="rot6d"):
    provider = MeshRegionProvider(
        density=density,
        body_model=body_model,
        pose_rep=pose_rep,
        device="cpu",
    )
    return provider
