import torch

from model.contact.contact_geometry import safe_normalize, temporal_diff
from model.crefine.restored_body_model import RestoredBodyModelForward
from model.crefine.mesh_regions import REACTOR_PATCH_NAMES, get_mesh_region_provider


PATCH_TYPE_IDS = {
    "reactor_fingertip_0": 0,
    "reactor_fingertip_1": 1,
    "reactor_fingertip_2": 2,
    "reactor_fingertip_3": 3,
    "reactor_fingertip_4": 4,
    "reactor_palm": 5,
    "reactor_knuckle": 6,
    "actor_left_hand": 7,
    "actor_right_hand": 8,
    "actor_left_arm": 9,
    "actor_right_arm": 10,
    "torso_front_upper": 11,
    "head_lower_front": 12,
}


def _softmin_distance(a_xyz, b_xyz, beta=30.0):
    if a_xyz.numel() == 0 or b_xyz.numel() == 0:
        return a_xyz.new_full((a_xyz.shape[0],), 1e6)
    dist = torch.linalg.norm(a_xyz[:, :, None, :] - b_xyz[:, None, :, :], dim=-1)
    dist = dist.reshape(dist.shape[0], -1)
    beta = float(beta)
    softmin = -torch.logsumexp(-beta * dist, dim=-1) / max(beta, 1e-6)
    count = max(int(dist.shape[-1]), 1)
    return (softmin + torch.log(torch.as_tensor(float(count), device=dist.device, dtype=dist.dtype)) / max(beta, 1e-6)).clamp(min=0.0)


class MeshContactFeatureBuilder:
    """
    Build sparse mesh tokens plus explicit geometry-state descriptors for the
    hand-centric geometry-first stage2 refiner.
    """

    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        density="medium",
        softmin_beta=30.0,
        max_nontarget_vertices=256,
        device="cpu",
    ):
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.density = density
        self.softmin_beta = float(softmin_beta)
        self.max_nontarget_vertices = int(max_nontarget_vertices)
        self.body_forward = RestoredBodyModelForward(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device=device,
        )
        self._provider = get_mesh_region_provider(
            density=density,
            body_model=body_model,
            pose_rep=pose_rep,
        )
        self.mesh_token_dim = 12
        self.mesh_relation_dim = 22
        self.geometry_state_dim = 13
        self.target_summary_dim = 10

    def _ensure_device(self, device):
        self.body_forward.to(device)

    def _to_vertices(self, motion, betas=None, gender_id=None, body_model_type=None):
        self._ensure_device(motion.device)
        num_frames = motion.shape[-1]
        mask = torch.ones(motion.shape[0], num_frames, device=motion.device, dtype=torch.bool)
        return self.body_forward.motion_to_xyz(
            motion,
            jointstype="vertices",
            betas=betas,
            gender_id=gender_id,
            mask=mask,
            body_model_type=body_model_type,
        )

    def _gather_patch(self, vertices, ids):
        if not ids:
            return None
        ids_t = torch.as_tensor(ids, device=vertices.device, dtype=torch.long)
        return vertices.index_select(1, ids_t).permute(0, 3, 1, 2)

    def _select_nontarget(self, actor_vertices, hand_center, ids):
        if not ids:
            return []
        ids_t = torch.as_tensor(ids, device=actor_vertices.device, dtype=torch.long)
        patch = actor_vertices.index_select(1, ids_t).permute(0, 3, 1, 2)
        dist = torch.linalg.norm(patch - hand_center.view(1, -1, 1, 3), dim=-1)
        dist_mean = dist.mean(dim=1).squeeze(0)
        k = min(self.max_nontarget_vertices, ids_t.shape[0])
        _, idx = torch.topk(dist_mean, k=k, largest=False)
        return ids_t[idx].detach().cpu().tolist()

    def _patch_stats(self, patch, reference_center):
        if patch is None:
            zeros = reference_center.new_zeros(reference_center.shape)
            return zeros, zeros, zeros, zeros, reference_center.new_zeros(reference_center.shape[0], 1)
        center = patch.mean(dim=2).squeeze(0)
        vel = temporal_diff(center)
        rel_dir = safe_normalize(center - reference_center)
        normal_proxy = safe_normalize(temporal_diff(rel_dir))
        spread = torch.linalg.norm(patch.squeeze(0) - center[:, None, :], dim=-1).mean(dim=-1, keepdim=True)
        return center, vel, rel_dir, normal_proxy, spread

    def build_window_features(self, actor_motion, reactor_motion, hand_side, target_part, metadata=None):
        actor_betas = None if metadata is None else metadata["actor_betas"]
        reactor_betas = None if metadata is None else metadata["reactor_betas"]
        actor_gender_id = None if metadata is None else metadata["actor_gender_id"]
        reactor_gender_id = None if metadata is None else metadata["reactor_gender_id"]
        body_model_type = None if metadata is None else metadata.get("body_model_type", self.body_model)

        actor_vertices = self._to_vertices(
            actor_motion,
            betas=actor_betas,
            gender_id=actor_gender_id,
            body_model_type=body_model_type,
        )
        reactor_vertices = self._to_vertices(
            reactor_motion,
            betas=reactor_betas,
            gender_id=reactor_gender_id,
            body_model_type=body_model_type,
        )

        reactor_patches = self._provider.reactor_hand_patch_ids(hand_side)
        actor_patches = self._provider.actor_target_patch_ids(target_part)
        actor_nontarget = self._provider.actor_nontarget_patch_ids(target_part)

        reactor_patch_ids = []
        reactor_patch_names = []
        reactor_centers = []
        reactor_vels = []
        reactor_dirs = []
        reactor_normals = []
        reactor_spreads = []
        hand_center = None

        hand_union_ids = []
        for name in REACTOR_PATCH_NAMES:
            ids = reactor_patches.get(name, [])
            hand_union_ids.extend(ids)
        hand_union_patch = self._gather_patch(reactor_vertices, sorted(set(hand_union_ids)))
        if hand_union_patch is None:
            hand_center = reactor_vertices.new_zeros(actor_motion.shape[-1], 3)
        else:
            hand_center = hand_union_patch.mean(dim=2).squeeze(0)

        for name in REACTOR_PATCH_NAMES:
            ids = reactor_patches.get(name, [])
            reactor_patch_ids.append(ids)
            reactor_patch_names.append(name)
            patch = self._gather_patch(reactor_vertices, ids)
            center, vel, rel_dir, normal_proxy, spread = self._patch_stats(patch, hand_center)
            reactor_centers.append(center)
            reactor_vels.append(vel)
            reactor_dirs.append(rel_dir)
            reactor_normals.append(normal_proxy)
            reactor_spreads.append(spread)

        actor_patch_ids = []
        actor_patch_names = []
        actor_centers = []
        actor_vels = []
        actor_dirs = []
        actor_normals = []
        actor_spreads = []

        target_union = sorted({vid for ids in actor_patches.values() for vid in ids})
        target_union_patch = self._gather_patch(actor_vertices, target_union)
        if target_union_patch is None:
            target_center = actor_vertices.new_zeros(actor_motion.shape[-1], 3)
            target_vel = target_center.clone()
            target_normal = target_center.clone()
            target_spread = actor_vertices.new_zeros(actor_motion.shape[-1], 1)
        else:
            target_center = target_union_patch.mean(dim=2).squeeze(0)
            target_vel = temporal_diff(target_center)
            target_normal = safe_normalize(temporal_diff(target_center))
            target_spread = torch.linalg.norm(
                target_union_patch.squeeze(0) - target_center[:, None, :],
                dim=-1,
            ).mean(dim=-1, keepdim=True)

        for name, ids in actor_patches.items():
            actor_patch_ids.append(ids)
            actor_patch_names.append(name)
            patch = self._gather_patch(actor_vertices, ids)
            center, vel, rel_dir, normal_proxy, spread = self._patch_stats(patch, target_center)
            actor_centers.append(center)
            actor_vels.append(vel)
            actor_dirs.append(rel_dir)
            actor_normals.append(normal_proxy)
            actor_spreads.append(spread)

        if actor_centers:
            actor_centers_t = torch.stack(actor_centers, dim=1)
            actor_vels_t = torch.stack(actor_vels, dim=1)
            actor_dirs_t = torch.stack(actor_dirs, dim=1)
            actor_normals_t = torch.stack(actor_normals, dim=1)
        else:
            actor_centers_t = reactor_vertices.new_zeros(actor_motion.shape[-1], 0, 3)
            actor_vels_t = actor_centers_t.clone()
            actor_dirs_t = actor_centers_t.clone()
            actor_normals_t = actor_centers_t.clone()

        reactor_centers_t = torch.stack(reactor_centers, dim=1)
        reactor_vels_t = torch.stack(reactor_vels, dim=1)
        reactor_dirs_t = torch.stack(reactor_dirs, dim=1)
        reactor_normals_t = torch.stack(reactor_normals, dim=1)

        actor_nontarget_ids = self._select_nontarget(actor_vertices, hand_center, actor_nontarget)
        nontarget_patch = self._gather_patch(actor_vertices, actor_nontarget_ids)

        hand_to_target = []
        hand_to_nontarget = []
        for ids in reactor_patch_ids:
            patch = self._gather_patch(reactor_vertices, ids)
            if patch is None or target_union_patch is None:
                dist_target = reactor_vertices.new_full((actor_motion.shape[-1],), 1e6)
            else:
                dist_target = _softmin_distance(patch.squeeze(0), target_union_patch.squeeze(0), beta=self.softmin_beta)
            if patch is None or nontarget_patch is None:
                dist_nontarget = reactor_vertices.new_full((actor_motion.shape[-1],), 1e6)
            else:
                dist_nontarget = _softmin_distance(patch.squeeze(0), nontarget_patch.squeeze(0), beta=self.softmin_beta)
            hand_to_target.append(dist_target)
            hand_to_nontarget.append(dist_nontarget)

        hand_to_target = torch.stack(hand_to_target, dim=-1)
        hand_to_nontarget = torch.stack(hand_to_nontarget, dim=-1)

        if target_union_patch is not None and nontarget_patch is not None:
            target_to_nontarget = _softmin_distance(
                target_union_patch.squeeze(0),
                nontarget_patch.squeeze(0),
                beta=self.softmin_beta,
            ).unsqueeze(-1)
        else:
            target_to_nontarget = reactor_vertices.new_full((actor_motion.shape[-1], 1), 1e6)

        hand_target_min = hand_to_target.min(dim=-1, keepdim=True).values
        hand_clearance = hand_to_nontarget.min(dim=-1, keepdim=True).values
        contact_conf = torch.exp(-hand_target_min / 0.03).clamp(0.0, 1.0)
        target_speed = torch.linalg.norm(target_vel, dim=-1, keepdim=True)
        approach_trend = torch.zeros_like(hand_target_min)
        approach_trend[1:] = -(hand_target_min[1:] - hand_target_min[:-1])
        target_dir_mean = safe_normalize(target_center - hand_center)
        target_normal_align = (safe_normalize(target_vel) * target_dir_mean).sum(dim=-1, keepdim=True)

        mesh_relation_feat = torch.cat(
            [
                hand_to_target,
                hand_to_nontarget,
                target_to_nontarget,
                hand_target_min,
                hand_clearance,
                contact_conf,
                target_speed,
                target_spread,
                target_normal_align,
                approach_trend,
            ],
            dim=-1,
        )

        geometry_state_feat = torch.cat(
            [
                contact_conf,
                hand_target_min,
                hand_clearance,
                target_to_nontarget,
                target_speed,
                target_spread,
                approach_trend,
                target_dir_mean,
                target_normal,
            ],
            dim=-1,
        )
        target_geometry_summary = torch.cat(
            [
                target_center,
                target_vel,
                target_normal,
                target_spread,
            ],
            dim=-1,
        )

        reactor_token_feat = torch.cat(
            [
                reactor_centers_t,
                reactor_vels_t,
                reactor_dirs_t,
                reactor_normals_t,
            ],
            dim=-1,
        )
        actor_token_feat = torch.cat(
            [
                actor_centers_t,
                actor_vels_t,
                actor_dirs_t,
                actor_normals_t,
            ],
            dim=-1,
        )
        mesh_token_feat = torch.cat([reactor_token_feat, actor_token_feat], dim=1)

        mesh_token_type = []
        for name in reactor_patch_names:
            mesh_token_type.append(PATCH_TYPE_IDS.get(f"reactor_{name}", 0))
        for name in actor_patch_names:
            mesh_token_type.append(PATCH_TYPE_IDS.get(name, 0))

        return {
            "mesh_token_feat": mesh_token_feat,
            "mesh_token_type": torch.as_tensor(mesh_token_type, device=mesh_token_feat.device, dtype=torch.long),
            "mesh_relation_feat": mesh_relation_feat,
            "geometry_state_feat": geometry_state_feat,
            "geometry_contact_conf": contact_conf,
            "geometry_target_distance": hand_target_min,
            "geometry_clearance": hand_clearance,
            "target_geometry_summary": target_geometry_summary,
            "reactor_patch_ids": reactor_patch_ids,
            "actor_target_patch_ids": actor_patch_ids,
            "actor_nontarget_patch_ids": actor_nontarget_ids,
        }
