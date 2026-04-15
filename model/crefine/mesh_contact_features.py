import torch

from model.contact.contact_geometry import temporal_diff
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
    return -torch.logsumexp(-beta * dist, dim=-1) / max(beta, 1e-6)


class MeshContactFeatureBuilder:
    """
    Build sparse mesh token features and relation descriptors per window.
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
            density=density, body_model=body_model, pose_rep=pose_rep
        )

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
        patch = vertices.index_select(1, ids_t).permute(0, 3, 1, 2)
        return patch

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

    def build_window_features(self, actor_motion, reactor_motion, hand_side, target_part, metadata=None):
        """
        actor_motion/reactor_motion: [1, J, 6, T]
        returns dict with mesh tokens and relation features.
        """
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

        reactor_centers = []
        reactor_vels = []
        reactor_patch_ids = []
        reactor_patch_names = []
        for name in REACTOR_PATCH_NAMES:
            ids = reactor_patches.get(name, [])
            reactor_patch_ids.append(ids)
            reactor_patch_names.append(name)
            patch = self._gather_patch(reactor_vertices, ids)
            if patch is None:
                center = reactor_vertices.new_zeros(actor_motion.shape[-1], 3)
                vel = center.clone()
            else:
                center = patch.mean(dim=2).squeeze(0)
                vel = temporal_diff(center)
            reactor_centers.append(center)
            reactor_vels.append(vel)

        actor_centers = []
        actor_vels = []
        actor_patch_ids = []
        actor_patch_names = []
        for name, ids in actor_patches.items():
            actor_patch_ids.append(ids)
            actor_patch_names.append(name)
            patch = self._gather_patch(actor_vertices, ids)
            if patch is None:
                center = actor_vertices.new_zeros(actor_motion.shape[-1], 3)
                vel = center.clone()
            else:
                center = patch.mean(dim=2).squeeze(0)
                vel = temporal_diff(center)
            actor_centers.append(center)
            actor_vels.append(vel)

        if actor_centers:
            actor_centers_t = torch.stack(actor_centers, dim=1)
        else:
            actor_centers_t = reactor_vertices.new_zeros(actor_motion.shape[-1], 0, 3)

        reactor_centers_t = torch.stack(reactor_centers, dim=1)
        reactor_vels_t = torch.stack(reactor_vels, dim=1)
        actor_vels_t = torch.stack(actor_vels, dim=1) if actor_vels else actor_centers_t.clone()

        if reactor_centers_t.numel() > 0:
            hand_center = reactor_centers_t.mean(dim=1)
        else:
            hand_center = reactor_vertices.new_zeros(actor_motion.shape[-1], 3)

        actor_nontarget_ids = self._select_nontarget(actor_vertices, hand_center, actor_nontarget)

        target_union = sorted({vid for ids in actor_patch_ids for vid in ids})
        target_patch = self._gather_patch(actor_vertices, target_union)
        if target_patch is not None:
            target_patch = target_patch.squeeze(0)

        hand_to_target = []
        hand_to_nontarget = []
        for ids in reactor_patch_ids:
            patch = self._gather_patch(reactor_vertices, ids)
            if patch is None or target_patch is None:
                dist = reactor_vertices.new_full((actor_motion.shape[-1],), 1e6)
            else:
                dist = _softmin_distance(patch.squeeze(0), target_patch, beta=self.softmin_beta)
            hand_to_target.append(dist)

            if actor_nontarget_ids:
                nontarget_patch = self._gather_patch(actor_vertices, actor_nontarget_ids)
                nontarget_patch = nontarget_patch.squeeze(0)
                if patch is None:
                    dist_nt = reactor_vertices.new_full((actor_motion.shape[-1],), 1e6)
                else:
                    dist_nt = _softmin_distance(patch.squeeze(0), nontarget_patch, beta=self.softmin_beta)
            else:
                dist_nt = reactor_vertices.new_full((actor_motion.shape[-1],), 1e6)
            hand_to_nontarget.append(dist_nt)

        hand_to_target = torch.stack(hand_to_target, dim=-1)
        hand_to_nontarget = torch.stack(hand_to_nontarget, dim=-1)

        if target_patch is not None and actor_nontarget_ids:
            nontarget_patch = self._gather_patch(actor_vertices, actor_nontarget_ids)
            nontarget_patch = nontarget_patch.squeeze(0)
            target_to_nontarget = _softmin_distance(
                target_patch, nontarget_patch, beta=self.softmin_beta
            ).unsqueeze(-1)
        else:
            target_to_nontarget = reactor_vertices.new_full((actor_motion.shape[-1], 1), 1e6)

        mesh_relation_feat = torch.cat([hand_to_target, hand_to_nontarget, target_to_nontarget], dim=-1)

        mesh_token_feat = torch.cat(
            [
                torch.cat([reactor_centers_t, reactor_vels_t], dim=-1),
                torch.cat([actor_centers_t, actor_vels_t], dim=-1),
            ],
            dim=1,
        )

        mesh_token_type = []
        for name in reactor_patch_names:
            mesh_token_type.append(PATCH_TYPE_IDS.get(f"reactor_{name}", 0))
        for name in actor_patch_names:
            mesh_token_type.append(PATCH_TYPE_IDS.get(name, 0))

        return {
            "mesh_token_feat": mesh_token_feat,
            "mesh_token_type": torch.as_tensor(mesh_token_type, device=mesh_token_feat.device, dtype=torch.long),
            "mesh_relation_feat": mesh_relation_feat,
            "reactor_patch_ids": reactor_patch_ids,
            "actor_target_patch_ids": actor_patch_ids,
            "actor_nontarget_patch_ids": actor_nontarget_ids,
        }
