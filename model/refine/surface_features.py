import torch

from model.rotation2xyz import Rotation2xyz_x
from model.refine.active_window import PART_JOINT_IDS


def default_part_joint_ids():
    return {
        "torso_head": PART_JOINT_IDS["torso_head"],
        "left_arm": PART_JOINT_IDS["left_arm"],
        "right_arm": PART_JOINT_IDS["right_arm"],
        "left_hand": PART_JOINT_IDS["left_hand"],
        "right_hand": PART_JOINT_IDS["right_hand"],
        "coord": [12, 15],
    }


def default_candidate_contact_pairs():
    pairs = []
    actor_parts = ["left_hand", "right_hand", "left_arm", "right_arm"]
    reactor_parts_hand = ["left_hand", "right_hand"]
    reactor_parts_arm = ["left_arm", "right_arm"]
    reactor_parts = reactor_parts_hand + reactor_parts_arm + ["torso_head"]
    for actor_part in actor_parts:
        for reactor_part in reactor_parts:
            if actor_part in ["left_hand", "right_hand"] and reactor_part in reactor_parts:
                pairs.append((actor_part, reactor_part))
            elif actor_part in ["left_arm", "right_arm"] and reactor_part in (reactor_parts_hand + reactor_parts_arm + ["torso_head"]):
                pairs.append((actor_part, reactor_part))
    return pairs


def _build_time_mask(lengths, num_frames, active_mask=None, device=None):
    """
    lengths: [B]
    active_mask: [B, T] or None
    returns mask: [B, T]
    """
    device = device or lengths.device
    frame_ids = torch.arange(num_frames, device=device).view(1, -1)
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
    mask = frame_ids < lengths.view(-1, 1)
    if active_mask is not None:
        mask = mask & active_mask.to(device)
    return mask.float()


def _pairwise_distance(actor_xyz, reactor_xyz, actor_ids, reactor_ids):
    """
    actor_xyz/reactor_xyz: [B, J, 3, T]
    actor_ids/reactor_ids: list[int] or 1D tensor with length P
    returns dist: [B, T, P]
    """
    device = actor_xyz.device
    actor_ids = torch.as_tensor(actor_ids, device=device, dtype=torch.long)
    reactor_ids = torch.as_tensor(reactor_ids, device=device, dtype=torch.long)
    if actor_ids.numel() != reactor_ids.numel():
        raise ValueError("actor_ids and reactor_ids must have same length")
    actor_sel = actor_xyz.index_select(1, actor_ids)
    reactor_sel = reactor_xyz.index_select(1, reactor_ids)
    dist = torch.linalg.norm(actor_sel - reactor_sel, dim=2)
    return dist.permute(0, 2, 1).contiguous()


def _topk_pairwise_distances(actor_xyz, reactor_xyz, actor_ids, reactor_ids, topk):
    device = actor_xyz.device
    actor_ids = torch.as_tensor(actor_ids, device=device, dtype=torch.long)
    reactor_ids = torch.as_tensor(reactor_ids, device=device, dtype=torch.long)
    batch_size, _, _, num_frames = actor_xyz.shape

    if actor_ids.numel() == 0 or reactor_ids.numel() == 0:
        dist_topk = torch.full((batch_size, num_frames, topk), 1e6, device=device, dtype=actor_xyz.dtype)
        actor_topk = torch.zeros((batch_size, num_frames, topk), device=device, dtype=torch.long)
        reactor_topk = torch.zeros((batch_size, num_frames, topk), device=device, dtype=torch.long)
        return dist_topk, actor_topk, reactor_topk

    actor_sel = actor_xyz.index_select(1, actor_ids).permute(0, 3, 1, 2)
    reactor_sel = reactor_xyz.index_select(1, reactor_ids).permute(0, 3, 1, 2)
    dist = torch.linalg.norm(actor_sel[:, :, :, None, :] - reactor_sel[:, :, None, :, :], dim=-1)
    dist_flat = dist.reshape(batch_size, num_frames, -1)

    total_pairs = dist_flat.shape[-1]
    k = min(int(topk), total_pairs)
    dist_topk, idx_topk = torch.topk(dist_flat, k=k, dim=-1, largest=False)

    if k < topk:
        pad = topk - k
        dist_pad = dist_topk[..., -1:].expand(batch_size, num_frames, pad)
        idx_pad = idx_topk[..., -1:].expand(batch_size, num_frames, pad)
        dist_topk = torch.cat([dist_topk, dist_pad], dim=-1)
        idx_topk = torch.cat([idx_topk, idx_pad], dim=-1)

    reactor_count = reactor_sel.shape[2]
    actor_idx = idx_topk // reactor_count
    reactor_idx = idx_topk % reactor_count
    actor_topk = actor_ids[actor_idx]
    reactor_topk = reactor_ids[reactor_idx]
    return dist_topk, actor_topk, reactor_topk


def build_semantic_topk_pairs(
    actor_xyz,
    reactor_xyz,
    candidate_pairs=None,
    part_joint_ids=None,
    topk=3,
    lengths=None,
    active_mask=None,
):
    if candidate_pairs is None:
        candidate_pairs = default_candidate_contact_pairs()
    if part_joint_ids is None:
        part_joint_ids = default_part_joint_ids()
    if not candidate_pairs:
        raise ValueError("candidate_pairs must be non-empty")

    dist_list = []
    actor_list = []
    reactor_list = []
    reactor_parts = []
    for actor_part, reactor_part in candidate_pairs:
        actor_ids = part_joint_ids.get(actor_part, [])
        reactor_ids = part_joint_ids.get(reactor_part, [])
        dist_topk, actor_topk, reactor_topk = _topk_pairwise_distances(
            actor_xyz, reactor_xyz, actor_ids, reactor_ids, topk
        )
        dist_list.append(dist_topk)
        actor_list.append(actor_topk)
        reactor_list.append(reactor_topk)
        reactor_parts.append(reactor_part)

    dist_topk = torch.stack(dist_list, dim=2)
    actor_topk = torch.stack(actor_list, dim=2)
    reactor_topk = torch.stack(reactor_list, dim=2)

    mask = None
    if lengths is not None or active_mask is not None:
        num_frames = dist_topk.shape[1]
        mask = _build_time_mask(lengths, num_frames, active_mask=active_mask, device=dist_topk.device)
        dist_topk = dist_topk * mask[:, :, None, None]

    return {
        "dist_topk": dist_topk,
        "actor_ids": actor_topk,
        "reactor_ids": reactor_topk,
        "mask": mask,
        "reactor_parts": reactor_parts,
    }


def gather_pairwise_distances(actor_xyz, reactor_xyz, actor_ids, reactor_ids):
    """
    actor_xyz/reactor_xyz: [B, J, 3, T]
    actor_ids/reactor_ids: [B, T, S, K] with joint indices
    returns dist: [B, T, S, K]
    """
    actor_t = actor_xyz.permute(0, 3, 1, 2)
    reactor_t = reactor_xyz.permute(0, 3, 1, 2)
    batch_size, num_frames, _, _ = actor_t.shape
    actor_flat = actor_t.reshape(batch_size * num_frames, actor_t.shape[2], 3)
    reactor_flat = reactor_t.reshape(batch_size * num_frames, reactor_t.shape[2], 3)
    actor_ids_flat = actor_ids.reshape(batch_size * num_frames, -1)
    reactor_ids_flat = reactor_ids.reshape(batch_size * num_frames, -1)
    actor_sel = torch.gather(actor_flat, 1, actor_ids_flat.unsqueeze(-1).expand(-1, -1, 3))
    reactor_sel = torch.gather(reactor_flat, 1, reactor_ids_flat.unsqueeze(-1).expand(-1, -1, 3))
    dist = torch.linalg.norm(actor_sel - reactor_sel, dim=-1)
    return dist.view(batch_size, num_frames, *actor_ids.shape[2:])


class SurfaceFeatureBuilder:
    """
    Joints-level local geometry feature builder.
    """

    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        sigma=0.1,
        device="cpu",
        use_contact_feature_aug=False,
        part_joint_ids=None,
        candidate_pairs=None,
        pair_feature_topk=3,
        use_closing_speed=True,
        use_part_contact_summary=True,
    ):
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.sigma = float(sigma)
        self.rot2xyz = Rotation2xyz_x(device=device)
        self.use_contact_feature_aug = bool(use_contact_feature_aug)
        self.part_joint_ids = part_joint_ids or default_part_joint_ids()
        self.candidate_pairs = candidate_pairs or default_candidate_contact_pairs()
        self.pair_feature_topk = int(pair_feature_topk)
        self.use_closing_speed = bool(use_closing_speed)
        self.use_part_contact_summary = bool(use_part_contact_summary)

    @property
    def feature_dim(self):
        base = 9
        if not self.use_contact_feature_aug:
            return base
        extra = 3
        if self.use_closing_speed:
            extra += 1
        if self.use_part_contact_summary:
            extra += 1
        return base + extra

    def _ensure_device(self, device):
        param = next(self.rot2xyz.smpl_model.parameters(), None)
        model_device = param.device if param is not None else torch.device("cpu")
        if model_device != device:
            self.rot2xyz.smpl_model.to(device)

    def to_xyz(self, motion, mask=None):
        self._ensure_device(motion.device)
        return self.rot2xyz(
            x=motion,
            mask=mask,
            pose_rep=self.pose_rep,
            translation=self.translation,
            glob=self.glob,
            jointstype=self.body_model,
            vertstrans=True,
            num_person=1,
            betas=None,
            beta=0,
            glob_rot=None,
        )

    def _joint_part_lookup(self, joint_ids):
        lookup = []
        for jid in joint_ids:
            part_name = None
            for name, ids in self.part_joint_ids.items():
                if jid in ids:
                    part_name = name
                    break
            lookup.append(part_name)
        return lookup

    def _build_contact_aug(
        self,
        actor_xyz,
        reactor_xyz,
        joint_ids,
        lengths=None,
        active_mask=None,
    ):
        stats = build_semantic_topk_pairs(
            actor_xyz,
            reactor_xyz,
            candidate_pairs=self.candidate_pairs,
            part_joint_ids=self.part_joint_ids,
            topk=self.pair_feature_topk,
            lengths=lengths,
            active_mask=active_mask,
        )
        dist_topk = stats["dist_topk"]
        reactor_parts = stats["reactor_parts"]

        batch_size, num_frames, _, _ = dist_topk.shape
        part_features = {}
        for part in set(reactor_parts):
            indices = [i for i, name in enumerate(reactor_parts) if name == part]
            if not indices:
                continue
            dist_sel = dist_topk[:, :, indices, :].reshape(batch_size, num_frames, -1)
            top1 = dist_sel.amin(dim=-1)
            topk_mean = dist_sel.mean(dim=-1)
            margin = topk_mean - top1
            feats = [top1, topk_mean, margin]
            if self.use_closing_speed:
                delta = top1[:, 1:] - top1[:, :-1]
                closing = torch.relu(-delta)
                closing = torch.cat([closing[:, :1] * 0.0, closing], dim=1)
                feats.append(closing)
            if self.use_part_contact_summary:
                soft_contact = torch.exp(-dist_sel / max(self.sigma, 1e-6))
                feats.append(soft_contact.mean(dim=-1))
            part_features[part] = torch.stack(feats, dim=-1)

        feat_dim = 3 + int(self.use_closing_speed) + int(self.use_part_contact_summary)
        joint_ids = torch.as_tensor(joint_ids, device=actor_xyz.device, dtype=torch.long)
        part_lookup = self._joint_part_lookup(joint_ids.tolist())
        aug_feat = torch.zeros(
            batch_size, num_frames, joint_ids.shape[0], feat_dim, device=actor_xyz.device, dtype=actor_xyz.dtype
        )
        for i, part in enumerate(part_lookup):
            if part in part_features:
                aug_feat[:, :, i, :] = part_features[part]
        return aug_feat

    def build(self, actor_xyz, reactor_xyz, joint_ids, lengths=None, active_mask=None):
        """
        actor_xyz/reactor_xyz: [B, J, 3, T]
        returns features: [B, T, R, F]
        """
        device = actor_xyz.device
        joint_ids = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
        actor_local = actor_xyz.index_select(1, joint_ids)
        reactor_local = reactor_xyz.index_select(1, joint_ids)

        rel_pos = actor_local - reactor_local
        dist = torch.linalg.norm(rel_pos, dim=2)

        actor_vel = actor_local[:, :, :, 1:] - actor_local[:, :, :, :-1]
        reactor_vel = reactor_local[:, :, :, 1:] - reactor_local[:, :, :, :-1]
        actor_vel = torch.cat([actor_vel[:, :, :, :1], actor_vel], dim=3)
        reactor_vel = torch.cat([reactor_vel[:, :, :, :1], reactor_vel], dim=3)
        rel_vel = actor_vel - reactor_vel
        speed = torch.linalg.norm(reactor_vel, dim=2)

        contact = torch.exp(-dist / max(self.sigma, 1e-6))

        feat = torch.cat(
            [
                rel_pos,
                dist.unsqueeze(2),
                rel_vel,
                contact.unsqueeze(2),
                speed.unsqueeze(2),
            ],
            dim=2,
        )
        feat = feat.permute(0, 3, 1, 2).contiguous()

        if self.use_contact_feature_aug:
            aug_feat = self._build_contact_aug(
                actor_xyz,
                reactor_xyz,
                joint_ids,
                lengths=lengths,
                active_mask=active_mask,
            )
            feat = torch.cat([feat, aug_feat], dim=-1)

        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
            frame_ids = torch.arange(feat.shape[1], device=device).view(1, -1)
            valid = frame_ids < lengths.view(-1, 1)
            feat = feat * valid[:, :, None, None].float()

        if active_mask is not None:
            active_mask = active_mask.to(device)
            feat = feat * active_mask[:, :, None, None].float()

        return feat



def build_pairwise_contact_stats(
    actor_xyz,
    reactor_xyz,
    actor_ids,
    reactor_ids,
    lengths=None,
    active_mask=None,
    sigma=0.1,
):
    """
    actor_xyz/reactor_xyz: [B, J, 3, T]
    actor_ids/reactor_ids: list[int] length P
    returns dict:
        dist: [B, T, P]
        soft_contact: [B, T, P]
        mask: [B, T]
    """
    dist = _pairwise_distance(actor_xyz, reactor_xyz, actor_ids, reactor_ids)
    soft_contact = torch.exp(-dist / max(float(sigma), 1e-6))
    mask = None
    if lengths is not None or active_mask is not None:
        num_frames = dist.shape[1]
        mask = _build_time_mask(lengths, num_frames, active_mask=active_mask, device=dist.device)
        dist = dist * mask[:, :, None]
        soft_contact = soft_contact * mask[:, :, None]
    return {"dist": dist, "soft_contact": soft_contact, "mask": mask}


def build_distance_prior_targets(
    actor_xyz,
    gt_xyz,
    actor_ids,
    reactor_ids,
    lengths=None,
    active_mask=None,
    tau=0.1,
):
    """
    actor_xyz/gt_xyz: [B, J, 3, T]
    actor_ids/reactor_ids: list[int] length P
    returns dict:
        dist_gt: [B, T, P]
        weight: [B, T, P]
        mask: [B, T]
    """
    dist_gt = _pairwise_distance(actor_xyz, gt_xyz, actor_ids, reactor_ids)
    weight = torch.exp(-dist_gt / max(float(tau), 1e-6))
    mask = None
    if lengths is not None or active_mask is not None:
        num_frames = dist_gt.shape[1]
        mask = _build_time_mask(lengths, num_frames, active_mask=active_mask, device=dist_gt.device)
        dist_gt = dist_gt * mask[:, :, None]
        weight = weight * mask[:, :, None]
    return {"dist_gt": dist_gt, "weight": weight, "mask": mask}
