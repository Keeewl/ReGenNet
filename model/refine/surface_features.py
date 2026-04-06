import torch

from model.rotation2xyz import Rotation2xyz_x


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
    ):
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.sigma = float(sigma)
        self.rot2xyz = Rotation2xyz_x(device=device)

    @property
    def feature_dim(self):
        return 9

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

        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
            frame_ids = torch.arange(feat.shape[1], device=device).view(1, -1)
            valid = frame_ids < lengths.view(-1, 1)
            feat = feat * valid[:, :, None, None].float()

        if active_mask is not None:
            active_mask = active_mask.to(device)
            feat = feat * active_mask[:, :, None, None].float()

        return feat


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
