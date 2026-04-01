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
        model_device = next(self.rot2xyz.smpl_model.parameters()).device
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
