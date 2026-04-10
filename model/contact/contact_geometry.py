import torch

from model.rotation2xyz import Rotation2xyz_x


def build_time_mask(lengths, num_frames, device=None):
    if lengths is None:
        return None
    device = device or lengths.device
    frame_ids = torch.arange(num_frames, device=device).view(1, -1)
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
    return frame_ids < lengths.view(-1, 1)


def temporal_diff(x):
    if x.shape[1] < 2:
        return torch.zeros_like(x)
    diff = x[:, 1:] - x[:, :-1]
    pad = diff[:, :1] * 0.0
    return torch.cat([pad, diff], dim=1)


def safe_normalize(x, eps=1e-6):
    denom = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
    return x / denom


def topk_pairwise_distance(actor_xyz, reactor_xyz, actor_ids, reactor_ids, topk):
    """
    actor_xyz/reactor_xyz: [B, J, 3, T]
    returns top1, topk_mean: [B, T]
    """
    device = actor_xyz.device
    actor_ids = torch.as_tensor(actor_ids, device=device, dtype=torch.long)
    reactor_ids = torch.as_tensor(reactor_ids, device=device, dtype=torch.long)
    batch_size, _, _, num_frames = actor_xyz.shape
    if actor_ids.numel() == 0 or reactor_ids.numel() == 0:
        pad = actor_xyz.new_full((batch_size, num_frames), 1e6)
        return pad, pad

    actor_sel = actor_xyz.index_select(1, actor_ids).permute(0, 3, 1, 2)
    reactor_sel = reactor_xyz.index_select(1, reactor_ids).permute(0, 3, 1, 2)
    dist = torch.linalg.norm(actor_sel[:, :, :, None, :] - reactor_sel[:, :, None, :, :], dim=-1)
    dist_flat = dist.reshape(batch_size, num_frames, -1)

    k = min(int(topk), dist_flat.shape[-1])
    dist_topk, _ = torch.topk(dist_flat, k=k, dim=-1, largest=False)
    if k < topk:
        pad = dist_topk[..., -1:].expand(batch_size, num_frames, topk - k)
        dist_topk = torch.cat([dist_topk, pad], dim=-1)
    return dist_topk[..., 0], dist_topk.mean(dim=-1)


class ContactGeometry:
    def __init__(self, body_model="smplx", pose_rep="rot6d", translation=True, glob=True, device="cpu"):
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.rot2xyz = Rotation2xyz_x(device=device)

    def _ensure_device(self, device):
        if self.rot2xyz.device != device:
            self.rot2xyz = Rotation2xyz_x(device=device)

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
