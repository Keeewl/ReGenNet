import torch


PART_JOINT_IDS = {
    "torso_head": [0, 3, 6, 9, 12, 15, 22, 23, 24],
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "left_hand": [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    "right_hand": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
}


def _default_refine_joint_ids():
    return sorted(
        set(
            PART_JOINT_IDS["left_arm"]
            + PART_JOINT_IDS["right_arm"]
            + PART_JOINT_IDS["left_hand"]
            + PART_JOINT_IDS["right_hand"]
            + [12, 15]
        )
    )


class ActiveWindowSelector:
    """
    Select active windows based on actor/reactor joint proximity.
    """

    def __init__(
        self,
        joint_ids=None,
        top_k=5,
        window_size=5,
        vel_threshold=None,
    ):
        self.joint_ids = joint_ids or _default_refine_joint_ids()
        self.top_k = int(top_k)
        self.window_size = int(window_size)
        self.vel_threshold = vel_threshold

    def select(self, actor_xyz, reactor_xyz, lengths=None):
        """
        actor_xyz/reactor_xyz: [B, J, 3, T]
        lengths: [B]
        Returns active_mask [B, T], joint_mask [J], scores [B, T]
        """
        device = actor_xyz.device
        batch_size, num_joints, _, num_frames = actor_xyz.shape
        joint_ids = torch.as_tensor(self.joint_ids, device=device, dtype=torch.long)

        actor_sel = actor_xyz.index_select(1, joint_ids)
        reactor_sel = reactor_xyz.index_select(1, joint_ids)

        scores = torch.zeros(batch_size, num_frames, device=device, dtype=actor_xyz.dtype)
        for t in range(num_frames):
            diff = actor_sel[:, :, None, :, t] - reactor_sel[:, None, :, :, t]
            dist = torch.linalg.norm(diff, dim=-1)
            scores[:, t] = dist.amin(dim=2).amin(dim=1)

        if self.vel_threshold is not None:
            vel = reactor_sel[:, :, :, 1:] - reactor_sel[:, :, :, :-1]
            vel_mag = torch.linalg.norm(vel, dim=2)
            vel_mag = torch.cat([vel_mag[:, :, :1], vel_mag], dim=2)
            active_vel = vel_mag.amax(dim=1) > self.vel_threshold
            scores = scores + (~active_vel).float() * 1e6

        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
            frame_ids = torch.arange(num_frames, device=device).view(1, -1)
            valid = frame_ids < lengths.view(-1, 1)
            scores = scores.masked_fill(~valid, 1e6)

        top_k = min(self.top_k, num_frames)
        _, indices = torch.topk(scores, k=top_k, dim=1, largest=False)

        active_mask = torch.zeros(batch_size, num_frames, device=device, dtype=torch.bool)
        radius = max(0, self.window_size // 2)
        for b in range(batch_size):
            for idx in indices[b]:
                start = max(0, int(idx) - radius)
                end = min(num_frames, int(idx) + radius + 1)
                active_mask[b, start:end] = True

        if lengths is not None:
            active_mask = active_mask & (frame_ids < lengths.view(-1, 1))

        joint_mask = torch.zeros(num_joints, device=device, dtype=torch.bool)
        joint_mask[joint_ids] = True

        return active_mask, joint_mask, scores
