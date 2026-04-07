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


class ActiveWindowSelectorV2:
    """
    Risk-aware active window selector.

    Inputs:
        actor_xyz/reactor_xyz: [B, J, 3, T]
        lengths: [B]

    Outputs:
        active_mask: [B, T]
        joint_mask: [J]
        scores: [B, T]
    """

    def __init__(
        self,
        joint_ids=None,
        top_k=5,
        window_size=5,
        vel_threshold=None,
        sigma_contact=0.1,
        alpha=1.0,
        beta=0.5,
        gamma=0.5,
    ):
        self.joint_ids = joint_ids or _default_refine_joint_ids()
        self.top_k = int(top_k)
        self.window_size = int(window_size)
        self.vel_threshold = vel_threshold
        self.sigma_contact = float(sigma_contact)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

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

        dist_min = torch.zeros(batch_size, num_frames, device=device, dtype=actor_xyz.dtype)
        for t in range(num_frames):
            diff = actor_sel[:, :, None, :, t] - reactor_sel[:, None, :, :, t]
            dist = torch.linalg.norm(diff, dim=-1)
            dist_min[:, t] = dist.amin(dim=2).amin(dim=1)

        dist_delta = dist_min[:, 1:] - dist_min[:, :-1]
        approach = torch.relu(-dist_delta)
        approach = torch.cat([approach[:, :1] * 0.0, approach], dim=1)

        contact = torch.exp(-dist_min / max(self.sigma_contact, 1e-6))
        scores = self.alpha * dist_min - self.beta * approach - self.gamma * contact

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



def _min_pairwise_distance(actor_xyz, reactor_xyz, joint_ids):
    """
    actor_xyz/reactor_xyz: [B, J, 3, T]
    joint_ids: 1D tensor
    returns dist_min: [B, T]
    """
    device = actor_xyz.device
    joint_ids = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
    actor_sel = actor_xyz.index_select(1, joint_ids)
    reactor_sel = reactor_xyz.index_select(1, joint_ids)
    batch_size, _, _, num_frames = actor_sel.shape
    dist_min = torch.zeros(batch_size, num_frames, device=device, dtype=actor_xyz.dtype)
    for t in range(num_frames):
        diff = actor_sel[:, :, None, :, t] - reactor_sel[:, None, :, :, t]
        dist = torch.linalg.norm(diff, dim=-1)
        dist_min[:, t] = dist.amin(dim=2).amin(dim=1)
    return dist_min


def expand_time_mask(mask, window_size):
    """
    mask: [B, T] bool
    window_size: int (full window length)
    returns expanded mask: [B, T] bool
    """
    if mask is None:
        return None
    radius = max(0, int(window_size) // 2)
    if radius == 0:
        return mask
    batch_size, num_frames = mask.shape
    expanded = mask.clone()
    for b in range(batch_size):
        active_idx = torch.nonzero(mask[b], as_tuple=False).flatten().tolist()
        for idx in active_idx:
            start = max(0, int(idx) - radius)
            end = min(num_frames, int(idx) + radius + 1)
            expanded[b, start:end] = True
    return expanded


def build_oracle_active_mask(
    actor_xyz,
    coarse_xyz,
    gt_xyz,
    selector,
    lengths=None,
    tau_contact=0.1,
    tau_near=0.18,
    contact_error_margin=0.05,
    train_window_size=10,
):
    """
    Oracle-enhanced training mask:
        M_train = M_gt_contact ∪ M_contact_error ∪ M_coarse_risk
    """
    coarse_mask, joint_mask, scores = selector.select(actor_xyz, coarse_xyz, lengths=lengths)
    dist_gt = _min_pairwise_distance(actor_xyz, gt_xyz, selector.joint_ids)
    dist_coarse = _min_pairwise_distance(actor_xyz, coarse_xyz, selector.joint_ids)

    gt_contact = dist_gt < float(tau_near)
    gt_contact_strict = dist_gt < float(tau_contact)
    coarse_contact = dist_coarse < float(tau_contact)

    contact_error = (gt_contact_strict != coarse_contact) | (
        gt_contact & ((dist_coarse - dist_gt) > float(contact_error_margin))
    )

    if lengths is not None:
        lengths_t = torch.as_tensor(lengths, device=actor_xyz.device, dtype=torch.long)
        frame_ids = torch.arange(dist_gt.shape[1], device=actor_xyz.device).view(1, -1)
        valid = frame_ids < lengths_t.view(-1, 1)
        gt_contact = gt_contact & valid
        gt_contact_strict = gt_contact_strict & valid
        contact_error = contact_error & valid

    train_mask = coarse_mask | gt_contact | contact_error
    if train_window_size is not None and int(train_window_size) > 0:
        train_mask = expand_time_mask(train_mask, train_window_size)

    return train_mask, {
        "coarse_mask": coarse_mask,
        "joint_mask": joint_mask,
        "scores": scores,
        "gt_contact_mask": gt_contact_strict,
        "gt_near_mask": gt_contact,
        "contact_error_mask": contact_error,
        "dist_gt_min": dist_gt,
        "dist_coarse_min": dist_coarse,
    }


def compute_overlap_metrics(coarse_mask, gt_mask):
    """
    coarse_mask/gt_mask: [B, T] bool
    returns dict with batch-mean metrics
    """
    coarse_mask = coarse_mask.bool()
    gt_mask = gt_mask.bool()
    inter = (coarse_mask & gt_mask).sum(dim=1).float()
    union = (coarse_mask | gt_mask).sum(dim=1).float().clamp(min=1.0)
    gt_count = gt_mask.sum(dim=1).float().clamp(min=1.0)
    coarse_count = coarse_mask.sum(dim=1).float().clamp(min=1.0)
    iou = inter / union
    recall = inter / gt_count
    precision = inter / coarse_count
    return {
        "overlap_iou": iou.mean(),
        "gt_contact_recall_by_coarse_risk": recall.mean(),
        "coarse_risk_precision_wrt_gt": precision.mean(),
    }
