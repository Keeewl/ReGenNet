import torch


COORD_JOINT_IDS = [12, 15]


def build_time_mask(lengths, num_frames, active_mask=None, device=None):
    device = device or lengths.device
    frame_ids = torch.arange(num_frames, device=device).view(1, -1)
    lengths = lengths.to(device)
    mask = frame_ids < lengths.view(-1, 1)
    if active_mask is not None:
        mask = mask & active_mask.to(device)
    return mask.float()


def masked_mse(diff, mask):
    if diff.numel() == 0:
        return diff.sum() * 0.0
    mask = mask.float()
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    extra = diff[0, 0].numel() if diff.dim() > 2 else 1
    denom = mask.sum() * extra
    denom = denom.clamp(min=1.0)
    return (diff * diff * mask).sum() / denom


def residual_loss(delta_pred, delta_gt, mask):
    return masked_mse(delta_pred - delta_gt, mask)


def residual_reg(delta_pred, mask):
    return masked_mse(delta_pred, mask)


def coordination_reg(delta_pred, refine_joint_ids, mask, coord_joint_ids=None):
    coord_joint_ids = coord_joint_ids or COORD_JOINT_IDS
    coord_indices = [i for i, jid in enumerate(refine_joint_ids) if jid in coord_joint_ids]
    if not coord_indices:
        return delta_pred.sum() * 0.0
    coord_delta = delta_pred[:, :, coord_indices, :]
    return masked_mse(coord_delta, mask)


def local_distance_loss(actor_xyz, refined_xyz, gt_xyz, joint_ids, mask):
    joint_ids = torch.as_tensor(joint_ids, device=actor_xyz.device, dtype=torch.long)
    actor_local = actor_xyz.index_select(1, joint_ids)
    refined_local = refined_xyz.index_select(1, joint_ids)
    gt_local = gt_xyz.index_select(1, joint_ids)
    dist_refined = torch.linalg.norm(actor_local - refined_local, dim=2)
    dist_gt = torch.linalg.norm(actor_local - gt_local, dim=2)
    diff = dist_refined - dist_gt
    return masked_mse(diff, mask)


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


def masked_weighted_mse(diff, mask, weight):
    """
    diff: [B, T, ...]
    mask: [B, T]
    weight: [B, T, ...]
    returns scalar
    """
    if diff.numel() == 0:
        return diff.sum() * 0.0
    mask = mask.float()
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    while weight.dim() < diff.dim():
        weight = weight.unsqueeze(-1)
    extra = diff[0, 0].numel() if diff.dim() > 2 else 1
    denom = (mask * weight).sum() * extra
    denom = denom.clamp(min=1.0)
    return (diff * diff * mask * weight).sum() / denom


def distance_prior_loss(
    actor_xyz,
    refined_xyz,
    gt_xyz,
    actor_ids,
    reactor_ids,
    mask,
    tau=0.1,
):
    """
    actor_xyz/refined_xyz/gt_xyz: [B, J, 3, T]
    actor_ids/reactor_ids: list[int] length P
    mask: [B, T]
    returns scalar
    """
    dist_refined = _pairwise_distance(actor_xyz, refined_xyz, actor_ids, reactor_ids)
    dist_gt = _pairwise_distance(actor_xyz, gt_xyz, actor_ids, reactor_ids)
    weight = torch.exp(-dist_gt / max(float(tau), 1e-6))
    diff = dist_refined - dist_gt
    return masked_weighted_mse(diff, mask, weight)


def soft_contact_loss(
    actor_xyz,
    refined_xyz,
    gt_xyz,
    actor_ids,
    reactor_ids,
    mask,
    sigma=0.1,
):
    """
    actor_xyz/refined_xyz/gt_xyz: [B, J, 3, T]
    actor_ids/reactor_ids: list[int] length P
    mask: [B, T]
    returns scalar
    """
    dist_refined = _pairwise_distance(actor_xyz, refined_xyz, actor_ids, reactor_ids)
    dist_gt = _pairwise_distance(actor_xyz, gt_xyz, actor_ids, reactor_ids)
    contact_refined = torch.exp(-dist_refined / max(float(sigma), 1e-6))
    contact_gt = torch.exp(-dist_gt / max(float(sigma), 1e-6))
    diff = contact_refined - contact_gt
    return masked_mse(diff, mask)


def smoothness_loss(delta_pred, mask):
    """
    delta_pred: [B, T, J, C]
    mask: [B, T]
    returns scalar
    """
    if delta_pred.shape[1] < 2:
        return delta_pred.sum() * 0.0
    diff = delta_pred[:, 1:] - delta_pred[:, :-1]
    mask_t = mask[:, 1:] * mask[:, :-1]
    return masked_mse(diff, mask_t)
