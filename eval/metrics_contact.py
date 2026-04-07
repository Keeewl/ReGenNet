import torch

from model.refine.surface_features import build_semantic_topk_pairs, gather_pairwise_distances


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
    return mask


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


def contact_distance(
    actor_xyz,
    pred_xyz,
    gt_xyz,
    actor_ids,
    reactor_ids,
    tau_contact=0.1,
    lengths=None,
    active_mask=None,
):
    """
    actor_xyz/pred_xyz/gt_xyz: [B, J, 3, T]
    actor_ids/reactor_ids: list[int] length P
    lengths: [B] or None
    active_mask: [B, T] or None
    returns dict:
        cd: scalar tensor
        count: scalar tensor (number of valid contact entries)
    """
    dist_pred = _pairwise_distance(actor_xyz, pred_xyz, actor_ids, reactor_ids)
    dist_gt = _pairwise_distance(actor_xyz, gt_xyz, actor_ids, reactor_ids)
    omega = dist_gt < float(tau_contact)
    if lengths is not None or active_mask is not None:
        num_frames = dist_gt.shape[1]
        mask_t = _build_time_mask(lengths, num_frames, active_mask=active_mask, device=dist_gt.device)
        omega = omega & mask_t[:, :, None]
    count = omega.sum().float()
    denom = count.clamp(min=1.0)
    cd = (dist_pred * omega.float()).sum() / denom
    return {"cd": cd, "count": count}


def compute_cd_metrics(
    actor_xyz,
    coarse_xyz,
    refined_xyz,
    gt_xyz,
    actor_ids,
    reactor_ids,
    tau_contact=0.1,
    lengths=None,
    active_mask=None,
):
    """
    actor_xyz/coarse_xyz/refined_xyz/gt_xyz: [B, J, 3, T]
    actor_ids/reactor_ids: list[int] length P
    returns dict:
        cd_coarse, cd_refined, cd_improve
        cd_active_coarse, cd_active_refined, cd_active_improve (if active_mask is not None)
    """
    coarse = contact_distance(
        actor_xyz,
        coarse_xyz,
        gt_xyz,
        actor_ids,
        reactor_ids,
        tau_contact=tau_contact,
        lengths=lengths,
        active_mask=None,
    )
    refined = contact_distance(
        actor_xyz,
        refined_xyz,
        gt_xyz,
        actor_ids,
        reactor_ids,
        tau_contact=tau_contact,
        lengths=lengths,
        active_mask=None,
    )
    metrics = {
        "cd_coarse": coarse["cd"],
        "cd_refined": refined["cd"],
        "cd_improve": coarse["cd"] - refined["cd"],
        "cd_count": coarse["count"],
    }

    if active_mask is not None:
        coarse_active = contact_distance(
            actor_xyz,
            coarse_xyz,
            gt_xyz,
            actor_ids,
            reactor_ids,
            tau_contact=tau_contact,
            lengths=lengths,
            active_mask=active_mask,
        )
        refined_active = contact_distance(
            actor_xyz,
            refined_xyz,
            gt_xyz,
            actor_ids,
            reactor_ids,
            tau_contact=tau_contact,
            lengths=lengths,
            active_mask=active_mask,
        )
        metrics.update({
            "cd_active_coarse": coarse_active["cd"],
            "cd_active_refined": refined_active["cd"],
            "cd_active_improve": coarse_active["cd"] - refined_active["cd"],
            "cd_active_count": coarse_active["count"],
        })

    return metrics



def contact_distance_semantic(
    actor_xyz,
    pred_xyz,
    gt_xyz,
    candidate_pairs,
    part_joint_ids,
    topk_pairs,
    tau_contact=0.1,
    lengths=None,
    active_mask=None,
):
    stats = build_semantic_topk_pairs(
        actor_xyz,
        gt_xyz,
        candidate_pairs=candidate_pairs,
        part_joint_ids=part_joint_ids,
        topk=topk_pairs,
        lengths=lengths,
        active_mask=active_mask,
    )
    dist_gt = stats["dist_topk"]
    dist_pred = gather_pairwise_distances(
        actor_xyz, pred_xyz, stats["actor_ids"], stats["reactor_ids"]
    )
    omega = dist_gt < float(tau_contact)
    if stats["mask"] is not None:
        omega = omega & stats["mask"][:, :, None, None].bool()
    count = omega.sum().float()
    denom = count.clamp(min=1.0)
    cd = (dist_pred * omega.float()).sum() / denom
    return {"cd": cd, "count": count}


def compute_cd_metrics_semantic(
    actor_xyz,
    coarse_xyz,
    refined_xyz,
    gt_xyz,
    candidate_pairs,
    part_joint_ids,
    topk_pairs,
    tau_contact=0.1,
    lengths=None,
    active_mask=None,
):
    coarse = contact_distance_semantic(
        actor_xyz,
        coarse_xyz,
        gt_xyz,
        candidate_pairs,
        part_joint_ids,
        topk_pairs,
        tau_contact=tau_contact,
        lengths=lengths,
        active_mask=None,
    )
    refined = contact_distance_semantic(
        actor_xyz,
        refined_xyz,
        gt_xyz,
        candidate_pairs,
        part_joint_ids,
        topk_pairs,
        tau_contact=tau_contact,
        lengths=lengths,
        active_mask=None,
    )
    metrics = {
        "cd_coarse": coarse["cd"],
        "cd_refined": refined["cd"],
        "cd_improve": coarse["cd"] - refined["cd"],
        "cd_count": coarse["count"],
    }

    if active_mask is not None:
        coarse_active = contact_distance_semantic(
            actor_xyz,
            coarse_xyz,
            gt_xyz,
            candidate_pairs,
            part_joint_ids,
            topk_pairs,
            tau_contact=tau_contact,
            lengths=lengths,
            active_mask=active_mask,
        )
        refined_active = contact_distance_semantic(
            actor_xyz,
            refined_xyz,
            gt_xyz,
            candidate_pairs,
            part_joint_ids,
            topk_pairs,
            tau_contact=tau_contact,
            lengths=lengths,
            active_mask=active_mask,
        )
        metrics.update({
            "cd_active_coarse": coarse_active["cd"],
            "cd_active_refined": refined_active["cd"],
            "cd_active_improve": coarse_active["cd"] - refined_active["cd"],
            "cd_active_count": coarse_active["count"],
        })

    return metrics
