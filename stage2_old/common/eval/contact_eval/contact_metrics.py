import torch

from stage2_old.common.geometry.contact_defs import (
    HAND_SIDES,
    HAND_JOINT_IDS,
    TARGET_PARTS,
    ACTOR_PART_JOINT_IDS,
    BAND_IDS,
)
from stage2_old.common.geometry.contact_geometry import ContactGeometry, topk_pairwise_distance
from stage2_old.common.geometry.proposal_labels import HandContactLabelBuilder

from stage2_old.common.eval.contact_eval.contact_segments import (
    build_union_contact_mask,
    extract_contact_segments,
)


def _compute_hand_part_distances(actor_xyz, reactor_xyz, topk=3):
    """
    returns top1/topk_mean: [B, T, 2, 5]
    """
    batch_size, _, _, num_frames = actor_xyz.shape
    device = actor_xyz.device
    top1 = torch.zeros(batch_size, num_frames, 2, 5, device=device, dtype=actor_xyz.dtype)
    topk_mean = torch.zeros_like(top1)
    for h_idx, side in enumerate(HAND_SIDES):
        hand_ids = HAND_JOINT_IDS[side]
        for p_idx, part_name in enumerate(TARGET_PARTS[1:]):
            actor_ids = ACTOR_PART_JOINT_IDS[part_name]
            dist_top1, dist_topk = topk_pairwise_distance(
                actor_xyz, reactor_xyz, actor_ids, hand_ids, topk
            )
            top1[:, :, h_idx, p_idx] = dist_top1
            topk_mean[:, :, h_idx, p_idx] = dist_topk
    return top1, topk_mean


def _ensure_label_builder(label_builder, **kwargs):
    if label_builder is not None:
        return label_builder
    return HandContactLabelBuilder(**kwargs)


def _ensure_geometry(geometry, **kwargs):
    if geometry is not None:
        return geometry
    return ContactGeometry(**kwargs)


def compute_hand_cd(
    actor_motion,
    reactor_motion,
    gt_reactor_motion,
    lengths=None,
    label_builder=None,
    geometry=None,
    topk=3,
    return_debug=False,
    actor_betas=None,
    reactor_betas=None,
    actor_gender_id=None,
    reactor_gender_id=None,
    body_model_type=None,
):
    if gt_reactor_motion is None:
        return {"hand_cd": None, "hand_cd_count": 0}

    label_builder = _ensure_label_builder(label_builder)
    geometry = _ensure_geometry(geometry)

    labels = label_builder.build(
        actor_motion,
        gt_reactor_motion,
        lengths=lengths,
        actor_betas=actor_betas,
        reactor_betas=reactor_betas,
        actor_gender_id=actor_gender_id,
        reactor_gender_id=reactor_gender_id,
        body_model_type=body_model_type,
        preserve_pair_space=True,
    )
    target_part = labels["target_part"]
    band = labels["band"]
    valid = (band == BAND_IDS["contact"]) & (target_part > 0)

    actor_xyz = geometry.to_xyz(
        actor_motion,
        betas=actor_betas,
        gender_id=actor_gender_id,
        body_model_type=body_model_type,
        preserve_pair_space=True,
    )
    reactor_xyz = geometry.to_xyz(
        reactor_motion,
        betas=reactor_betas,
        gender_id=reactor_gender_id,
        body_model_type=body_model_type,
        preserve_pair_space=True,
    )

    dist_top1, dist_topk_mean = _compute_hand_part_distances(actor_xyz, reactor_xyz, topk=topk)

    part_idx = (target_part - 1).clamp(min=0)
    dist_sel = dist_top1.gather(-1, part_idx.unsqueeze(-1)).squeeze(-1)
    dist_sel_topk = dist_topk_mean.gather(-1, part_idx.unsqueeze(-1)).squeeze(-1)

    valid_f = valid.float()
    count = valid_f.sum()
    if int(count.item()) == 0:
        results = {
            "hand_cd": None,
            "hand_cd_count": 0,
        }
        if return_debug:
            results["hand_cd_topk_mean"] = None
        return results

    denom = count.clamp(min=1.0)
    hand_cd = (dist_sel * valid_f).sum() / denom

    results = {
        "hand_cd": hand_cd,
        "hand_cd_count": count,
    }
    if return_debug:
        results["hand_cd_topk_mean"] = (dist_sel_topk * valid_f).sum() / denom
    return results


def compute_contact_ratio(contact_mask, lengths=None):
    """
    contact_mask: [B, T] bool
    returns dict with contact_ratio and num_valid_sequences
    """
    if not torch.is_tensor(contact_mask):
        contact_mask = torch.as_tensor(contact_mask)
    contact_mask = contact_mask.bool()

    batch_size, num_frames = contact_mask.shape
    if lengths is None:
        valid_frames = torch.full((batch_size,), num_frames, device=contact_mask.device)
    else:
        valid_frames = torch.as_tensor(lengths, device=contact_mask.device, dtype=torch.long)
    valid_frames_f = valid_frames.float()
    valid_seq = valid_frames_f > 0

    contact_frames = contact_mask.sum(dim=1).float()
    ratio = torch.zeros_like(contact_frames)
    ratio = torch.where(valid_seq, contact_frames / valid_frames_f.clamp(min=1.0), ratio)

    if valid_seq.any():
        ratio_mean = ratio[valid_seq].mean()
    else:
        ratio_mean = torch.zeros((), device=contact_mask.device)

    return {
        "contact_ratio": ratio_mean,
        "num_valid_sequences": valid_seq.sum(),
        "num_contact_frames": contact_frames.sum(),
    }


def compute_avg_contact_duration(contact_mask, lengths=None):
    segments = extract_contact_segments(contact_mask, lengths=lengths)
    all_lengths = [seg_len for seq in segments for seg_len in seq]
    total_segments = len(all_lengths)
    if total_segments == 0:
        avg_duration = 0.0
    else:
        avg_duration = float(sum(all_lengths)) / float(total_segments)
    return {
        "avg_contact_duration": avg_duration,
        "num_contact_segments": total_segments,
    }


def compute_contact_frequency(contact_mask, lengths=None):
    if torch.is_tensor(contact_mask):
        mask = contact_mask.detach().to("cpu").bool()
        batch_size, num_frames = mask.shape
    else:
        mask = torch.as_tensor(contact_mask, dtype=torch.bool)
        batch_size, num_frames = mask.shape

    if lengths is None:
        lengths_list = [num_frames] * batch_size
    else:
        if torch.is_tensor(lengths):
            lengths_list = lengths.detach().to("cpu").tolist()
        else:
            lengths_list = [int(x) for x in lengths]

    segments = extract_contact_segments(mask, lengths=lengths_list)
    freqs = []
    for seq_len, seq_segments in zip(lengths_list, segments):
        seq_len = int(max(0, seq_len))
        if seq_len == 0:
            continue
        freq = float(len(seq_segments)) / (float(seq_len) / 100.0)
        freqs.append(freq)

    if freqs:
        mean_freq = sum(freqs) / float(len(freqs))
    else:
        mean_freq = 0.0

    return {
        "contact_frequency": mean_freq,
        "num_valid_sequences": len(freqs),
    }


def build_contact_labels(
    actor_motion,
    reactor_motion,
    lengths=None,
    label_builder=None,
    actor_betas=None,
    reactor_betas=None,
    actor_gender_id=None,
    reactor_gender_id=None,
    body_model_type=None,
    **builder_kwargs,
):
    label_builder = _ensure_label_builder(label_builder, **builder_kwargs)
    return label_builder.build(
        actor_motion,
        reactor_motion,
        lengths=lengths,
        actor_betas=actor_betas,
        reactor_betas=reactor_betas,
        actor_gender_id=actor_gender_id,
        reactor_gender_id=reactor_gender_id,
        body_model_type=body_model_type,
        preserve_pair_space=True,
    )


def build_contact_mask(
    actor_motion,
    reactor_motion,
    lengths=None,
    label_builder=None,
    actor_betas=None,
    reactor_betas=None,
    actor_gender_id=None,
    reactor_gender_id=None,
    body_model_type=None,
    **builder_kwargs,
):
    labels = build_contact_labels(
        actor_motion,
        reactor_motion,
        lengths=lengths,
        label_builder=label_builder,
        actor_betas=actor_betas,
        reactor_betas=reactor_betas,
        actor_gender_id=actor_gender_id,
        reactor_gender_id=reactor_gender_id,
        body_model_type=body_model_type,
        **builder_kwargs,
    )
    return build_union_contact_mask(labels["band"], lengths=lengths)
