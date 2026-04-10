import torch
import torch.nn as nn
import torch.nn.functional as F

from model.contact.contact_geometry import build_time_mask


def _masked_mean(loss, mask):
    if loss.numel() == 0:
        return loss.sum() * 0.0
    mask = mask.float()
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def _masked_mse(diff, mask):
    if diff.numel() == 0:
        return diff.sum() * 0.0
    mask = mask.float()
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    extra = diff[0, 0].numel() if diff.dim() > 2 else 1
    denom = (mask.sum() * extra).clamp(min=1.0)
    return (diff * diff * mask).sum() / denom


def active_loss(logits, targets, mask, use_focal=False, gamma=2.0, alpha=0.25):
    logits = logits.squeeze(-1)
    targets = targets.float()
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if use_focal:
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        weight = (alpha * targets + (1.0 - alpha) * (1.0 - targets)) * ((1.0 - pt) ** gamma)
        loss = loss * weight
    return _masked_mean(loss, mask)


def masked_ce(logits, targets, mask):
    num_classes = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, num_classes), targets.reshape(-1), reduction="none")
    mask_flat = mask.reshape(-1).float()
    return (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)


def temporal_smoothness_loss(logits, mask):
    if logits.shape[1] < 2:
        return logits.sum() * 0.0
    diff = logits[:, 1:] - logits[:, :-1]
    mask_t = mask[:, 1:] * mask[:, :-1]
    return _masked_mse(diff, mask_t)


def consistency_loss(active_logits, target_logits, band_logits, mask, none_id=0, far_id=0, contact_id=2):
    active_prob = torch.sigmoid(active_logits.squeeze(-1))
    target_prob = torch.softmax(target_logits, dim=-1)
    band_prob = torch.softmax(band_logits, dim=-1)

    contact_prob = band_prob[..., contact_id]
    target_non_none = 1.0 - target_prob[..., none_id]
    far_prob = band_prob[..., far_id]

    loss_contact = contact_prob * (1.0 - active_prob)
    loss_target = target_non_none * (1.0 - active_prob)
    loss_far = far_prob * (1.0 - target_prob[..., none_id])
    return _masked_mean(loss_contact + loss_target + loss_far, mask)


class HandContactProposalLoss(nn.Module):
    """
    Compute losses for contact proposal heads.
    """

    def __init__(
        self,
        lambda_smooth=0.1,
        lambda_consistency=0.1,
        use_focal=False,
        focal_gamma=2.0,
        focal_alpha=0.25,
    ):
        super().__init__()
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_consistency = float(lambda_consistency)
        self.use_focal = bool(use_focal)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)

    def forward(self, logits, labels, lengths=None):
        active_logits = logits["active"]
        target_logits = logits["target"]
        band_logits = logits["band"]
        phase_logits = logits["phase"]

        active = labels["active"]
        target = labels["target_part"]
        band = labels["band"]
        phase = labels["phase"]

        num_frames = active_logits.shape[1]
        time_mask = build_time_mask(lengths, num_frames, device=active_logits.device)
        if time_mask is None:
            mask = torch.ones_like(active[..., 0])
        else:
            mask = time_mask.float()
        mask = mask[:, :, None].expand(-1, -1, active_logits.shape[2])

        loss_active = active_loss(
            active_logits,
            active,
            mask,
            use_focal=self.use_focal,
            gamma=self.focal_gamma,
            alpha=self.focal_alpha,
        )
        loss_target = masked_ce(target_logits, target, mask)
        loss_band = masked_ce(band_logits, band, mask)
        loss_phase = masked_ce(phase_logits, phase, mask)

        loss_smooth = (
            temporal_smoothness_loss(active_logits, mask)
            + temporal_smoothness_loss(target_logits, mask)
            + temporal_smoothness_loss(band_logits, mask)
            + temporal_smoothness_loss(phase_logits, mask)
        )

        loss_consistency = consistency_loss(active_logits, target_logits, band_logits, mask)

        total = (
            loss_active
            + loss_target
            + loss_band
            + loss_phase
            + self.lambda_smooth * loss_smooth
            + self.lambda_consistency * loss_consistency
        )

        return total, {
            "loss_active": loss_active,
            "loss_target": loss_target,
            "loss_band": loss_band,
            "loss_phase": loss_phase,
            "loss_smooth": loss_smooth,
            "loss_consistency": loss_consistency,
            "loss_total": total,
        }
