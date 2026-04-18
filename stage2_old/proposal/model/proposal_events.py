import torch

from stage2_old.common.geometry.contact_defs import HAND_SIDES, TARGET_PARTS


def parse_contact_events(
    active,
    target_part,
    band,
    phase,
    lengths=None,
    threshold=0.5,
):
    """
    Parse frame-wise labels/logits into event segments.
    Returns list(list(dict)) with per-batch event lists.
    """
    def _to_active_mask(x):
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.dtype.is_floating_point and (x.min() < 0 or x.max() > 1):
            x = torch.sigmoid(x)
        return x > float(threshold)

    def _to_labels(x, num_classes):
        if x.dim() == 4 and x.shape[-1] == num_classes:
            return torch.argmax(x, dim=-1)
        if x.dim() == 3:
            return x.long()
        raise ValueError("Unsupported label/logit shape")

    active_mask = _to_active_mask(active)
    target_ids = _to_labels(target_part, 6)
    band_ids = _to_labels(band, 3)
    phase_ids = _to_labels(phase, 4)

    if torch.is_tensor(lengths):
        lengths = lengths.detach().cpu().tolist()
    batch_size, num_frames, num_hands = active_mask.shape
    results = []
    for b in range(batch_size):
        events = []
        length = num_frames if lengths is None else int(lengths[b])
        for h in range(num_hands):
            in_event = False
            start = 0
            for t in range(length):
                is_active = bool(active_mask[b, t, h])
                if is_active and not in_event:
                    in_event = True
                    start = t
                is_last = t == length - 1
                if in_event and (not is_active or is_last):
                    end = t if is_active and is_last else t - 1
                    seg_targets = target_ids[b, start : end + 1, h]
                    non_none = seg_targets[seg_targets > 0]
                    if non_none.numel() > 0:
                        dominant_id = int(torch.mode(non_none).values.item())
                    else:
                        dominant_id = 0
                    events.append(
                        {
                            "hand_side": HAND_SIDES[h],
                            "start_frame": int(start),
                            "end_frame": int(end),
                            "target_part_id": dominant_id,
                            "target_part": TARGET_PARTS[dominant_id],
                            "band_seq": band_ids[b, start : end + 1, h].tolist(),
                            "phase_seq": phase_ids[b, start : end + 1, h].tolist(),
                        }
                    )
                    in_event = False
        results.append(events)
    return results


class ContactEventParser:
    def __init__(self, threshold=0.5):
        self.threshold = float(threshold)

    def __call__(self, active, target_part, band, phase, lengths=None):
        return parse_contact_events(
            active,
            target_part,
            band,
            phase,
            lengths=lengths,
            threshold=self.threshold,
        )
