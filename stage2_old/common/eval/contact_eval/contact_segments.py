import torch

from stage2_old.common.geometry.contact_geometry import build_time_mask


def build_union_contact_mask(band, lengths=None, contact_id=2):
    """
    band: [B, T, 2] or [B, T] or bool mask
    lengths: [B] or None
    returns mask: [B, T] (bool)
    """
    if torch.is_tensor(band):
        band_t = band
    else:
        band_t = torch.as_tensor(band)

    if band_t.dim() == 3:
        mask = (band_t == contact_id).any(dim=-1)
    elif band_t.dim() == 2:
        mask = band_t if band_t.dtype == torch.bool else (band_t == contact_id)
    else:
        raise ValueError(f"Unexpected band shape: {tuple(band_t.shape)}")

    if lengths is not None:
        time_mask = build_time_mask(lengths, mask.shape[1], device=band_t.device)
        if time_mask is not None:
            mask = mask & time_mask

    return mask.bool()


def extract_contact_segments(contact_mask, lengths=None):
    """
    contact_mask: [B, T] bool
    lengths: [B] or None
    returns list[list[int]] of segment lengths per sequence
    """
    if torch.is_tensor(contact_mask):
        mask = contact_mask.detach().to("cpu").bool()
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

    segments = []
    for b in range(batch_size):
        max_len = int(max(0, lengths_list[b]))
        seq_segments = []
        run = 0
        for t in range(max_len):
            if bool(mask[b, t]):
                run += 1
            elif run > 0:
                seq_segments.append(run)
                run = 0
        if run > 0:
            seq_segments.append(run)
        segments.append(seq_segments)

    return segments
