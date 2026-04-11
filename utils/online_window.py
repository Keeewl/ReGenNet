import random
from typing import Iterable, Tuple

import torch


def validate_window_args(window_size: int, window_stride: int) -> None:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if window_stride <= 0:
        raise ValueError("window_stride must be > 0")
    if window_stride > window_size:
        raise ValueError("window_stride must be <= window_size")


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    rng = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return rng < lengths.unsqueeze(1)


def _slice_window(x: torch.Tensor, start: int, end: int, window_size: int, pad_mode: str) -> Tuple[torch.Tensor, int]:
    window = x[..., start:end]
    win_len = end - start
    if win_len < window_size:
        pad_len = window_size - win_len
        if pad_mode == "edge" and win_len > 0:
            pad = window[..., -1:].expand(*window.shape[:-1], pad_len)
        else:
            pad = torch.zeros(*window.shape[:-1], pad_len, device=x.device, dtype=x.dtype)
        window = torch.cat([window, pad], dim=-1)
    return window, win_len


def _emit_bounds(window_len: int, window_stride: int, window_emit: str) -> Tuple[int, int]:
    if window_len <= 0:
        return 0, 0
    if window_emit == "last":
        emit_len = 1
    elif window_emit == "stride":
        emit_len = min(window_stride, window_len)
    else:
        raise ValueError(f"Unsupported window_emit: {window_emit}")
    emit_start = max(0, window_len - emit_len)
    return emit_start, window_len


def iter_windows(length: int, window_size: int, window_stride: int) -> Iterable[Tuple[int, int]]:
    validate_window_args(window_size, window_stride)
    if length <= 0:
        return
    if length <= window_size:
        yield 0, length
        return
    start = 0
    while True:
        end = start + window_size
        if end >= length:
            yield max(0, length - window_size), length
            break
        yield start, end
        start += window_stride


def window_batch_for_online_training(
    motion: torch.Tensor,
    cond: dict,
    window_size: int,
    window_stride: int,
    window_emit: str = "stride",
    pad_mode: str = "edge",
    random_offset: bool = True,
) -> Tuple[torch.Tensor, dict]:
    validate_window_args(window_size, window_stride)
    y = cond["y"]
    lengths = y["lengths"].to(torch.long)
    device = motion.device
    batch = motion.shape[0]

    window_motions = []
    window_cmotions = []
    window_lengths = []

    for idx in range(batch):
        seq_len = int(lengths[idx].item())
        if seq_len <= 0:
            start = 0
        else:
            max_start = max(0, seq_len - window_size)
            if random_offset and max_start > 0:
                start = random.randint(0, max_start)
            else:
                start = max_start
        end = min(seq_len, start + window_size)

        win_motion, win_len = _slice_window(motion[idx], start, end, window_size, pad_mode)
        win_cmotion, _ = _slice_window(y["cmotion"][idx], start, end, window_size, pad_mode)

        window_motions.append(win_motion)
        window_cmotions.append(win_cmotion)
        window_lengths.append(win_len)

    window_motion = torch.stack(window_motions, dim=0)
    window_cmotion = torch.stack(window_cmotions, dim=0)
    window_lengths = torch.as_tensor(window_lengths, device=device, dtype=torch.long)

    base_mask = _lengths_to_mask(window_lengths, window_size)
    emit_mask = torch.zeros_like(base_mask)
    for idx, win_len in enumerate(window_lengths.tolist()):
        emit_start, emit_end = _emit_bounds(win_len, window_stride, window_emit)
        if emit_end > emit_start:
            emit_mask[idx, emit_start:emit_end] = True
    emit_mask = emit_mask.unsqueeze(1).unsqueeze(1)

    new_y = {}
    for key, value in y.items():
        if key in {"cmotion", "mask", "lengths"}:
            continue
        new_y[key] = value
    new_y["cmotion"] = window_cmotion
    new_y["mask"] = emit_mask
    new_y["lengths"] = window_lengths

    return window_motion, {"y": new_y}


def sliding_window_sample(
    model,
    diffusion,
    base_cond: dict,
    window_size: int,
    window_stride: int,
    window_emit: str = "stride",
    pad_mode: str = "edge",
    overlap_handling: str = "latest",
    sample_fn=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_window_args(window_size, window_stride)
    if overlap_handling != "latest":
        raise ValueError(f"Unsupported overlap handling: {overlap_handling}")
    if sample_fn is None:
        sample_fn = diffusion.p_sample_loop

    y = base_cond["y"]
    cmotion = y["cmotion"]
    lengths = y["lengths"].to(torch.long)
    device = cmotion.device
    batch, njoints, nfeats, total_len = cmotion.shape

    output = torch.zeros((batch, njoints, nfeats, total_len), device=device, dtype=cmotion.dtype)
    filled = torch.zeros((batch, total_len), device=device, dtype=torch.bool)

    for b in range(batch):
        seq_len = int(lengths[b].item())
        if seq_len <= 0:
            continue
        for start, end in iter_windows(seq_len, window_size, window_stride):
            cmotion_win, win_len = _slice_window(cmotion[b], start, end, window_size, pad_mode)
            win_len_t = torch.tensor([win_len], device=device, dtype=torch.long)
            mask = _lengths_to_mask(win_len_t, window_size).unsqueeze(1).unsqueeze(1)

            window_y = {}
            for key, value in y.items():
                if key in {"cmotion", "mask", "lengths"}:
                    continue
                if torch.is_tensor(value):
                    window_y[key] = value[b:b + 1]
                elif isinstance(value, list):
                    window_y[key] = [value[b]]
                else:
                    window_y[key] = value
            window_y["cmotion"] = cmotion_win.unsqueeze(0)
            window_y["mask"] = mask
            window_y["lengths"] = win_len_t

            model_kwargs = {"y": window_y}
            sample = sample_fn(
                model,
                (1, model.njoints, model.nfeats, window_size),
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )

            emit_start, emit_end = _emit_bounds(win_len, window_stride, window_emit)
            emit_len = emit_end - emit_start
            if emit_len <= 0:
                continue
            out_start = start + emit_start
            out_end = start + emit_end
            output[b, :, :, out_start:out_end] = sample[0, :, :, emit_start:emit_end]
            filled[b, out_start:out_end] = True

    return output, filled
