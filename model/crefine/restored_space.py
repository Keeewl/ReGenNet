import numpy as np
import torch


REQUIRED_CACHE_FIELDS = (
    "dataset_key",
    "actor_is_p1",
    "reactor_is_p2",
    "processed_frame_ix",
    "raw_frame_ix",
    "processed_nframes",
    "raw_nframes",
    "processed_fps",
    "raw_fps",
    "downsample",
    "actor_betas",
    "reactor_betas",
    "actor_gender_id",
    "reactor_gender_id",
    "body_model_type",
    "num_betas",
    "ground_offset_y_actor",
    "ground_offset_y_reactor",
    "pair_base_trans",
    "loader_base_trans",
)

OPTIONAL_CACHE_FIELDS = (
    "actor_raw_trans_clip",
    "reactor_raw_trans_clip",
    "actor_raw_root_orient_clip",
    "reactor_raw_root_orient_clip",
)

GENDER_NAME_TO_ID = {
    "neutral": 0,
    "male": 1,
    "female": 2,
}

GENDER_ID_TO_NAME = {idx: name for name, idx in GENDER_NAME_TO_ID.items()}


def normalize_gender_name(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None:
        return "neutral"
    text = str(value).strip().lower()
    if text in {"m", "man"}:
        text = "male"
    elif text in {"f", "woman"}:
        text = "female"
    if text not in GENDER_NAME_TO_ID:
        return "neutral"
    return text


def normalize_gender_id(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes) or isinstance(value, str):
        value = normalize_gender_name(value)
        return GENDER_NAME_TO_ID[value]
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 0
    if value not in GENDER_ID_TO_NAME:
        value = 0
    return value


def gender_id_to_name(value):
    return GENDER_ID_TO_NAME.get(normalize_gender_id(value), "neutral")


def validate_required_cache_fields(field_names, context="crefine cache"):
    missing = [name for name in REQUIRED_CACHE_FIELDS if name not in field_names]
    if missing:
        raise KeyError(
            f"{context} is missing required restored-space fields: {', '.join(missing)}"
        )


def ensure_restored_batch_fields(batch, context="crefine batch"):
    missing = [name for name in REQUIRED_CACHE_FIELDS if name not in batch]
    if missing:
        raise KeyError(
            f"{context} is missing required restored-space fields: {', '.join(missing)}"
        )


def _to_device_tensor(value, device=None, dtype=None):
    if torch.is_tensor(value):
        out = value
    else:
        out = torch.as_tensor(value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device)
    return out


def extract_restoration_metadata(batch, device=None):
    ensure_restored_batch_fields(batch)
    meta = {}
    for key in REQUIRED_CACHE_FIELDS + OPTIONAL_CACHE_FIELDS:
        if key not in batch:
            continue
        value = batch[key]
        if key in {"dataset_key", "body_model_type"}:
            meta[key] = value
            continue
        if torch.is_tensor(value):
            meta[key] = value.to(device=device) if device is not None else value
            continue
        if isinstance(value, np.ndarray):
            if value.dtype.kind in {"U", "S", "O"}:
                meta[key] = value
            else:
                meta[key] = _to_device_tensor(value, device=device)
            continue
        if isinstance(value, list):
            if value and isinstance(value[0], (str, bytes)):
                meta[key] = value
            elif value and isinstance(value[0], np.ndarray) and len({tuple(np.asarray(v).shape) for v in value}) != 1:
                meta[key] = [
                    _to_device_tensor(v, device=device)
                    for v in value
                ]
            else:
                meta[key] = _to_device_tensor(value, device=device)
            continue
        meta[key] = value
    return meta


def _translation_joint(motion):
    if motion.dim() != 4:
        raise ValueError("motion must be [B, J, F, T]")
    return motion[:, -1, :3, :]


def apply_restored_pair_space(motion, common_shift, y_shift=0.0):
    out = motion.clone()
    transl = _translation_joint(out)
    common_shift = _to_device_tensor(common_shift, device=out.device, dtype=out.dtype).view(-1, 3, 1)
    transl = transl + common_shift
    y_shift = _to_device_tensor(y_shift, device=out.device, dtype=out.dtype).view(-1, 1)
    transl[:, 1, :] = transl[:, 1, :] + y_shift
    out[:, -1, :3, :] = transl
    return out


def restore_motion_batch(actor_motion, reactor_motion, metadata):
    """
    Recover motions from canonical clip space into restored pair space.

    `loader_base_trans` undoes the DataLoader root subtraction used for stage1 inputs.
    `pair_base_trans` then moves the clip back from the processed canonical pair frame
    into the raw pair frame. `ground_offset_y_*` corrects the per-person floor offset
    mismatch introduced by neutral-shape preprocessing without using reactor GT global
    translations at test time.
    """
    common_shift = metadata["loader_base_trans"] + metadata["pair_base_trans"]
    actor_restored = apply_restored_pair_space(
        actor_motion,
        common_shift=common_shift,
        y_shift=metadata["ground_offset_y_actor"],
    )
    reactor_restored = apply_restored_pair_space(
        reactor_motion,
        common_shift=common_shift,
        y_shift=metadata["ground_offset_y_reactor"],
    )
    return actor_restored, reactor_restored


def select_window_metadata(metadata, batch_index):
    out = {}
    for key, value in metadata.items():
        if key in {"dataset_key", "body_model_type"}:
            if isinstance(value, (list, tuple)):
                out[key] = value[batch_index]
            elif isinstance(value, np.ndarray) and value.ndim > 0:
                out[key] = value[batch_index]
            else:
                out[key] = value
            continue
        if torch.is_tensor(value):
            out[key] = value[batch_index : batch_index + 1]
        elif isinstance(value, list):
            out[key] = value[batch_index]
        elif isinstance(value, np.ndarray) and value.ndim > 0:
            out[key] = value[batch_index : batch_index + 1]
        else:
            out[key] = value
    return out


def metadata_to_cpu_numpy(metadata):
    out = {}
    for key, value in metadata.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().numpy()
        else:
            out[key] = np.asarray(value) if isinstance(value, list) else value
    return out
