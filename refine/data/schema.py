"""Schema helpers for the Stage1 -> Stage2-lite reaction_data bridge package."""

from __future__ import annotations

from typing import Iterable


REACTION_DATA_SPACE = "restored_pair_space"

REQUIRED_REACTION_DATA_FIELDS = (
    "actor_motion",
    "reactor_gt",
    "reactor_coarse",
    "lengths",
    "sample_indices",
)

OPTIONAL_REACTION_DATA_FIELDS = (
    "space_definition",
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
    "actor_raw_trans_clip",
    "reactor_raw_trans_clip",
    "actor_raw_root_orient_clip",
    "reactor_raw_root_orient_clip",
)


def _as_field_names(field_names_or_data) -> set[str]:
    if hasattr(field_names_or_data, "files"):
        return {str(name) for name in field_names_or_data.files}
    if hasattr(field_names_or_data, "keys"):
        return {str(name) for name in field_names_or_data.keys()}
    return {str(name) for name in field_names_or_data}


def validate_reaction_data_fields(
    field_names: Iterable[str],
    *,
    context: str = "reaction_data",
    allow_unknown: bool = True,
) -> tuple[str, ...]:
    """
    Validate the new Stage1 -> Stage2-lite reaction_data schema.

    The new Stage2-lite data path intentionally uses `reaction_data` instead of
    legacy names such as coarse_cache or restored_cache.
    """

    field_name_set = _as_field_names(field_names)
    missing = [name for name in REQUIRED_REACTION_DATA_FIELDS if name not in field_name_set]
    if missing:
        raise KeyError(
            f"{context} is missing required reaction_data fields: {', '.join(missing)}"
        )
    allowed = set(REQUIRED_REACTION_DATA_FIELDS) | set(OPTIONAL_REACTION_DATA_FIELDS)
    unexpected = sorted(field_name_set - allowed)
    if unexpected and not allow_unknown:
        raise KeyError(
            f"{context} has unknown reaction_data fields: {', '.join(unexpected)}"
        )
    return tuple(sorted(field_name_set))


def normalize_space_definition(value, default: str = "") -> str:
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        try:
            if value.shape == ():
                value = value.item()
            elif value.size > 0:
                value = value.reshape(-1)[0]
        except Exception:
            pass
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip().lower()
    return text or default


def check_reaction_data_schema(
    data,
    *,
    context: str = "reaction_data",
    allow_unknown: bool = True,
    require_space_definition: bool = False,
    expected_space_definition: str = REACTION_DATA_SPACE,
):
    validate_reaction_data_fields(data, context=context, allow_unknown=allow_unknown)
    if not require_space_definition:
        return data
    if hasattr(data, "get"):
        value = data.get("space_definition", None)
    else:
        raise TypeError(f"{context} does not provide key access for schema validation.")
    actual = normalize_space_definition(value)
    expected = normalize_space_definition(expected_space_definition)
    if not actual:
        raise ValueError(
            f"{context} is missing space_definition; expected '{expected}'."
        )
    if actual != expected:
        raise ValueError(
            f"{context} has space_definition='{actual}', expected '{expected}'."
        )
    return data
