from __future__ import annotations

from dataclasses import dataclass
import os
import pickle

import numpy as np


ACTOR_COLOR = (0.10, 0.47, 0.78, 1.0)
REACTOR_COLOR = (0.88, 0.30, 0.20, 1.0)

# 这个值越大，最早那个 snapshot 越白，看起来就越像“透明”
TIME_GRADIENT_LIGHTEST_MIX = 0.30
# TIME_GRADIENT_LIGHTEST_MIX = 0.15


def _meta_value(params: dict, key: str, default=None):
    if key not in params:
        return default
    value = params[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _reshape_pose(params: dict, key: str, num_frames: int) -> np.ndarray:
    return np.asarray(params[key], dtype=np.float32).reshape(num_frames, -1)


@dataclass
class PersonClip:
    params: dict
    root_orient: np.ndarray
    pose_body: np.ndarray
    pose_lhand: np.ndarray
    pose_rhand: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    gender: str

    @property
    def num_frames(self) -> int:
        return int(self.pose_body.shape[0])


@dataclass
class ClipData:
    clip_name: str
    clip_dir: str
    p1: PersonClip
    p2: PersonClip

    @property
    def num_frames(self) -> int:
        return min(self.p1.num_frames, self.p2.num_frames)


def _load_person(npz_path: str) -> PersonClip:
    raw = np.load(npz_path, allow_pickle=True)
    params = {key: raw[key] for key in raw.files}
    raw.close()

    num_frames = int(np.asarray(params["pose_body"]).shape[0])
    return PersonClip(
        params=params,
        root_orient=np.asarray(params["root_orient"], dtype=np.float32).reshape(num_frames, -1),
        pose_body=_reshape_pose(params, "pose_body", num_frames),
        pose_lhand=_reshape_pose(params, "pose_lhand", num_frames),
        pose_rhand=_reshape_pose(params, "pose_rhand", num_frames),
        trans=np.asarray(params["trans"], dtype=np.float32).reshape(num_frames, 3),
        betas=np.asarray(params["betas"], dtype=np.float32),
        gender=str(_meta_value(params, "gender", "neutral")),
    )


def _apply_share_shape(clip: ClipData, share_shape: str) -> ClipData:
    share_shape = str(share_shape or "none")
    if share_shape == "none":
        return clip

    if share_shape == "p1":
        betas = clip.p1.betas.copy()
        gender = clip.p1.gender
        clip.p2.betas = betas
        clip.p2.gender = gender
    elif share_shape == "p2":
        betas = clip.p2.betas.copy()
        gender = clip.p2.gender
        clip.p1.betas = betas
        clip.p1.gender = gender
    elif share_shape == "mean":
        betas = ((clip.p1.betas + clip.p2.betas) * 0.5).astype(np.float32)
        gender = clip.p1.gender
        clip.p1.betas = betas
        clip.p2.betas = betas.copy()
        clip.p2.gender = gender
    else:
        raise ValueError(f"Unsupported share_shape={share_shape}")
    return clip


def load_clip(clip_dir: str, share_shape: str = "none") -> ClipData:
    clip_dir = os.path.abspath(clip_dir)
    if not os.path.isdir(clip_dir):
        raise FileNotFoundError(f"Clip folder not found: {clip_dir}")

    p1_path = os.path.join(clip_dir, "P1.npz")
    p2_path = os.path.join(clip_dir, "P2.npz")
    if not os.path.exists(p1_path) or not os.path.exists(p2_path):
        raise FileNotFoundError(f"Clip folder must contain P1.npz and P2.npz: {clip_dir}")

    clip = ClipData(
        clip_name=os.path.basename(clip_dir),
        clip_dir=clip_dir,
        p1=_load_person(p1_path),
        p2=_load_person(p2_path),
    )
    return _apply_share_shape(clip, share_shape)


def resolve_clip_dir(
    clip_dir: str | None,
    data_dir: str | None,
    clip_name: str | None,
    dataset: str | None,
) -> str:
    if clip_dir:
        return os.path.abspath(clip_dir)

    if not data_dir and dataset == "interx":
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "interx_data")
    elif not data_dir and dataset == "chi3d":
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chi3d_data")

    if not data_dir or not clip_name:
        raise ValueError("Provide either --clip_dir, or provide both --data_dir and --clip_name")

    return os.path.abspath(os.path.join(data_dir, clip_name))


def infer_interaction_order_path(dataset: str | None, explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path
    if dataset != "interx":
        return ""

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sibling_interx_root = os.path.join(os.path.dirname(repo_root), "Inter-X")
    candidates = [
        os.path.join(repo_root, "dataset", "interx", "annots", "interaction_order.pkl"),
        os.path.join(sibling_interx_root, "datasets", "interx", "annots", "interaction_order.pkl"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return ""


def load_interaction_order(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as handle:
        return pickle.load(handle)


def resolve_person_roles(clip: ClipData, order_dict: dict | None = None) -> tuple[str, str]:
    order_dict = order_dict or {}
    role_p1 = str(_meta_value(clip.p1.params, "source_role", "") or "")
    role_p2 = str(_meta_value(clip.p2.params, "source_role", "") or "")
    if not role_p1 and not role_p2 and clip.clip_name in order_dict:
        actor_is_p1 = int(order_dict[clip.clip_name]) == 1
        role_p1 = "actor" if actor_is_p1 else "reactor"
        role_p2 = "reactor" if actor_is_p1 else "actor"
    if not role_p1:
        role_p1 = "actor"
    if not role_p2:
        role_p2 = "reactor"
    return role_p1, role_p2


def resolve_person_colors(clip: ClipData, order_dict: dict | None = None):
    role_p1, role_p2 = resolve_person_roles(clip, order_dict=order_dict)
    p1_color = ACTOR_COLOR if role_p1 == "actor" else REACTOR_COLOR
    p2_color = ACTOR_COLOR if role_p2 == "actor" else REACTOR_COLOR
    return p1_color, p2_color


def blend_rgb_towards_white(color, white_mix: float):
    rgba = np.asarray(color, dtype=np.float32).copy()
    white_mix = float(np.clip(white_mix, 0.0, 1.0))
    rgba[:3] = rgba[:3] * (1.0 - white_mix) + white_mix
    return tuple(float(value) for value in rgba)


# 
def compute_time_gradient_mixes(snapshot_specs, lightest_mix: float = TIME_GRADIENT_LIGHTEST_MIX) -> dict[int, float]:
    specs = list(snapshot_specs)
    if not specs:
        return {}
    if len(specs) == 1:
        return {spec.index: 0.0 for spec in specs}

    lightest_mix = float(np.clip(lightest_mix, 0.0, 1.0))
    # Respect the user-provided frame order instead of re-sorting by frame id.
    ordered_specs = sorted(specs, key=lambda spec: int(spec.index))
    denom = len(ordered_specs) - 1
    mixes = {}
    for rank, spec in enumerate(ordered_specs):
        progress = rank / denom
        mixes[spec.index] = lightest_mix * (1.0 - progress)
    return mixes


def validate_frame_ids(clip: ClipData, frame_ids) -> None:
    max_frame = clip.num_frames - 1
    if max_frame < 0:
        raise ValueError(f"Clip {clip.clip_name} contains no frames")
    invalid = [int(frame_id) for frame_id in frame_ids if int(frame_id) < 0 or int(frame_id) > max_frame]
    if invalid:
        raise ValueError(
            f"Invalid frame_ids {invalid} for clip {clip.clip_name}; valid range is [0, {max_frame}]"
        )


def build_frame_sequence_kwargs(person: PersonClip, frame_id: int, offset, smpl_layer, device, color):
    frame_id = int(frame_id)
    trans = person.trans[frame_id : frame_id + 1].copy()
    trans[0] += np.asarray(offset, dtype=np.float32).reshape(3)
    return {
        "poses_body": person.pose_body[frame_id : frame_id + 1].copy(),
        "smpl_layer": smpl_layer,
        "poses_root": person.root_orient[frame_id : frame_id + 1].copy(),
        "betas": person.betas.copy(),
        "trans": trans,
        "poses_left_hand": person.pose_lhand[frame_id : frame_id + 1].copy(),
        "poses_right_hand": person.pose_rhand[frame_id : frame_id + 1].copy(),
        "device": device,
        "color": color,
    }
