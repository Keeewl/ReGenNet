"""Restored-pair-space and shape-aware SMPL-X vertex helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import utils.rotation_conversions as geometry
from model.smpl import JOINTSTYPE_ROOT, SMPLX
from refine.data.restored_space import (
    REQUIRED_RESTORATION_METADATA_FIELDS,
    extract_restoration_metadata,
    restore_pair_batch,
)
from refine.data.schema import normalize_space_definition

from .schema import REQUIRED_BODY_METADATA_FIELDS, RESTORED_PAIR_SPACE


GENDER_ID_TO_NAME = {0: "neutral", 1: "male", 2: "female"}
GENDER_NAME_TO_ID = {name: idx for idx, name in GENDER_ID_TO_NAME.items()}


def normalize_gender_id(value: Any) -> int:
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1)[0].item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return GENDER_NAME_TO_ID.get(value.strip().lower(), 0)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 0
    return value if value in GENDER_ID_TO_NAME else 0


def gender_id_to_name(value: Any) -> str:
    return GENDER_ID_TO_NAME.get(normalize_gender_id(value), "neutral")


def _as_tensor(value, *, device, dtype=None):
    out = value if torch.is_tensor(value) else torch.as_tensor(value)
    out = out.to(device=device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _first_string(value: Any, default: str = "") -> str:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    elif isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip().lower()
    return text or default


def batch_space_definition(batch: dict[str, Any]) -> str:
    if "space_definition" not in batch:
        return ""
    return normalize_space_definition(_first_string(batch["space_definition"]), default="")


def restore_pair_if_needed(actor_motion: torch.Tensor, reactor_motion: torch.Tensor, batch: dict[str, Any]):
    """Return actor/reactor motions in restored_pair_space.

    If the pack already declares restored_pair_space, motions are used directly.
    Otherwise all stable restoration metadata fields must be present and the
    current refine.data restored-space helper is used.
    """

    space = batch_space_definition(batch)
    if space == RESTORED_PAIR_SPACE:
        return actor_motion, reactor_motion, {"space_definition": RESTORED_PAIR_SPACE}

    missing = [name for name in REQUIRED_RESTORATION_METADATA_FIELDS if name not in batch]
    if missing:
        raise KeyError(
            "reaction_data is not declared as restored_pair_space and cannot be "
            "restored because metadata is missing: " + ", ".join(missing)
        )

    meta = extract_restoration_metadata(batch, device=actor_motion.device)
    actor_restored, reactor_restored = restore_pair_batch(actor_motion, reactor_motion, meta)
    return actor_restored, reactor_restored, meta


def extract_body_metadata(batch: dict[str, Any], *, device, dtype=torch.float32) -> dict[str, Any]:
    missing = [name for name in REQUIRED_BODY_METADATA_FIELDS if name not in batch]
    if missing:
        raise KeyError(
            "refine_v2 mesh contact requires body shape/gender metadata: "
            + ", ".join(missing)
        )
    body_model_type = _first_string(batch["body_model_type"])
    if body_model_type != "smplx":
        raise ValueError(f"refine_v2 currently supports body_model_type='smplx', got '{body_model_type}'.")
    return {
        "actor_betas": _as_tensor(batch["actor_betas"], device=device, dtype=dtype),
        "reactor_betas": _as_tensor(batch["reactor_betas"], device=device, dtype=dtype),
        "actor_gender_id": _as_tensor(batch["actor_gender_id"], device=device, dtype=torch.long).view(-1),
        "reactor_gender_id": _as_tensor(batch["reactor_gender_id"], device=device, dtype=torch.long).view(-1),
        "body_model_type": body_model_type,
    }


def lengths_to_mask(lengths: torch.Tensor, num_frames: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long)
    frame_ids = torch.arange(num_frames, device=lengths.device).view(1, -1)
    return frame_ids < lengths.view(-1, 1)


class RestoredBodyModelForward:
    """Shape-aware SMPL-X forward that preserves restored pair-space translation."""

    def __init__(self, body_model: str = "smplx", pose_rep: str = "rot6d", device: str | torch.device = "cpu"):
        if body_model != "smplx":
            raise NotImplementedError("refine_v2 mesh contact currently supports SMPL-X only.")
        if pose_rep != "rot6d":
            raise NotImplementedError("refine_v2 mesh contact currently supports pose_rep=rot6d only.")
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.device = torch.device(device)
        self._smplx_models: dict[str, SMPLX] = {}

    def to(self, device):
        self.device = torch.device(device)
        for model in self._smplx_models.values():
            model.to(self.device)
        return self

    def _get_model(self, gender_id):
        gender_name = gender_id_to_name(gender_id)
        if gender_name not in self._smplx_models:
            self._smplx_models[gender_name] = SMPLX(gender=gender_name).eval().to(self.device)
        return self._smplx_models[gender_name]

    def _rotations_from_motion(self, motion: torch.Tensor) -> dict[str, torch.Tensor]:
        x_rotations = motion[:, :-1].permute(0, 3, 1, 2).contiguous()
        batch, num_frames, num_joints, _ = x_rotations.shape
        rotations = geometry.rotation_6d_to_matrix(x_rotations.reshape(batch * num_frames, num_joints, 6))
        global_orient = rotations[:, 0]
        rotations = rotations[:, 1:]
        return {
            "body_pose": rotations[:, 0:21],
            "jaw_pose": rotations[:, 21:22],
            "leye_pose": rotations[:, 22:23],
            "reye_pose": rotations[:, 23:24],
            "left_hand_pose": rotations[:, 24:39],
            "right_hand_pose": rotations[:, 39:54],
            "global_orient": global_orient,
        }

    @torch.no_grad()
    def motion_to_xyz(
        self,
        motion: torch.Tensor,
        *,
        jointstype: str = "vertices",
        betas: torch.Tensor | None = None,
        gender_id: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        body_model_type: str | None = "smplx",
    ) -> torch.Tensor:
        body_model_type = str(body_model_type or "smplx").lower()
        if body_model_type != "smplx":
            raise NotImplementedError("refine_v2 mesh contact currently supports body_model_type=smplx only.")
        if motion.dim() != 4:
            raise ValueError("motion must have shape [B, J, F, T].")
        self.to(motion.device)
        batch, _, _, num_frames = motion.shape
        if mask is None:
            mask = torch.ones(batch, num_frames, dtype=torch.bool, device=motion.device)
        else:
            mask = mask.to(device=motion.device, dtype=torch.bool)
        if betas is None:
            betas = torch.zeros(batch, 10, dtype=motion.dtype, device=motion.device)
        else:
            betas = betas.to(device=motion.device, dtype=motion.dtype)
        if gender_id is None:
            gender_id = torch.zeros(batch, dtype=torch.long, device=motion.device)
        else:
            gender_id = gender_id.to(device=motion.device, dtype=torch.long).view(-1)

        translations = motion[:, -1, :3, :].permute(0, 2, 1).reshape(batch * num_frames, 3)
        bt_index = torch.arange(batch, device=motion.device).view(-1, 1).expand(batch, num_frames).reshape(-1)
        betas_bt = betas.index_select(0, bt_index)
        gender_bt = gender_id.index_select(0, bt_index)
        rot_parts = self._rotations_from_motion(motion)

        out_flat = None
        for gid in torch.unique(gender_bt, sorted=True).tolist():
            select = torch.nonzero(gender_bt == gid, as_tuple=False).flatten()
            if select.numel() == 0:
                continue
            model = self._get_model(gid)
            model_out = model(
                betas=betas_bt.index_select(0, select),
                body_pose=rot_parts["body_pose"].index_select(0, select),
                left_hand_pose=rot_parts["left_hand_pose"].index_select(0, select),
                right_hand_pose=rot_parts["right_hand_pose"].index_select(0, select),
                global_orient=rot_parts["global_orient"].index_select(0, select),
                return_verts=True,
            )
            xyz = model_out[jointstype].to(dtype=motion.dtype)
            if out_flat is None:
                out_flat = torch.zeros(
                    batch * num_frames,
                    xyz.shape[1],
                    3,
                    dtype=motion.dtype,
                    device=motion.device,
                )
            out_flat.index_copy_(0, select, xyz)

        if out_flat is None:
            raise RuntimeError("SMPL-X forward produced no vertices/joints.")

        out_flat = out_flat + translations[:, None, :]
        out_bt = out_flat.view(batch, num_frames, out_flat.shape[1], 3)
        if jointstype != "vertices":
            root_index = JOINTSTYPE_ROOT[jointstype]
            out_bt = out_bt - out_bt[:, :, root_index : root_index + 1, :]
            out_bt = out_bt + translations.view(batch, num_frames, 1, 3)
        out_bt = out_bt * mask[:, :, None, None].to(dtype=out_bt.dtype)
        return out_bt.permute(0, 2, 3, 1).contiguous()


def motions_to_vertices(
    actor_motion: torch.Tensor,
    reactor_motion: torch.Tensor,
    lengths: torch.Tensor,
    batch: dict[str, Any],
    *,
    body_forward: RestoredBodyModelForward | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    body_meta = extract_body_metadata(batch, device=actor_motion.device, dtype=actor_motion.dtype)
    body_forward = body_forward or RestoredBodyModelForward(device=actor_motion.device)
    mask = lengths_to_mask(lengths.to(actor_motion.device), actor_motion.shape[-1])
    actor_vertices = body_forward.motion_to_xyz(
        actor_motion,
        jointstype="vertices",
        betas=body_meta["actor_betas"],
        gender_id=body_meta["actor_gender_id"],
        mask=mask,
        body_model_type=body_meta["body_model_type"],
    )
    reactor_vertices = body_forward.motion_to_xyz(
        reactor_motion,
        jointstype="vertices",
        betas=body_meta["reactor_betas"],
        gender_id=body_meta["reactor_gender_id"],
        mask=mask,
        body_model_type=body_meta["body_model_type"],
    )
    return actor_vertices, reactor_vertices
