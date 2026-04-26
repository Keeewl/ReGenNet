"""Run refine_v2 on one viewer-ready clip and export viewer-ready refined clips."""

from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import utils.rotation_conversions as geometry
from refine_v2.data.contact_labels import compute_contact_for_batch
from refine_v2.data.restored_space import GENDER_NAME_TO_ID, RestoredBodyModelForward
from refine_v2.data.schema import (
    DEFAULT_GAP_MERGE,
    DEFAULT_PER_HAND_MAX_WINDOWS,
    DEFAULT_PER_SEQ_MAX_WINDOWS,
    DEFAULT_RAW_L_MIN,
    DEFAULT_TAU_CONTACT,
    DEFAULT_TOP_K_REGIONS,
    DEFAULT_WINDOW_SIZE,
    RESTORED_PAIR_SPACE,
)
from refine_v2.eval.full_sequence_stitch import build_center_weight
from refine_v2.model.refiner_v2 import RefineV2WindowRefiner, RefineV2WindowRefinerConfig
from refine_v2.model.regions import load_region_map
from refine_v2.model.selector_v2 import (
    _build_hand_segments_for_batch,
    _limit_windows,
    windows_from_segment,
)
from refine_v2.refiner_data.feature_pack import build_window_feature_sample
from refine_v2.refiner_data.window_loader import collate_refine_v2_window_batch
from refine_v2.tools.build_geometry_feature_cache import _compute_batch_geometry
from refine_v2.train.eval_window import batch_to_device
from visualize.converters.convert_results_to_motions import rot6d_to_rotvec
from visualize.viewer.snapshot.clip import (
    infer_interaction_order_path,
    load_clip,
    load_interaction_order,
    resolve_clip_dir,
    resolve_person_roles,
)


def _normalize_gender_name(value: str) -> str:
    text = str(value or "neutral").strip().lower()
    if text not in GENDER_NAME_TO_ID:
        return "neutral"
    return text


def _meta_value(params: dict[str, Any], key: str, default=None):
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


def _frame_array(params: dict[str, Any], key: str, length: int) -> np.ndarray:
    value = params.get(key, None)
    if value is None:
        return np.arange(length, dtype=np.int64)
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.arange(length, dtype=np.int64)
    if arr.shape[0] < length:
        pad = np.arange(arr.shape[0], length, dtype=np.int64)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:length]


def _person_to_motion(person) -> np.ndarray:
    num_frames = int(person.num_frames)
    root = torch.as_tensor(np.asarray(person.root_orient, dtype=np.float32))
    body = torch.as_tensor(np.asarray(person.pose_body, dtype=np.float32))
    lhand = torch.as_tensor(np.asarray(person.pose_lhand, dtype=np.float32))
    rhand = torch.as_tensor(np.asarray(person.pose_rhand, dtype=np.float32))
    zeros_face = torch.zeros((num_frames, 3, 3), dtype=torch.float32)
    pose = torch.cat([root[:, None, :], body.view(num_frames, 21, 3), zeros_face, lhand.view(num_frames, 15, 3), rhand.view(num_frames, 15, 3)], dim=1)
    rot6d = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose)).cpu().numpy().astype(np.float32)
    motion = np.zeros((56, 6, num_frames), dtype=np.float32)
    motion[:55] = np.transpose(rot6d, (1, 2, 0))
    motion[55, :3, :] = np.asarray(person.trans, dtype=np.float32).T
    return motion


def _motion_to_params(
    motion: np.ndarray,
    *,
    template_params: dict[str, Any],
    source_role: str,
    stage2_variant: str,
) -> dict[str, Any]:
    length = int(motion.shape[-1])
    rot6d = np.transpose(motion[:55, :6, :], (2, 0, 1))
    rotvec = rot6d_to_rotvec(rot6d).astype(np.float32)[:length]
    trans = motion[55, :3, :].T.astype(np.float32)[:length]
    params = dict(copy.deepcopy(template_params))
    params.update(
        {
            "root_orient": rotvec[:, 0],
            "pose_body": rotvec[:, 1:22],
            "pose_lhand": rotvec[:, 25:40],
            "pose_rhand": rotvec[:, 40:55],
            "trans": trans,
            "betas": np.asarray(_meta_value(template_params, "betas", np.zeros(10, dtype=np.float32)), dtype=np.float32),
            "gender": str(_meta_value(template_params, "gender", "neutral")),
            "source_role": str(source_role),
            "stage2_variant": str(stage2_variant),
        }
    )
    return params


@dataclass
class ClipInferenceArtifacts:
    actor_motion: np.ndarray
    coarse_motion: np.ndarray
    refined_motion: np.ndarray
    windows: list[dict[str, Any]]
    selector_result: dict[str, Any]
    pack: dict[str, Any]


class _ReactionLikeDict:
    def __init__(self, payload: dict[str, Any]):
        self._payload = dict(payload)
        self.files = tuple(self._payload.keys())

    def __getitem__(self, key: str):
        return self._payload[key]


class _SingleReactionDataset:
    def __init__(self, reaction: dict[str, np.ndarray]):
        self.reaction = _ReactionLikeDict(reaction)


def _build_single_batch(
    *,
    actor_motion: np.ndarray,
    coarse_motion: np.ndarray,
    actor_betas: np.ndarray,
    reactor_betas: np.ndarray,
    actor_gender_id: int,
    reactor_gender_id: int,
    dataset_key: str,
    actor_is_p1: int,
    processed_frame_ix: np.ndarray,
    raw_frame_ix: np.ndarray,
) -> dict[str, Any]:
    length = int(actor_motion.shape[-1])
    return {
        "actor_motion": torch.from_numpy(actor_motion[None]).float(),
        "coarse_motion": torch.from_numpy(coarse_motion[None]).float(),
        "gt_motion": torch.from_numpy(coarse_motion[None]).float(),
        "lengths": torch.as_tensor([length], dtype=torch.long),
        "sample_index": torch.as_tensor([0], dtype=torch.long),
        "dataset_row_index": torch.as_tensor([0], dtype=torch.long),
        "dataset_key": [str(dataset_key)],
        "space_definition": [RESTORED_PAIR_SPACE],
        "actor_betas": torch.from_numpy(actor_betas[None]).float(),
        "reactor_betas": torch.from_numpy(reactor_betas[None]).float(),
        "actor_gender_id": torch.as_tensor([actor_gender_id], dtype=torch.long),
        "reactor_gender_id": torch.as_tensor([reactor_gender_id], dtype=torch.long),
        "body_model_type": ["smplx"],
        "actor_is_p1": torch.as_tensor([actor_is_p1], dtype=torch.long),
        "reactor_is_p2": torch.as_tensor([1 - actor_is_p1], dtype=torch.long),
        "processed_frame_ix": torch.from_numpy(processed_frame_ix[None]).long(),
        "raw_frame_ix": torch.from_numpy(raw_frame_ix[None]).long(),
        "processed_nframes": torch.as_tensor([length], dtype=torch.long),
        "raw_nframes": torch.as_tensor([int(raw_frame_ix[-1]) + 1 if raw_frame_ix.size else length], dtype=torch.long),
        "processed_fps": torch.as_tensor([30], dtype=torch.long),
        "raw_fps": torch.as_tensor([30], dtype=torch.long),
        "downsample": torch.as_tensor([1], dtype=torch.long),
    }


def _run_selector_on_clip(
    *,
    batch_cpu: dict[str, Any],
    region_map: dict[str, np.ndarray],
    device: torch.device,
    tau_contact: float,
    gap_merge: int,
    raw_L_min: int,
    window_size: int,
    per_hand_max_windows: int,
    per_seq_max_windows: int,
    top_k_regions: int,
    frame_chunk: int,
    target_chunk: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    body_forward = RestoredBodyModelForward(device=device)
    batch_dev = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch_cpu.items()
    }
    result = compute_contact_for_batch(
        batch_dev["actor_motion"],
        batch_dev["coarse_motion"],
        batch_dev["lengths"],
        batch_dev,
        region_map,
        tau_contact=tau_contact,
        gap_merge=gap_merge,
        raw_L_min=raw_L_min,
        body_forward=body_forward,
        frame_chunk=frame_chunk,
        target_chunk=target_chunk,
    )
    pred_mask = result["contact_mask"]
    min_dist = result["min_region_dist"]
    _, segments = _build_hand_segments_for_batch(
        pred_mask,
        min_dist,
        result["lengths"],
        result["sample_indices"],
        result["dataset_row_indices"],
        result["dataset_keys"],
        gap_merge=gap_merge,
        raw_L_min=raw_L_min,
        top_k_regions=top_k_regions,
    )
    windows_pre_cap: list[dict[str, Any]] = []
    valid_len = int(result["lengths"][0])
    for seg in segments:
        seg = dict(seg)
        seg["batch_index"] = 0
        windows_pre_cap.extend(windows_from_segment(seg, valid_len, window_size=window_size))
    windows, _ = _limit_windows(
        windows_pre_cap,
        pred_mask,
        per_hand_max_windows=per_hand_max_windows,
        per_seq_max_windows=per_seq_max_windows,
    )
    for idx, item in enumerate(windows):
        item["window_index"] = int(idx)
        item["sequence_window_index"] = int(idx)
    return result, windows


def _build_inference_window_batch(
    *,
    actor_motion: np.ndarray,
    coarse_motion: np.ndarray,
    selector_result: dict[str, Any],
    windows: list[dict[str, Any]],
    action_type: str,
    dataset_key: str,
    geometry_meta: dict[str, Any],
    device: torch.device,
    model_uses_geometry: bool,
) -> list[dict[str, Any]]:
    reaction_pack = {
        "actor_motion": actor_motion[None],
        "reactor_coarse": coarse_motion[None],
        "reactor_gt": coarse_motion[None],
        "lengths": np.asarray([actor_motion.shape[-1]], dtype=np.int64),
        "sample_indices": np.asarray([0], dtype=np.int64),
    }
    manifest_record = {
        "action_type": str(action_type),
        "action_name": str(action_type),
        "action_label": str(action_type),
        "bucket_label": "INFER",
        "dataset_key": str(dataset_key),
        "sample_index": 0,
        "is_gt_positive": False,
        "is_pred_positive": True,
    }
    samples = []
    for item in windows:
        sample = build_window_feature_sample(
            window=item,
            manifest_record=manifest_record,
            reaction_pack=reaction_pack,
            label_index=0,
            selector_index=0,
            gt_contact_mask=selector_result["contact_mask"],
            gt_min_region_dist=selector_result["min_region_dist"],
            pred_contact_mask=selector_result["contact_mask"],
            pred_min_region_dist=selector_result["min_region_dist"],
            strict_checks=False,
        )
        samples.append(sample)
    if not samples:
        return samples
    batch = collate_refine_v2_window_batch(samples)
    batch_dev = batch_to_device(batch, device)
    if model_uses_geometry:
        fake_dataset = _SingleReactionDataset(
            {
                "actor_betas": np.asarray([geometry_meta["actor_betas"]], dtype=np.float32),
                "reactor_betas": np.asarray([geometry_meta["reactor_betas"]], dtype=np.float32),
                "actor_gender_id": np.asarray([geometry_meta["actor_gender_id"]], dtype=np.int64),
                "reactor_gender_id": np.asarray([geometry_meta["reactor_gender_id"]], dtype=np.int64),
                "body_model_type": np.asarray(["smplx"], dtype=object),
            }
        )
        geom = _compute_batch_geometry(
            batch=batch_dev,
            dataset=fake_dataset,
            body_forward=RestoredBodyModelForward(device=device),
            region_map=geometry_meta["region_map"],
            device=device,
        )
        for key, value in geom.items():
            if key.startswith("gt_") or key.endswith("_gap_window") or key == "contact_geometry_weight_window":
                continue
            batch[key] = value.detach().cpu()
    return [batch]


@torch.no_grad()
def _run_refiner_and_stitch(
    *,
    model: RefineV2WindowRefiner,
    window_batches: list[dict[str, Any]],
    coarse_motion: np.ndarray,
) -> np.ndarray:
    refined = coarse_motion.copy()
    delta_sum = np.zeros_like(coarse_motion, dtype=np.float32)
    weight_sum = np.zeros((coarse_motion.shape[-1],), dtype=np.float32)
    for batch in window_batches:
        batch_dev = batch_to_device(batch, next(model.parameters()).device)
        outputs = model(batch_dev)
        pred_delta = outputs["pred_delta_motion_window"].detach().cpu().numpy().astype(np.float32)
        valid_mask = np.asarray(batch["valid_mask"]).astype(bool)
        start_frames = np.asarray(batch["start_frame"]).astype(np.int64)
        for i in range(pred_delta.shape[0]):
            local_valid = np.flatnonzero(valid_mask[i])
            if local_valid.size == 0:
                continue
            global_frames = start_frames[i] + local_valid
            keep = (global_frames >= 0) & (global_frames < coarse_motion.shape[-1])
            if not np.any(keep):
                continue
            global_frames = global_frames[keep]
            local_valid = local_valid[keep]
            local_weights = build_center_weight(pred_delta.shape[-1])[local_valid]
            window_delta = np.take(pred_delta[i], local_valid, axis=-1)
            weighted_delta = window_delta * local_weights.reshape(1, 1, -1)
            for local_idx, frame_idx in enumerate(global_frames.tolist()):
                delta_sum[:, :, int(frame_idx)] += weighted_delta[:, :, local_idx]
            weight_sum[global_frames] += local_weights
    covered = weight_sum > 0
    if np.any(covered):
        refined[:, :, covered] = coarse_motion[:, :, covered] + delta_sum[:, :, covered] / np.maximum(weight_sum[covered], 1e-6).reshape(1, 1, -1)
    return refined.astype(np.float32)


def _write_variant_clip(
    *,
    output_root: str,
    clip_name: str,
    variant: str,
    p1_is_actor: bool,
    actor_motion: np.ndarray,
    reactor_motion: np.ndarray,
    p1_template: dict[str, Any],
    p2_template: dict[str, Any],
):
    clip_dir = os.path.join(output_root, variant, clip_name)
    os.makedirs(clip_dir, exist_ok=True)
    actor_params = _motion_to_params(
        actor_motion,
        template_params=p1_template if p1_is_actor else p2_template,
        source_role="actor",
        stage2_variant=variant,
    )
    reactor_params = _motion_to_params(
        reactor_motion,
        template_params=p2_template if p1_is_actor else p1_template,
        source_role="reactor",
        stage2_variant=variant,
    )
    p1_params, p2_params = (actor_params, reactor_params) if p1_is_actor else (reactor_params, actor_params)
    np.savez(os.path.join(clip_dir, "P1.npz"), **p1_params)
    np.savez(os.path.join(clip_dir, "P2.npz"), **p2_params)


def infer_refiner_on_viewer_clip(
    *,
    checkpoint: str,
    clip_dir: str = "",
    data_dir: str = "",
    clip_name: str = "",
    dataset: str = "interx",
    interaction_order: str = "",
    region_map_path: str = "",
    output_dir: str,
    variant: str = "refined",
    device: str = "cuda",
    tau_contact: float = DEFAULT_TAU_CONTACT,
    gap_merge: int = DEFAULT_GAP_MERGE,
    raw_L_min: int = DEFAULT_RAW_L_MIN,
    window_size: int = DEFAULT_WINDOW_SIZE,
    per_hand_max_windows: int = DEFAULT_PER_HAND_MAX_WINDOWS,
    per_seq_max_windows: int = DEFAULT_PER_SEQ_MAX_WINDOWS,
    top_k_regions: int = DEFAULT_TOP_K_REGIONS,
    frame_chunk: int = 1,
    target_chunk: int = 2048,
    save_pack: bool = True,
) -> dict[str, Any]:
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    state = torch.load(checkpoint, map_location=dev)
    model = RefineV2WindowRefiner(RefineV2WindowRefinerConfig(**state["model_config"])).to(dev)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    resolved_clip_dir = resolve_clip_dir(clip_dir or None, data_dir or None, clip_name or None, dataset or None)
    clip = load_clip(resolved_clip_dir)
    order = load_interaction_order(infer_interaction_order_path(dataset, interaction_order or None))
    role_p1, role_p2 = resolve_person_roles(clip, order_dict=order)
    p1_is_actor = role_p1 == "actor"
    actor_person = clip.p1 if p1_is_actor else clip.p2
    reactor_person = clip.p2 if p1_is_actor else clip.p1
    actor_motion = _person_to_motion(actor_person)
    coarse_motion = _person_to_motion(reactor_person)
    length = min(actor_motion.shape[-1], coarse_motion.shape[-1])
    actor_motion = actor_motion[:, :, :length]
    coarse_motion = coarse_motion[:, :, :length]

    p1_template = dict(clip.p1.params)
    p2_template = dict(clip.p2.params)
    dataset_key = str(_meta_value(actor_person.params, "dataset_key", _meta_value(reactor_person.params, "dataset_key", clip.clip_name)))
    processed_frame_ix = _frame_array(actor_person.params, "processed_frame_ix", length)
    raw_frame_ix = _frame_array(actor_person.params, "raw_frame_ix", length)
    actor_gender = _normalize_gender_name(actor_person.gender)
    reactor_gender = _normalize_gender_name(reactor_person.gender)
    batch_cpu = _build_single_batch(
        actor_motion=actor_motion,
        coarse_motion=coarse_motion,
        actor_betas=np.asarray(actor_person.betas, dtype=np.float32).reshape(-1)[:10],
        reactor_betas=np.asarray(reactor_person.betas, dtype=np.float32).reshape(-1)[:10],
        actor_gender_id=int(GENDER_NAME_TO_ID.get(actor_gender, 0)),
        reactor_gender_id=int(GENDER_NAME_TO_ID.get(reactor_gender, 0)),
        dataset_key=dataset_key,
        actor_is_p1=1 if p1_is_actor else 0,
        processed_frame_ix=processed_frame_ix,
        raw_frame_ix=raw_frame_ix,
    )
    region_map = load_region_map(region_map_path or None)
    selector_result, windows = _run_selector_on_clip(
        batch_cpu=batch_cpu,
        region_map=region_map,
        device=dev,
        tau_contact=tau_contact,
        gap_merge=gap_merge,
        raw_L_min=raw_L_min,
        window_size=window_size,
        per_hand_max_windows=per_hand_max_windows,
        per_seq_max_windows=per_seq_max_windows,
        top_k_regions=top_k_regions,
        frame_chunk=frame_chunk,
        target_chunk=target_chunk,
    )
    action_type = clip.clip_name.split("_", 1)[-1] if "_" in clip.clip_name else clip.clip_name
    geometry_meta = {
        "actor_betas": np.asarray(actor_person.betas, dtype=np.float32).reshape(-1)[:10],
        "reactor_betas": np.asarray(reactor_person.betas, dtype=np.float32).reshape(-1)[:10],
        "actor_gender_id": int(GENDER_NAME_TO_ID.get(actor_gender, 0)),
        "reactor_gender_id": int(GENDER_NAME_TO_ID.get(reactor_gender, 0)),
        "region_map": region_map,
    }
    window_batches = _build_inference_window_batch(
        actor_motion=actor_motion,
        coarse_motion=coarse_motion,
        selector_result=selector_result,
        windows=windows,
        action_type=action_type,
        dataset_key=dataset_key,
        geometry_meta=geometry_meta,
        device=dev,
        model_uses_geometry=bool(state["model_config"].get("use_geometry_features", False)),
    )
    refined_motion = _run_refiner_and_stitch(
        model=model,
        window_batches=window_batches,
        coarse_motion=coarse_motion,
    )
    pack = {
        "actor_motion": actor_motion[None],
        "reactor_coarse": coarse_motion[None],
        "reactor_refined": refined_motion[None],
        "reactor_gt": coarse_motion[None],
        "lengths": np.asarray([length], dtype=np.int64),
        "sample_indices": np.asarray([0], dtype=np.int64),
        "dataset_row_indices": np.asarray([0], dtype=np.int64),
        "dataset_key": np.asarray([dataset_key], dtype=object),
        "action_type": np.asarray([action_type], dtype=object),
        "bucket_label": np.asarray(["INFER"], dtype=object),
        "space_definition": np.asarray([RESTORED_PAIR_SPACE], dtype=object),
        "actor_betas": np.asarray([geometry_meta["actor_betas"]], dtype=np.float32),
        "reactor_betas": np.asarray([geometry_meta["reactor_betas"]], dtype=np.float32),
        "actor_gender_id": np.asarray([geometry_meta["actor_gender_id"]], dtype=np.int64),
        "reactor_gender_id": np.asarray([geometry_meta["reactor_gender_id"]], dtype=np.int64),
        "body_model_type": np.asarray(["smplx"], dtype=object),
        "actor_is_p1": np.asarray([1 if p1_is_actor else 0], dtype=np.int64),
        "reactor_is_p2": np.asarray([0 if p1_is_actor else 1], dtype=np.int64),
        "processed_frame_ix": processed_frame_ix[None],
        "raw_frame_ix": raw_frame_ix[None],
        "processed_nframes": np.asarray([length], dtype=np.int64),
        "raw_nframes": np.asarray([int(raw_frame_ix[-1]) + 1 if raw_frame_ix.size else length], dtype=np.int64),
        "processed_fps": np.asarray([30], dtype=np.int64),
        "raw_fps": np.asarray([30], dtype=np.int64),
        "downsample": np.asarray([1], dtype=np.int64),
    }
    os.makedirs(output_dir, exist_ok=True)
    variants = ["refined", "coarse", "gt"] if variant == "all" else [variant]
    for name in variants:
        reactor_motion = {
            "refined": refined_motion,
            "coarse": coarse_motion,
            "gt": coarse_motion,
        }[name]
        _write_variant_clip(
            output_root=output_dir,
            clip_name=clip.clip_name,
            variant=name,
            p1_is_actor=p1_is_actor,
            actor_motion=actor_motion,
            reactor_motion=reactor_motion,
            p1_template=p1_template,
            p2_template=p2_template,
        )
    pack_path = ""
    if save_pack:
        pack_path = os.path.join(output_dir, "refiner_viewer_infer_pack.npz")
        np.savez_compressed(pack_path, **pack)
    return {
        "clip_name": clip.clip_name,
        "resolved_clip_dir": resolved_clip_dir,
        "output_dir": output_dir,
        "pack_path": pack_path,
        "num_windows": len(windows),
        "length": length,
        "variant": variant,
        "p1_is_actor": bool(p1_is_actor),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Run refine_v2 on one viewer-ready clip and export viewer-ready refined clips.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip_dir", default="")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--clip_name", default="")
    parser.add_argument("--dataset", default="interx", choices=["interx", "chi3d"])
    parser.add_argument("--interaction_order", default="")
    parser.add_argument("--region_map_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", default="refined", choices=["refined", "coarse", "gt", "all"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tau_contact", type=float, default=DEFAULT_TAU_CONTACT)
    parser.add_argument("--gap_merge", type=int, default=DEFAULT_GAP_MERGE)
    parser.add_argument("--raw_L_min", type=int, default=DEFAULT_RAW_L_MIN)
    parser.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--per_hand_max_windows", type=int, default=DEFAULT_PER_HAND_MAX_WINDOWS)
    parser.add_argument("--per_seq_max_windows", type=int, default=DEFAULT_PER_SEQ_MAX_WINDOWS)
    parser.add_argument("--top_k_regions", type=int, default=DEFAULT_TOP_K_REGIONS)
    parser.add_argument("--frame_chunk", type=int, default=1)
    parser.add_argument("--target_chunk", type=int, default=2048)
    parser.add_argument("--no_save_pack", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = infer_refiner_on_viewer_clip(
        checkpoint=args.checkpoint,
        clip_dir=args.clip_dir,
        data_dir=args.data_dir,
        clip_name=args.clip_name,
        dataset=args.dataset,
        interaction_order=args.interaction_order,
        region_map_path=args.region_map_path,
        output_dir=args.output_dir,
        variant=args.variant,
        device=args.device,
        tau_contact=args.tau_contact,
        gap_merge=args.gap_merge,
        raw_L_min=args.raw_L_min,
        window_size=args.window_size,
        per_hand_max_windows=args.per_hand_max_windows,
        per_seq_max_windows=args.per_seq_max_windows,
        top_k_regions=args.top_k_regions,
        frame_chunk=args.frame_chunk,
        target_chunk=args.target_chunk,
        save_pack=not args.no_save_pack,
    )
    print(f"saved viewer-refined clip(s): {result['output_dir']}")
    print(f"clip_name: {result['clip_name']}")
    print(f"num_windows: {result['num_windows']}")
    print(f"length: {result['length']}")
    print(f"p1_is_actor: {result['p1_is_actor']}")
    if result["pack_path"]:
        print(f"saved pack: {result['pack_path']}")


if __name__ == "__main__":
    main()
