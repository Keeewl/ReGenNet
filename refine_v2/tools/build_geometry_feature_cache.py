"""CLI/tool for offline refine_v2 relative geometry feature caches."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from refine_v2.data.schema import HAND_SIDE_NAMES, RESTORED_PAIR_SPACE, TARGET_REGION_NAMES, to_jsonable
from refine_v2.data.restored_space import RestoredBodyModelForward
from refine_v2.model.regions import load_region_map, region_map_summary
from refine_v2.refiner_data.window_dataset import RefineV2WindowDataset
from refine_v2.refiner_data.window_loader import collate_refine_v2_window_batch
from refine_v2.utils.progress import ProgressBar


def _as_tensor(value, *, device, dtype=None):
    out = value if torch.is_tensor(value) else torch.as_tensor(value)
    out = out.to(device=device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _metadata_for_rows(dataset: RefineV2WindowDataset, rows: torch.Tensor, *, device, dtype) -> dict[str, Any]:
    rows_np = rows.detach().cpu().numpy().astype(np.int64).reshape(-1)
    reaction = dataset.reaction
    required = ("actor_betas", "reactor_betas", "actor_gender_id", "reactor_gender_id", "body_model_type")
    missing = [key for key in required if key not in reaction.files]
    if missing:
        raise KeyError("geometry feature cache requires body metadata in reaction_data: " + ", ".join(missing))
    max_row = int(rows_np.max()) if rows_np.size else 0

    def take_rows(key: str):
        arr = np.asarray(reaction[key])
        if arr.shape == ():
            return np.repeat(arr.reshape(1), rows_np.shape[0], axis=0)
        if arr.shape[0] == 1 and max_row >= 1:
            return np.repeat(arr, rows_np.shape[0], axis=0)
        return arr[rows_np]

    body_model_type_values = take_rows("body_model_type")
    first = body_model_type_values.reshape(-1)[0]
    if isinstance(first, bytes):
        first = first.decode("utf-8")
    body_model_type = str(first).lower()
    if body_model_type != "smplx":
        raise ValueError(f"geometry feature cache currently supports body_model_type=smplx, got {body_model_type}")
    return {
        "actor_betas": _as_tensor(take_rows("actor_betas"), device=device, dtype=dtype),
        "reactor_betas": _as_tensor(take_rows("reactor_betas"), device=device, dtype=dtype),
        "actor_gender_id": _as_tensor(take_rows("actor_gender_id"), device=device, dtype=torch.long).view(-1),
        "reactor_gender_id": _as_tensor(take_rows("reactor_gender_id"), device=device, dtype=torch.long).view(-1),
        "body_model_type": body_model_type,
    }


@torch.no_grad()
def _motion_to_vertices(
    body_forward: RestoredBodyModelForward,
    motion: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    betas: torch.Tensor,
    gender_id: torch.Tensor,
    body_model_type: str,
) -> torch.Tensor:
    return body_forward.motion_to_xyz(
        motion,
        jointstype="vertices",
        betas=betas,
        gender_id=gender_id,
        mask=valid_mask.bool(),
        body_model_type=body_model_type,
    )


def _centroid(vertices: torch.Tensor, ids: np.ndarray) -> torch.Tensor:
    index = torch.as_tensor(np.asarray(ids, dtype=np.int64), device=vertices.device, dtype=torch.long)
    return vertices.index_select(1, index).mean(dim=1)


@torch.no_grad()
def _compute_batch_geometry(
    *,
    batch: dict[str, Any],
    dataset: RefineV2WindowDataset,
    body_forward: RestoredBodyModelForward,
    region_map: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    valid_mask = batch["valid_mask"].bool()
    meta = _metadata_for_rows(dataset, batch["dataset_row_index"], device=device, dtype=batch["actor_motion_window"].dtype)
    actor_vertices = _motion_to_vertices(
        body_forward,
        batch["actor_motion_window"].float(),
        valid_mask,
        betas=meta["actor_betas"],
        gender_id=meta["actor_gender_id"],
        body_model_type=meta["body_model_type"],
    )
    reactor_vertices = _motion_to_vertices(
        body_forward,
        batch["coarse_motion_window"].float(),
        valid_mask,
        betas=meta["reactor_betas"],
        gender_id=meta["reactor_gender_id"],
        body_model_type=meta["body_model_type"],
    )

    bsz = int(actor_vertices.shape[0])
    num_frames = int(actor_vertices.shape[-1])
    hand_centroid = actor_vertices.new_zeros((bsz, 3, num_frames))
    for hand_id, hand_side in enumerate(HAND_SIDE_NAMES):
        idx = torch.nonzero(batch["hand_side_id"].long() == hand_id, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        sub_centroid = _centroid(reactor_vertices.index_select(0, idx), region_map[f"{hand_side}_hand"])
        hand_centroid.index_copy_(0, idx, sub_centroid)

    actor_region_centroids = torch.stack(
        [_centroid(actor_vertices, region_map[name]) for name in TARGET_REGION_NAMES],
        dim=1,
    )
    primary_ids = batch["primary_target_region_id"].long().clamp(0, len(TARGET_REGION_NAMES) - 1)
    batch_ids = torch.arange(bsz, device=device)
    primary_target = actor_region_centroids[batch_ids, primary_ids]

    topk_ids = batch["topk_target_region_ids"].long().clamp(0, len(TARGET_REGION_NAMES) - 1)
    topk_target = actor_vertices.new_zeros((bsz, int(topk_ids.shape[1]), 3, num_frames))
    for region_id in range(len(TARGET_REGION_NAMES)):
        pos = torch.nonzero(topk_ids == region_id, as_tuple=False)
        if pos.numel() == 0:
            continue
        row_ids = pos[:, 0]
        topk_pos = pos[:, 1]
        topk_target[row_ids, topk_pos] = actor_region_centroids[row_ids, region_id]

    primary_vec = primary_target - hand_centroid
    topk_vec = topk_target - hand_centroid[:, None, :, :]
    return {
        "primary_relative_vector_window": primary_vec,
        "primary_relative_dist_window": torch.linalg.norm(primary_vec, dim=1),
        "topk_relative_vectors_window": topk_vec,
        "topk_relative_dists_window": torch.linalg.norm(topk_vec, dim=2),
    }


def build_geometry_feature_cache(
    *,
    reaction_data_path: str,
    contact_labels_path: str,
    subset_manifest_path: str,
    selector_windows_path: str,
    output_path: str,
    region_map_path: str = "",
    include_buckets: list[str] | None = None,
    selected_action_types: list[str] | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cuda",
    progress: bool = True,
) -> dict[str, Any]:
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    dataset = RefineV2WindowDataset(
        reaction_data_path,
        contact_labels_path,
        subset_manifest_path,
        selector_windows_path,
        include_buckets=include_buckets or ["GT+ / Pred+"],
        selected_action_types=selected_action_types,
        strict_checks=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=dev.type == "cuda",
        collate_fn=collate_refine_v2_window_batch,
    )
    region_map = load_region_map(region_map_path or None)
    body_forward = RestoredBodyModelForward(device=dev)
    chunks: dict[str, list[np.ndarray]] = {
        "primary_relative_vector_window": [],
        "primary_relative_dist_window": [],
        "topk_relative_vectors_window": [],
        "topk_relative_dists_window": [],
    }
    bar = ProgressBar("build_geometry_cache", total=len(dataset), unit="windows", enabled=progress).start()
    for batch in loader:
        batch_dev = {
            key: value.to(dev, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        geom = _compute_batch_geometry(
            batch=batch_dev,
            dataset=dataset,
            body_forward=body_forward,
            region_map=region_map,
            device=dev,
        )
        bsz = int(batch_dev["dataset_row_index"].shape[0])
        for key, value in geom.items():
            chunks[key].append(value.detach().cpu().numpy().astype(np.float32))
        bar.update(bsz)
    bar.finish()

    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    records = dataset.window_records
    metadata = {
        "artifact": "refine_v2_geometry_feature_cache",
        "description": "Offline actor-target centroid minus coarse-reactor selected-hand centroid features.",
        "paths": {
            "reaction_data_path": reaction_data_path,
            "contact_labels_path": contact_labels_path,
            "subset_manifest_path": subset_manifest_path,
            "selector_windows_path": selector_windows_path,
            "region_map_path": str(region_map_path or ""),
        },
        "params": {
            "include_buckets": list(include_buckets or ["GT+ / Pred+"]),
            "selected_action_types": list(selected_action_types or []),
            "batch_size": int(batch_size),
            "device": str(dev),
        },
        "space_definition": RESTORED_PAIR_SPACE,
        "num_windows": int(len(records)),
        "region_map_summary": region_map_summary(region_map),
        "feature_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "notes": [
            "Features are computed from actor_motion and reactor_coarse only.",
            "No GT reactor motion is used, so the cache is safe as model input.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        dataset_row_indices=np.asarray([int(w["dataset_row_index"]) for w in records], dtype=np.int64),
        sample_indices=np.asarray([int(w.get("sample_index", w["dataset_row_index"])) for w in records], dtype=np.int64),
        window_indices=np.asarray([int(w["window_index"]) for w in records], dtype=np.int64),
        sequence_window_indices=np.asarray([int(w.get("sequence_window_index", -1)) for w in records], dtype=np.int64),
        start_frames=np.asarray([int(w["start_frame"]) for w in records], dtype=np.int64),
        end_frames=np.asarray([int(w["end_frame"]) for w in records], dtype=np.int64),
        hand_side_ids=np.asarray([int(w["hand_side_id"]) for w in records], dtype=np.int64),
        primary_target_region_ids=np.asarray([int(w["primary_target_region_id"]) for w in records], dtype=np.int64),
        topk_target_region_ids=np.asarray([list(map(int, w.get("topk_target_region_ids", []))) for w in records], dtype=np.int64),
        dataset_keys=np.asarray([str(w.get("dataset_key", "")) for w in records], dtype=object),
        space_definition=np.asarray(RESTORED_PAIR_SPACE),
        metadata_json=np.asarray(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True)),
    )
    metadata_path = os.path.splitext(output_path)[0] + "_summary.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(metadata), f, indent=2, sort_keys=True)
    return {"output_path": output_path, "summary_path": metadata_path, **metadata}


def build_parser():
    parser = argparse.ArgumentParser(description="Build offline refine_v2 relative geometry feature cache.")
    parser.add_argument("--reaction_data_path", required=True)
    parser.add_argument("--contact_labels_path", required=True)
    parser.add_argument("--subset_manifest_path", required=True)
    parser.add_argument("--selector_windows_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--region_map_path", default="")
    parser.add_argument("--include_buckets", nargs="+", default=["GT+ / Pred+"])
    parser.add_argument("--selected_action_types", nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_progress", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = build_geometry_feature_cache(
        reaction_data_path=args.reaction_data_path,
        contact_labels_path=args.contact_labels_path,
        subset_manifest_path=args.subset_manifest_path,
        selector_windows_path=args.selector_windows_path,
        output_path=args.output_path,
        region_map_path=args.region_map_path,
        include_buckets=args.include_buckets,
        selected_action_types=args.selected_action_types,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        progress=not args.no_progress,
    )
    print(f"saved geometry feature cache: {result['output_path']}")
    print(f"saved summary: {result['summary_path']}")
    print(f"num_windows: {result['num_windows']}")
    print(f"feature_shapes: {result['feature_shapes']}")


if __name__ == "__main__":
    main()
