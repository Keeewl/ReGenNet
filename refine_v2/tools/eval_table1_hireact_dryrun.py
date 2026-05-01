"""Dry-run table1 HiReact evaluation from prebuilt Stage1 reaction_data packs.

This script is intentionally scoped to the first minimal closure:

1. consume one-seed Stage1 reaction_data for train and test splits;
2. run inference-time selector + Stage2 exp8 refine on the full split;
3. evaluate refined STGCN in canonical space for both train/test.

It does not aggregate multi-seed intervals yet. That is the next step once the
single-seed bridge is validated.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from refine.eval.global_motion import _write_global_csv, evaluate_global_motion
from refine_v2.data.reaction_data import make_reaction_data_loader
from refine_v2.data.schema import RESTORED_PAIR_SPACE, dumps_metadata, to_jsonable
from refine_v2.eval.full_sequence_stitch import stitch_refiner_full_sequences
from refine_v2.model.regions import load_region_map, region_map_summary
from refine_v2.model.selector_v2 import build_windows_for_loader, save_selector_windows
from refine_v2.tools.build_geometry_feature_cache import build_geometry_feature_cache


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _as_str(value.item())
        if value.size == 1:
            return _as_str(value.reshape(-1)[0])
    return str(value)


def _write_json(path: str, payload: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_pack(path: str, pack: dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **pack)


def _build_pseudo_contact_labels(selector_artifact: dict[str, Any], output_path: str) -> str:
    payload = {
        "gt_contact_mask": np.asarray(selector_artifact["pred_contact_mask"], dtype=np.uint8),
        "gt_min_region_dist": np.asarray(selector_artifact["pred_min_region_dist"], dtype=np.float32),
        "lengths": np.asarray(selector_artifact["lengths"], dtype=np.int64),
        "dataset_row_indices": np.asarray(selector_artifact["dataset_row_indices"], dtype=np.int64),
        "space_definition": np.asarray(RESTORED_PAIR_SPACE),
        "metadata_json": np.asarray(
            dumps_metadata(
                {
                    "artifact": "table1_hireact_dryrun_pseudo_contact_labels",
                    "description": "Pseudo GT labels copied from selector coarse contact for Stage2 inference-only dry-run.",
                    "space_definition": RESTORED_PAIR_SPACE,
                }
            )
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def _build_full_manifest(reaction_data_path: str, output_path: str, *, bucket_label: str = "ALL") -> str:
    reaction = np.load(reaction_data_path, allow_pickle=True)
    lengths = np.asarray(reaction["lengths"], dtype=np.int64)
    sample_indices = np.asarray(reaction["sample_indices"], dtype=np.int64)
    dataset_keys = (
        np.asarray(reaction["dataset_key"], dtype=object)
        if "dataset_key" in reaction.files
        else np.asarray([f"sample_{i}" for i in range(int(lengths.shape[0]))], dtype=object)
    )
    sequences = []
    for row in range(int(lengths.shape[0])):
        key = _as_str(dataset_keys[row])
        sequences.append(
            {
                "dataset_row_index": int(row),
                "sample_index": int(sample_indices[row]),
                "dataset_key": key,
                "action_type": key,
                "action_name": key,
                "action_label": key,
                "bucket_label": str(bucket_label),
                "is_gt_positive": False,
                "is_pred_positive": True,
                "length": int(lengths[row]),
            }
        )
    payload = {
        "artifact": "table1_hireact_dryrun_full_manifest",
        "bucket_label": str(bucket_label),
        "num_sequences": int(len(sequences)),
        "sequences": sequences,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return output_path


def _prepare_split_artifacts(
    *,
    reaction_data_path: str,
    split_name: str,
    work_dir: str,
    region_map_path: str,
    device: str,
    selector_batch_size: int,
    selector_num_workers: int,
    selector_tau_contact: float,
    selector_gap_merge: int,
    selector_raw_L_min: int,
    selector_window_size: int,
    selector_per_hand_max_windows: int,
    selector_per_seq_max_windows: int,
    selector_top_k_regions: int,
    frame_chunk: int,
    target_chunk: int,
    geometry_batch_size: int,
    geometry_num_workers: int,
) -> dict[str, str]:
    os.makedirs(work_dir, exist_ok=True)

    selector_windows_path = os.path.join(work_dir, f"{split_name}_selector_windows.npz")
    pseudo_labels_path = os.path.join(work_dir, f"{split_name}_pseudo_contact_labels.npz")
    manifest_path = os.path.join(work_dir, f"{split_name}_full_manifest.json")
    geometry_cache_path = os.path.join(work_dir, f"{split_name}_geometry_feature_cache.npz")

    region_map = load_region_map(region_map_path or None)
    loader = make_reaction_data_loader(
        reaction_data_path,
        batch_size=int(selector_batch_size),
        num_workers=int(selector_num_workers),
    )
    selector_artifact = build_windows_for_loader(
        loader,
        region_map,
        tau_contact=selector_tau_contact,
        gap_merge=selector_gap_merge,
        raw_L_min=selector_raw_L_min,
        window_size=selector_window_size,
        per_hand_max_windows=selector_per_hand_max_windows,
        per_seq_max_windows=selector_per_seq_max_windows,
        top_k_regions=selector_top_k_regions,
        device=device,
        frame_chunk=frame_chunk,
        target_chunk=target_chunk,
        show_progress=True,
    )
    save_selector_windows(selector_windows_path, selector_artifact)
    _build_pseudo_contact_labels(selector_artifact, pseudo_labels_path)
    _build_full_manifest(reaction_data_path, manifest_path, bucket_label="ALL")
    build_geometry_feature_cache(
        reaction_data_path=reaction_data_path,
        contact_labels_path=pseudo_labels_path,
        subset_manifest_path=manifest_path,
        selector_windows_path=selector_windows_path,
        output_path=geometry_cache_path,
        region_map_path=region_map_path,
        include_buckets=["ALL"],
        selected_action_types=None,
        batch_size=int(geometry_batch_size),
        num_workers=int(geometry_num_workers),
        device=device,
        progress=True,
    )
    return {
        "selector_windows_path": selector_windows_path,
        "pseudo_labels_path": pseudo_labels_path,
        "manifest_path": manifest_path,
        "geometry_cache_path": geometry_cache_path,
        "selector_stats_json": str(_as_str(selector_artifact.get("selector_stats_json", ""))),
    }


def _run_split(
    *,
    split_name: str,
    reaction_data_path: str,
    checkpoint_path: str,
    region_map_path: str,
    stgcn_model_path: str,
    output_dir: str,
    device: str,
    stitch_batch_size: int,
    stitch_num_workers: int,
    stgcn_batch_size: int,
    selector_batch_size: int,
    selector_num_workers: int,
    selector_tau_contact: float,
    selector_gap_merge: int,
    selector_raw_L_min: int,
    selector_window_size: int,
    selector_per_hand_max_windows: int,
    selector_per_seq_max_windows: int,
    selector_top_k_regions: int,
    frame_chunk: int,
    target_chunk: int,
    geometry_batch_size: int,
    geometry_num_workers: int,
) -> dict[str, Any]:
    artifacts = _prepare_split_artifacts(
        reaction_data_path=reaction_data_path,
        split_name=split_name,
        work_dir=os.path.join(output_dir, split_name),
        region_map_path=region_map_path,
        device=device,
        selector_batch_size=selector_batch_size,
        selector_num_workers=selector_num_workers,
        selector_tau_contact=selector_tau_contact,
        selector_gap_merge=selector_gap_merge,
        selector_raw_L_min=selector_raw_L_min,
        selector_window_size=selector_window_size,
        selector_per_hand_max_windows=selector_per_hand_max_windows,
        selector_per_seq_max_windows=selector_per_seq_max_windows,
        selector_top_k_regions=selector_top_k_regions,
        frame_chunk=frame_chunk,
        target_chunk=target_chunk,
        geometry_batch_size=geometry_batch_size,
        geometry_num_workers=geometry_num_workers,
    )

    stitched = stitch_refiner_full_sequences(
        checkpoint_path=checkpoint_path,
        reaction_data_path=reaction_data_path,
        contact_labels_path=artifacts["pseudo_labels_path"],
        subset_manifest_path=artifacts["manifest_path"],
        selector_windows_path=artifacts["selector_windows_path"],
        include_buckets=["ALL"],
        geometry_feature_cache_path=artifacts["geometry_cache_path"],
        selected_action_types=None,
        max_sequences_per_action_type=0,
        sample_seed=0,
        batch_size=stitch_batch_size,
        num_workers=stitch_num_workers,
        device=device,
    )
    pack = stitched["pack"]
    pack_path = os.path.join(output_dir, split_name, f"{split_name}_refined_pack.npz")
    _write_pack(pack_path, pack)

    stgcn_payload = evaluate_global_motion(
        pack,
        dataset="interx",
        stgcn_model_path=stgcn_model_path,
        body_model="smplx",
        batch_size=stgcn_batch_size,
        device=device,
        num_classes=None,
    )
    stgcn_json = os.path.join(output_dir, split_name, f"{split_name}_stgcn.json")
    stgcn_csv = os.path.join(output_dir, split_name, f"{split_name}_stgcn.csv")
    _write_json(stgcn_json, stgcn_payload)
    _write_global_csv(stgcn_csv, stgcn_payload)

    return {
        "split": split_name,
        "reaction_data_path": reaction_data_path,
        "artifacts": artifacts,
        "stitch_summary": stitched["summary"],
        "num_sequences": int(np.asarray(pack["lengths"]).shape[0]),
        "stgcn_metrics": stgcn_payload,
        "pack_path": pack_path,
        "stgcn_json": stgcn_json,
        "stgcn_csv": stgcn_csv,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run one-seed table1 HiReact evaluation from prebuilt train/test reaction_data.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--train_reaction_data_path", required=True, type=str)
    parser.add_argument("--test_reaction_data_path", required=True, type=str)
    parser.add_argument("--region_map_path", required=True, type=str)
    parser.add_argument("--stgcn_model_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--selector_batch_size", default=32, type=int)
    parser.add_argument("--selector_num_workers", default=0, type=int)
    parser.add_argument("--selector_tau_contact", default=0.10, type=float)
    parser.add_argument("--selector_gap_merge", default=4, type=int)
    parser.add_argument("--selector_raw_L_min", default=2, type=int)
    parser.add_argument("--selector_window_size", default=30, type=int)
    parser.add_argument("--selector_per_hand_max_windows", default=2, type=int)
    parser.add_argument("--selector_per_seq_max_windows", default=3, type=int)
    parser.add_argument("--selector_top_k_regions", default=3, type=int)
    parser.add_argument("--stitch_batch_size", default=32, type=int)
    parser.add_argument("--stitch_num_workers", default=0, type=int)
    parser.add_argument("--stgcn_batch_size", default=64, type=int)
    parser.add_argument("--geometry_batch_size", default=32, type=int)
    parser.add_argument("--geometry_num_workers", default=0, type=int)
    parser.add_argument("--frame_chunk", default=1, type=int)
    parser.add_argument("--target_chunk", default=2048, type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    train_result = _run_split(
        split_name="train",
        reaction_data_path=args.train_reaction_data_path,
        checkpoint_path=args.checkpoint,
        region_map_path=args.region_map_path,
        stgcn_model_path=args.stgcn_model_path,
        output_dir=args.output_dir,
        device=args.device,
        stitch_batch_size=args.stitch_batch_size,
        stitch_num_workers=args.stitch_num_workers,
        stgcn_batch_size=args.stgcn_batch_size,
        selector_batch_size=args.selector_batch_size,
        selector_num_workers=args.selector_num_workers,
        selector_tau_contact=args.selector_tau_contact,
        selector_gap_merge=args.selector_gap_merge,
        selector_raw_L_min=args.selector_raw_L_min,
        selector_window_size=args.selector_window_size,
        selector_per_hand_max_windows=args.selector_per_hand_max_windows,
        selector_per_seq_max_windows=args.selector_per_seq_max_windows,
        selector_top_k_regions=args.selector_top_k_regions,
        frame_chunk=args.frame_chunk,
        target_chunk=args.target_chunk,
        geometry_batch_size=args.geometry_batch_size,
        geometry_num_workers=args.geometry_num_workers,
    )
    test_result = _run_split(
        split_name="test",
        reaction_data_path=args.test_reaction_data_path,
        checkpoint_path=args.checkpoint,
        region_map_path=args.region_map_path,
        stgcn_model_path=args.stgcn_model_path,
        output_dir=args.output_dir,
        device=args.device,
        stitch_batch_size=args.stitch_batch_size,
        stitch_num_workers=args.stitch_num_workers,
        stgcn_batch_size=args.stgcn_batch_size,
        selector_batch_size=args.selector_batch_size,
        selector_num_workers=args.selector_num_workers,
        selector_tau_contact=args.selector_tau_contact,
        selector_gap_merge=args.selector_gap_merge,
        selector_raw_L_min=args.selector_raw_L_min,
        selector_window_size=args.selector_window_size,
        selector_per_hand_max_windows=args.selector_per_hand_max_windows,
        selector_per_seq_max_windows=args.selector_per_seq_max_windows,
        selector_top_k_regions=args.selector_top_k_regions,
        frame_chunk=args.frame_chunk,
        target_chunk=args.target_chunk,
        geometry_batch_size=args.geometry_batch_size,
        geometry_num_workers=args.geometry_num_workers,
    )

    summary = {
        "artifact": "table1_hireact_dryrun",
        "checkpoint": args.checkpoint,
        "protocol": {
            "description": "Single-seed dry-run of table1 HiReact: prebuilt Stage1 reaction_data -> inference-time selector -> Stage2 -> STGCN.",
            "full_dataset": True,
            "space_definition": {
                "stage1_reaction_data": RESTORED_PAIR_SPACE,
                "stgcn_eval": "canonical_after_inverse_restore",
            },
            "selector": {
                "tau_contact": args.selector_tau_contact,
                "gap_merge": args.selector_gap_merge,
                "raw_L_min": args.selector_raw_L_min,
                "window_size": args.selector_window_size,
                "per_hand_max_windows": args.selector_per_hand_max_windows,
                "per_seq_max_windows": args.selector_per_seq_max_windows,
                "top_k_regions": args.selector_top_k_regions,
            },
        },
        "train": train_result,
        "test": test_result,
        "table1_row_preview": {
            "train_conditioned": to_jsonable(train_result["stgcn_metrics"]["refined"]),
            "test_conditioned": to_jsonable(test_result["stgcn_metrics"]["refined"]),
        },
    }
    summary_path = os.path.join(args.output_dir, "table1_hireact_dryrun_summary.json")
    _write_json(summary_path, summary)
    print(f"saved dry-run summary: {summary_path}")
    print("train refined:", json.dumps(to_jsonable(train_result["stgcn_metrics"]["refined"]), ensure_ascii=False))
    print("test refined:", json.dumps(to_jsonable(test_result["stgcn_metrics"]["refined"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
