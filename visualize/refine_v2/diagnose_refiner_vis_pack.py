"""Diagnose whether refine_v2 contact gaps are transl- or hand-pose-limited.

This script reads a refiner visualization pack exported by
`python -m refine_v2.cli_export_refiner_vis_pack`. It does not run the refiner.
It forwards actor/coarse/refined/GT motions through SMPL-X, measures window-level
hand-target distances, translation error, and root-local hand error, then labels
each window with a simple diagnostic bucket.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

from refine_v2.data.restored_space import RestoredBodyModelForward, lengths_to_mask
from refine_v2.data.schema import HAND_SIDE_NAMES, TARGET_REGION_IDS, TARGET_REGION_NAMES, to_jsonable
from refine_v2.model.regions import load_region_map, region_map_summary
from refine_v2.subset.reporting import markdown_table, write_json


MOTION_FIELDS = {
    "coarse": "reactor_coarse_motion",
    "refined": "reactor_refined_motion",
    "gt": "reactor_gt_motion",
}


def _load_pack(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    pack = {key: data[key] for key in data.files}
    manifest = {}
    if "manifest_json" in pack:
        item = pack["manifest_json"].item()
        if isinstance(item, str):
            manifest = json.loads(item)
        else:
            manifest = dict(item)
    return pack, manifest


def _as_tensor(value, *, device, dtype=None):
    out = value if torch.is_tensor(value) else torch.as_tensor(value)
    out = out.to(device=device)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _body_model_type_values(pack: dict[str, Any], start: int, end: int) -> str:
    values = np.asarray(pack.get("body_model_type", np.asarray(["smplx"])))
    if values.shape == ():
        item = values.item()
    else:
        item = values.reshape(-1)[start]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    body_model_type = str(item).lower()
    if body_model_type != "smplx":
        raise ValueError(f"Only body_model_type=smplx is supported, got {body_model_type}")
    return body_model_type


@torch.no_grad()
def _motion_vertices(
    body_forward: RestoredBodyModelForward,
    motion: np.ndarray,
    *,
    start: int,
    end: int,
    lengths: np.ndarray,
    betas: np.ndarray,
    gender_id: np.ndarray,
    body_model_type: str,
    device: torch.device,
) -> torch.Tensor:
    x = _as_tensor(motion[start:end], device=device, dtype=torch.float32)
    lens = _as_tensor(lengths[start:end], device=device, dtype=torch.long)
    valid = lengths_to_mask(lens, int(x.shape[-1]))
    return body_forward.motion_to_xyz(
        x,
        jointstype="vertices",
        betas=_as_tensor(betas[start:end], device=device, dtype=torch.float32),
        gender_id=_as_tensor(gender_id[start:end], device=device, dtype=torch.long).view(-1),
        mask=valid,
        body_model_type=body_model_type,
    )


def _region_centroids(vertices: torch.Tensor, region_map: dict[str, np.ndarray]) -> torch.Tensor:
    centroids = []
    for name in TARGET_REGION_NAMES:
        ids = torch.as_tensor(region_map[name], device=vertices.device, dtype=torch.long)
        centroids.append(vertices.index_select(1, ids).mean(dim=1))
    return torch.stack(centroids, dim=1)


def _hand_centroid(vertices: torch.Tensor, region_map: dict[str, np.ndarray], hand_side: str) -> torch.Tensor:
    ids = torch.as_tensor(region_map[f"{hand_side}_hand"], device=vertices.device, dtype=torch.long)
    return vertices.index_select(1, ids).mean(dim=1)


def _safe_mean(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(value.float().mean().detach().cpu().item())


def _window_frames(window: dict[str, Any], length: int) -> tuple[int, int]:
    start = max(0, int(window.get("start_frame", 0)))
    end = min(int(window.get("end_frame", length)), int(length))
    if end <= start:
        raise ValueError(f"Invalid window bounds [{start},{end}) for length={length}")
    return start, end


def _topk_region_ids(window: dict[str, Any]) -> list[int]:
    ids = [int(x) for x in window.get("topk_target_region_ids", [])]
    if ids:
        return [x for x in ids if 0 <= x < len(TARGET_REGION_NAMES)]
    names = [str(x) for x in window.get("topk_target_regions", [])]
    out = [TARGET_REGION_IDS[name] for name in names if name in TARGET_REGION_IDS]
    if out:
        return out
    primary = int(window.get("primary_target_region_id", window.get("target_region_id", -1)))
    return [primary] if 0 <= primary < len(TARGET_REGION_NAMES) else [0]


def _classify_window(
    *,
    refined_contact_gap: float,
    refined_transl_error: float,
    refined_local_hand_error: float,
    transl_error_high: float,
    local_hand_error_high: float,
    contact_gap_high: float,
) -> str:
    if refined_contact_gap <= contact_gap_high:
        return "already_good"
    transl_high = refined_transl_error > transl_error_high
    hand_high = refined_local_hand_error > local_hand_error_high
    if transl_high and not hand_high:
        return "transl_issue"
    if hand_high and not transl_high:
        return "hand_pose_issue"
    if transl_high and hand_high:
        return "mixed_issue"
    return "metric_or_region_issue"


def _diagnose_window(
    *,
    sequence: dict[str, Any],
    sequence_index: int,
    local_index: int,
    local_vertices: dict[str, torch.Tensor],
    actor_region_centroids: torch.Tensor,
    pack: dict[str, Any],
    local_batch_index: int,
    window: dict[str, Any],
    length: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    start, end = _window_frames(window, length)
    frames = slice(start, end)
    hand_side = str(window.get("hand_side", ""))
    if hand_side not in HAND_SIDE_NAMES:
        hand_side = HAND_SIDE_NAMES[int(window.get("hand_side_id", 0))]

    primary_id = int(window.get("primary_target_region_id", window.get("target_region_id", _topk_region_ids(window)[0])))
    primary_id = max(0, min(primary_id, len(TARGET_REGION_NAMES) - 1))
    topk_ids = _topk_region_ids(window)

    target_primary = actor_region_centroids[local_batch_index, primary_id, :, frames]
    target_topk = actor_region_centroids[local_batch_index, topk_ids, :, frames]

    metrics: dict[str, float] = {}
    hand_centroids = {
        name: vertices[local_batch_index, :, frames]
        for name, vertices in local_vertices.items()
    }

    row_seq = int(sequence["sequence_index"])
    trans = {
        name: _as_tensor(pack[field][row_seq, 55, :3, frames], device=target_primary.device, dtype=torch.float32)
        for name, field in MOTION_FIELDS.items()
    }

    gt_hand = hand_centroids["gt"]
    gt_local = gt_hand - trans["gt"]
    gt_primary_dist = torch.linalg.norm(gt_hand - target_primary, dim=0)
    gt_topk_dist = torch.linalg.norm(gt_hand[None, :, :] - target_topk, dim=1).amin(dim=0)
    metrics["gt_primary_hand_target_dist"] = _safe_mean(gt_primary_dist)
    metrics["gt_topk_hand_target_dist"] = _safe_mean(gt_topk_dist)

    for name in ("coarse", "refined"):
        hand = hand_centroids[name]
        local = hand - trans[name]
        primary_dist = torch.linalg.norm(hand - target_primary, dim=0)
        topk_dist = torch.linalg.norm(hand[None, :, :] - target_topk, dim=1).amin(dim=0)
        transl_err = torch.linalg.norm(trans[name] - trans["gt"], dim=0)
        local_hand_err = torch.linalg.norm(local - gt_local, dim=0)
        metrics[f"{name}_primary_hand_target_dist"] = _safe_mean(primary_dist)
        metrics[f"{name}_topk_hand_target_dist"] = _safe_mean(topk_dist)
        metrics[f"{name}_transl_error"] = _safe_mean(transl_err)
        metrics[f"{name}_local_hand_error"] = _safe_mean(local_hand_err)
        metrics[f"{name}_primary_gap_to_gt"] = metrics[f"{name}_primary_hand_target_dist"] - metrics["gt_primary_hand_target_dist"]
        metrics[f"{name}_topk_gap_to_gt"] = metrics[f"{name}_topk_hand_target_dist"] - metrics["gt_topk_hand_target_dist"]

    metrics["topk_dist_improvement_coarse_to_refined"] = (
        metrics["coarse_topk_hand_target_dist"] - metrics["refined_topk_hand_target_dist"]
    )
    metrics["primary_dist_improvement_coarse_to_refined"] = (
        metrics["coarse_primary_hand_target_dist"] - metrics["refined_primary_hand_target_dist"]
    )
    metrics["transl_error_change_coarse_to_refined"] = metrics["coarse_transl_error"] - metrics["refined_transl_error"]
    metrics["local_hand_error_change_coarse_to_refined"] = (
        metrics["coarse_local_hand_error"] - metrics["refined_local_hand_error"]
    )

    label = _classify_window(
        refined_contact_gap=metrics["refined_topk_gap_to_gt"],
        refined_transl_error=metrics["refined_transl_error"],
        refined_local_hand_error=metrics["refined_local_hand_error"],
        transl_error_high=thresholds["transl_error_high"],
        local_hand_error_high=thresholds["local_hand_error_high"],
        contact_gap_high=thresholds["contact_gap_high"],
    )
    return {
        "sequence_index": int(sequence_index),
        "dataset_row_index": int(sequence.get("dataset_row_index", -1)),
        "dataset_key": str(sequence.get("dataset_key", "")),
        "action_type": str(sequence.get("action_type", "")),
        "sequence_window_index": int(window.get("sequence_window_index", local_index)),
        "window_index": int(window.get("window_index", -1)),
        "start_frame": int(start),
        "end_frame": int(end),
        "hand_side": hand_side,
        "primary_target_region": str(window.get("primary_target_region", TARGET_REGION_NAMES[primary_id])),
        "topk_target_regions": list(window.get("topk_target_regions", [TARGET_REGION_NAMES[idx] for idx in topk_ids])),
        "diagnosis": label,
        "metrics": metrics,
    }


class _Averager:
    def __init__(self):
        self.sum: dict[str, float] = defaultdict(float)
        self.count: dict[str, int] = defaultdict(int)
        self.labels: Counter[str] = Counter()

    def add(self, item: dict[str, Any]):
        self.labels.update([str(item.get("diagnosis", ""))])
        for key, value in item.get("metrics", {}).items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                self.sum[key] += float(value)
                self.count[key] += 1

    def finalize(self) -> dict[str, Any]:
        out = {key: self.sum[key] / max(1, self.count[key]) for key in sorted(self.sum)}
        total = sum(self.labels.values())
        out.update({f"diagnosis_count_{key}": int(value) for key, value in sorted(self.labels.items())})
        out.update({f"diagnosis_ratio_{key}": float(value) / max(1, total) for key, value in sorted(self.labels.items())})
        out["num_windows"] = int(total)
        return out


def _summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _Averager()
    by_action: dict[str, _Averager] = defaultdict(_Averager)
    by_hand: dict[str, _Averager] = defaultdict(_Averager)
    by_region: dict[str, _Averager] = defaultdict(_Averager)
    for item in items:
        overall.add(item)
        by_action[str(item.get("action_type", ""))].add(item)
        by_hand[str(item.get("hand_side", ""))].add(item)
        by_region[str(item.get("primary_target_region", ""))].add(item)
    return {
        "overall": overall.finalize(),
        "by_action_type": {key: avg.finalize() for key, avg in sorted(by_action.items())},
        "by_hand_side": {key: avg.finalize() for key, avg in sorted(by_hand.items())},
        "by_primary_target_region": {key: avg.finalize() for key, avg in sorted(by_region.items())},
    }


def _write_csv(path: str, rows: list[dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "sequence_index",
        "dataset_row_index",
        "dataset_key",
        "action_type",
        "window_index",
        "sequence_window_index",
        "start_frame",
        "end_frame",
        "hand_side",
        "primary_target_region",
        "diagnosis",
        "coarse_topk_hand_target_dist",
        "refined_topk_hand_target_dist",
        "gt_topk_hand_target_dist",
        "refined_topk_gap_to_gt",
        "refined_transl_error",
        "refined_local_hand_error",
        "topk_dist_improvement_coarse_to_refined",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            flat = {key: item.get(key, "") for key in fields}
            for key, value in item.get("metrics", {}).items():
                if key in fields:
                    flat[key] = value
            writer.writerow(flat)


def _summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    overall = summary.get("overall", {})
    keys = [
        "num_windows",
        "refined_topk_hand_target_dist",
        "gt_topk_hand_target_dist",
        "refined_topk_gap_to_gt",
        "topk_dist_improvement_coarse_to_refined",
        "refined_transl_error",
        "refined_local_hand_error",
        "diagnosis_ratio_already_good",
        "diagnosis_ratio_hand_pose_issue",
        "diagnosis_ratio_transl_issue",
        "diagnosis_ratio_mixed_issue",
        "diagnosis_ratio_metric_or_region_issue",
    ]
    return [{"metric": key, "value": overall.get(key, 0)} for key in keys]


def _group_rows(group: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in group.items():
        rows.append(
            {
                "name": name,
                "num_windows": metrics.get("num_windows", 0),
                "refined_gap": metrics.get("refined_topk_gap_to_gt", 0),
                "transl_error": metrics.get("refined_transl_error", 0),
                "local_hand_error": metrics.get("refined_local_hand_error", 0),
                "hand_issue_ratio": metrics.get("diagnosis_ratio_hand_pose_issue", 0),
                "transl_issue_ratio": metrics.get("diagnosis_ratio_transl_issue", 0),
                "mixed_ratio": metrics.get("diagnosis_ratio_mixed_issue", 0),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["hand_issue_ratio"]), -float(row["refined_gap"]), str(row["name"])))


def _write_md(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    summary = payload["summary"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# refine_v2 Refiner Vis Pack Diagnosis\n\n")
        f.write("This report diagnoses whether residual contact gaps are more consistent with translation/global placement errors or local hand/arm pose errors.\n\n")
        f.write("## Inputs\n\n")
        f.write(f"- vis_pack_path: `{payload['paths']['vis_pack_path']}`\n")
        f.write(f"- region_map_path: `{payload['paths']['region_map_path']}`\n")
        f.write(f"- device: `{payload['params']['device']}`\n\n")
        f.write("## Thresholds\n\n")
        for key, value in payload["thresholds"].items():
            f.write(f"- `{key}`: `{value}`\n")
        f.write("\n## Overall\n\n")
        f.write(markdown_table(_summary_rows(summary), ["metric", "value"]))
        f.write("\n\n## By Action Type\n\n")
        f.write(markdown_table(_group_rows(summary.get("by_action_type", {})), ["name", "num_windows", "refined_gap", "transl_error", "local_hand_error", "hand_issue_ratio", "transl_issue_ratio", "mixed_ratio"]))
        f.write("\n\n## By Primary Target Region\n\n")
        f.write(markdown_table(_group_rows(summary.get("by_primary_target_region", {})), ["name", "num_windows", "refined_gap", "transl_error", "local_hand_error", "hand_issue_ratio", "transl_issue_ratio", "mixed_ratio"]))
        f.write("\n\n## Interpretation\n\n")
        f.write("- `transl_issue`: refined translation remains far from GT while local hand pose is relatively close.\n")
        f.write("- `hand_pose_issue`: refined translation is not the main error, but root-local hand pose remains far from GT.\n")
        f.write("- `mixed_issue`: both translation and local hand pose are high-error.\n")
        f.write("- `already_good`: refined hand-target gap is close enough to GT under the configured threshold.\n")
        f.write("- `metric_or_region_issue`: neither translation nor local hand pose is high-error, but the hand-target gap remains high.\n")


@torch.no_grad()
def diagnose_refiner_vis_pack(
    *,
    vis_pack_path: str,
    region_map_path: str,
    output_json: str = "",
    output_md: str = "",
    output_csv: str = "",
    device: str = "cuda",
    batch_size_sequences: int = 4,
    transl_error_high: float = 0.05,
    local_hand_error_high: float = 0.05,
    contact_gap_high: float = 0.03,
) -> dict[str, Any]:
    pack, manifest = _load_pack(vis_pack_path)
    sequences = [dict(item) for item in manifest.get("sequences", [])]
    if not sequences:
        raise ValueError("vis pack manifest has no sequences.")
    region_map = load_region_map(region_map_path or None)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    body_forward = RestoredBodyModelForward(device=dev)
    lengths = np.asarray(pack["lengths"], dtype=np.int64)
    thresholds = {
        "transl_error_high": float(transl_error_high),
        "local_hand_error_high": float(local_hand_error_high),
        "contact_gap_high": float(contact_gap_high),
    }

    items: list[dict[str, Any]] = []
    total_sequences = len(sequences)
    batch_size_sequences = max(1, int(batch_size_sequences))
    for batch_start in range(0, total_sequences, batch_size_sequences):
        batch_end = min(total_sequences, batch_start + batch_size_sequences)
        body_model_type = _body_model_type_values(pack, batch_start, batch_end)
        actor_vertices = _motion_vertices(
            body_forward,
            np.asarray(pack["actor_motion"]),
            start=batch_start,
            end=batch_end,
            lengths=lengths,
            betas=np.asarray(pack["actor_betas"], dtype=np.float32),
            gender_id=np.asarray(pack["actor_gender_id"], dtype=np.int64),
            body_model_type=body_model_type,
            device=dev,
        )
        actor_region_centroids = _region_centroids(actor_vertices, region_map)
        local_vertices: dict[str, torch.Tensor] = {}
        for name, field in MOTION_FIELDS.items():
            vertices = _motion_vertices(
                body_forward,
                np.asarray(pack[field]),
                start=batch_start,
                end=batch_end,
                lengths=lengths,
                betas=np.asarray(pack["reactor_betas"], dtype=np.float32),
                gender_id=np.asarray(pack["reactor_gender_id"], dtype=np.int64),
                body_model_type=body_model_type,
                device=dev,
            )
            # Store selected-hand centroids on demand per side to avoid keeping full vertices per window.
            local_vertices[name] = vertices

        hand_centroid_cache: dict[tuple[str, str], torch.Tensor] = {}

        def hand_centroids(name: str, side: str) -> torch.Tensor:
            key = (name, side)
            if key not in hand_centroid_cache:
                hand_centroid_cache[key] = _hand_centroid(local_vertices[name], region_map, side)
            return hand_centroid_cache[key]

        for seq_idx in range(batch_start, batch_end):
            sequence = dict(sequences[seq_idx])
            local_b = seq_idx - batch_start
            for local_window_idx, window in enumerate(sequence.get("windows", [])):
                window = dict(window)
                hand_side = str(window.get("hand_side", ""))
                if hand_side not in HAND_SIDE_NAMES:
                    hand_side = HAND_SIDE_NAMES[int(window.get("hand_side_id", 0))]
                selected_vertices = {
                    name: hand_centroids(name, hand_side)
                    for name in MOTION_FIELDS
                }
                item = _diagnose_window(
                    sequence=sequence,
                    sequence_index=seq_idx,
                    local_index=local_window_idx,
                    local_vertices=selected_vertices,
                    actor_region_centroids=actor_region_centroids,
                    pack=pack,
                    local_batch_index=local_b,
                    window=window,
                    length=int(lengths[seq_idx]),
                    thresholds=thresholds,
                )
                items.append(item)

    summary = _summarize(items)
    payload = {
        "artifact": "refine_v2_refiner_vis_pack_diagnosis",
        "paths": {
            "vis_pack_path": vis_pack_path,
            "region_map_path": region_map_path,
        },
        "params": {
            "device": str(dev),
            "batch_size_sequences": int(batch_size_sequences),
        },
        "thresholds": thresholds,
        "region_map_summary": region_map_summary(region_map),
        "counts": {
            "num_sequences": int(total_sequences),
            "num_windows": int(len(items)),
        },
        "summary": summary,
        "windows": items,
    }
    if output_json:
        write_json(output_json, payload)
    if output_md:
        _write_md(output_md, payload)
    if output_csv:
        _write_csv(output_csv, items)
    return to_jsonable(payload)


def build_parser():
    parser = argparse.ArgumentParser(description="Diagnose transl vs hand-pose limits from a refine_v2 refiner vis pack.")
    parser.add_argument("--vis_pack_path", required=True)
    parser.add_argument("--region_map_path", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--output_md", default="")
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size_sequences", type=int, default=4)
    parser.add_argument("--transl_error_high", type=float, default=0.05)
    parser.add_argument("--local_hand_error_high", type=float, default=0.05)
    parser.add_argument("--contact_gap_high", type=float, default=0.03)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = diagnose_refiner_vis_pack(
        vis_pack_path=args.vis_pack_path,
        region_map_path=args.region_map_path,
        output_json=args.output_json,
        output_md=args.output_md,
        output_csv=args.output_csv,
        device=args.device,
        batch_size_sequences=args.batch_size_sequences,
        transl_error_high=args.transl_error_high,
        local_hand_error_high=args.local_hand_error_high,
        contact_gap_high=args.contact_gap_high,
    )
    overall = payload["summary"]["overall"]
    print("refine_v2 vis-pack diagnosis")
    for key in (
        "num_windows",
        "refined_topk_gap_to_gt",
        "topk_dist_improvement_coarse_to_refined",
        "refined_transl_error",
        "refined_local_hand_error",
        "diagnosis_ratio_already_good",
        "diagnosis_ratio_hand_pose_issue",
        "diagnosis_ratio_transl_issue",
        "diagnosis_ratio_mixed_issue",
        "diagnosis_ratio_metric_or_region_issue",
    ):
        print(f"{key}: {overall.get(key, 0)}")
    if args.output_json:
        print(f"saved json: {args.output_json}")
    if args.output_md:
        print(f"saved md: {args.output_md}")
    if args.output_csv:
        print(f"saved csv: {args.output_csv}")


if __name__ == "__main__":
    main()
