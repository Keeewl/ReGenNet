"""Local restored-space contact evaluation for Stage2-lite.

Protocol:

- input space: restored_pair_space
- purpose: main Stage2-Lite contact-oriented evaluation
- compared fields: reactor_gt, reactor_coarse, reactor_refined

This evaluator directly consumes the new infer pack field names and implements a
lightweight joint-based contact proxy. It intentionally does not import old
Stage2 runtime modules or mesh-aware losses. Evaluation is batch-wise so large
packs do not need to be converted to SMPL-X xyz in one GPU pass.
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Any

import h5py
import numpy as np
import torch

from refine.data.restored_space import RESTORED_PAIR_SPACE
from refine.data.schema import normalize_space_definition


LEFT_HAND_IDS = tuple(range(25, 40))
RIGHT_HAND_IDS = tuple(range(40, 55))
REACTOR_HAND_IDS = LEFT_HAND_IDS + RIGHT_HAND_IDS
ACTOR_TARGET_IDS = tuple(range(55))


def _load_pack(path: str) -> dict[str, Any]:
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    if path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            return {key: f[key][()] for key in f.keys()}
    raise ValueError(f"Unsupported pack format: {path}")


def _space_values(value) -> set[str]:
    arr = np.asarray(value).reshape(-1)
    return {
        normalize_space_definition(x.decode("utf-8") if isinstance(x, bytes) else str(x))
        for x in arr
    }


def _check_local_pack(pack: dict[str, Any]):
    required = (
        "actor_motion",
        "reactor_gt",
        "reactor_coarse",
        "reactor_refined",
        "lengths",
        "actor_betas",
        "reactor_betas",
        "actor_gender_id",
        "reactor_gender_id",
        "body_model_type",
        "space_definition",
    )
    missing = [key for key in required if key not in pack]
    if missing:
        raise KeyError(f"local contact eval pack missing fields: {', '.join(missing)}")
    spaces = _space_values(pack["space_definition"])
    if spaces != {RESTORED_PAIR_SPACE}:
        raise ValueError(
            f"local_contact requires space_definition={RESTORED_PAIR_SPACE}, got {sorted(spaces)}"
        )


def _length_mask(lengths: torch.Tensor, num_frames: int) -> torch.Tensor:
    return torch.arange(num_frames, device=lengths.device)[None, :] < lengths[:, None]


def _motions_to_xyz(motion: torch.Tensor, lengths: torch.Tensor, *, body_model: str, pose_rep: str, device: torch.device) -> torch.Tensor:
    if motion.shape[2] == 3:
        return motion[:, :55]
    from model.rotation2xyz import Rotation2xyz_x

    rot2xyz = Rotation2xyz_x(device=str(device), dataset="interx")
    mask = _length_mask(lengths.to(device), motion.shape[-1]).bool()
    xyz = rot2xyz(
        x=motion,
        mask=mask,
        pose_rep=pose_rep,
        glob=True,
        translation=True,
        jointstype=body_model,
        vertstrans=True,
        num_person=1,
        betas=None,
        beta=0,
        glob_rot=None,
    )
    return xyz[:, :55]


def _pairwise_min_dist(actor_xyz: torch.Tensor, reactor_xyz: torch.Tensor) -> torch.Tensor:
    actor_ids = torch.as_tensor(ACTOR_TARGET_IDS, dtype=torch.long, device=actor_xyz.device)
    hand_ids = torch.as_tensor(REACTOR_HAND_IDS, dtype=torch.long, device=reactor_xyz.device)
    actor = actor_xyz.index_select(1, actor_ids).permute(0, 3, 1, 2).contiguous()
    hand = reactor_xyz.index_select(1, hand_ids).permute(0, 3, 1, 2).contiguous()
    return torch.cdist(hand, actor).amin(dim=(-1, -2))


def _segment_durations(contact_mask: torch.Tensor, valid_mask: torch.Tensor) -> list[int]:
    durations = []
    for seq_idx in range(contact_mask.shape[0]):
        valid_len = int(valid_mask[seq_idx].sum().item())
        current = 0
        for frame_idx in range(valid_len):
            if bool(contact_mask[seq_idx, frame_idx].item()):
                current += 1
            elif current > 0:
                durations.append(current)
                current = 0
        if current > 0:
            durations.append(current)
    return durations


def _empty_metric_accumulator() -> dict[str, float]:
    return {
        "hand_cd_sum": 0.0,
        "hand_cd_count": 0.0,
        "contact_count": 0.0,
        "valid_count": 0.0,
        "duration_sum": 0.0,
        "num_contact_segments": 0.0,
        "region_dist_sum": 0.0,
        "penetration_count": 0.0,
        "penetration_depth_sum": 0.0,
        "num_valid_sequences": 0.0,
    }


def _accumulate_variant_stats(
    acc: dict[str, float],
    actor_xyz: torch.Tensor,
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    lengths: torch.Tensor,
    *,
    tau_contact: float,
    penetration_threshold: float,
) -> None:
    valid_mask = _length_mask(lengths, actor_xyz.shape[-1])
    gt_dist = _pairwise_min_dist(actor_xyz, gt_xyz)
    pred_dist = _pairwise_min_dist(actor_xyz, pred_xyz)

    gt_contact = (gt_dist < tau_contact) & valid_mask
    pred_contact = (pred_dist < tau_contact) & valid_mask
    valid = valid_mask.float()
    contact_weight = gt_contact.float()
    durations = _segment_durations(pred_contact, valid_mask)
    penetration_depth = (penetration_threshold - pred_dist).clamp_min(0.0)

    acc["hand_cd_sum"] += float((pred_dist * contact_weight).sum().item())
    acc["hand_cd_count"] += float(contact_weight.sum().item())
    acc["contact_count"] += float(pred_contact.float().sum().item())
    acc["valid_count"] += float(valid.sum().item())
    acc["duration_sum"] += float(sum(durations))
    acc["num_contact_segments"] += float(len(durations))
    acc["region_dist_sum"] += float((pred_dist * valid).sum().item())
    acc["penetration_count"] += float(((penetration_depth > 0) & valid_mask).float().sum().item())
    acc["penetration_depth_sum"] += float((penetration_depth * valid).sum().item())
    acc["num_valid_sequences"] += float(lengths.numel())


def _finalize_variant_stats(acc: dict[str, float]) -> dict[str, float | int]:
    valid_count = max(acc["valid_count"], 1.0)
    hand_cd_count = max(acc["hand_cd_count"], 1.0)
    num_segments = int(acc["num_contact_segments"])
    num_sequences = int(acc["num_valid_sequences"])
    return {
        "hand_cd": float(acc["hand_cd_sum"] / hand_cd_count),
        "contact_ratio": float(acc["contact_count"] / valid_count),
        "avg_contact_duration": float(acc["duration_sum"] / max(num_segments, 1)),
        "contact_frequency": float(num_segments / max(num_sequences, 1)),
        "region_hand_dist": float(acc["region_dist_sum"] / valid_count),
        "penetration_rate": float(acc["penetration_count"] / valid_count),
        "penetration_depth": float(acc["penetration_depth_sum"] / valid_count),
        "num_valid_sequences": num_sequences,
        "num_contact_segments": num_segments,
    }


def evaluate_local_contact(
    pack_or_path,
    *,
    device: str = "cpu",
    body_model: str = "smplx",
    pose_rep: str = "rot6d",
    tau_contact: float = 0.10,
    penetration_threshold: float = 0.015,
    batch_size: int = 16,
) -> dict[str, Any]:
    pack = _load_pack(pack_or_path) if isinstance(pack_or_path, str) else pack_or_path
    _check_local_pack(pack)

    dev = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    lengths_np = np.asarray(pack["lengths"])
    num_sequences = int(lengths_np.shape[0])
    batch_size = max(int(batch_size), 1)
    accumulators = {
        "gt": _empty_metric_accumulator(),
        "coarse": _empty_metric_accumulator(),
        "refined": _empty_metric_accumulator(),
    }

    with torch.no_grad():
        for start in range(0, num_sequences, batch_size):
            end = min(start + batch_size, num_sequences)
            lengths = torch.as_tensor(lengths_np[start:end], device=dev, dtype=torch.long)
            actor = torch.as_tensor(pack["actor_motion"][start:end], device=dev, dtype=torch.float32)
            gt = torch.as_tensor(pack["reactor_gt"][start:end], device=dev, dtype=torch.float32)

            actor_xyz = _motions_to_xyz(actor, lengths, body_model=body_model, pose_rep=pose_rep, device=dev)
            gt_xyz = _motions_to_xyz(gt, lengths, body_model=body_model, pose_rep=pose_rep, device=dev)

            _accumulate_variant_stats(
                accumulators["gt"],
                actor_xyz,
                gt_xyz,
                gt_xyz,
                lengths,
                tau_contact=tau_contact,
                penetration_threshold=penetration_threshold,
            )

            coarse = torch.as_tensor(pack["reactor_coarse"][start:end], device=dev, dtype=torch.float32)
            coarse_xyz = _motions_to_xyz(coarse, lengths, body_model=body_model, pose_rep=pose_rep, device=dev)
            _accumulate_variant_stats(
                accumulators["coarse"],
                actor_xyz,
                coarse_xyz,
                gt_xyz,
                lengths,
                tau_contact=tau_contact,
                penetration_threshold=penetration_threshold,
            )

            refined = torch.as_tensor(pack["reactor_refined"][start:end], device=dev, dtype=torch.float32)
            refined_xyz = _motions_to_xyz(refined, lengths, body_model=body_model, pose_rep=pose_rep, device=dev)
            _accumulate_variant_stats(
                accumulators["refined"],
                actor_xyz,
                refined_xyz,
                gt_xyz,
                lengths,
                tau_contact=tau_contact,
                penetration_threshold=penetration_threshold,
            )
            del actor, gt, coarse, refined, actor_xyz, gt_xyz, coarse_xyz, refined_xyz, lengths

    results = {
        "gt": _finalize_variant_stats(accumulators["gt"]),
        "coarse": _finalize_variant_stats(accumulators["coarse"]),
        "refined": _finalize_variant_stats(accumulators["refined"]),
        "field_info": {
            "actor_field": "actor_motion",
            "gt_field": "reactor_gt",
            "coarse_field": "reactor_coarse",
            "refined_field": "reactor_refined",
            "num_sequences": num_sequences,
            "batch_size": batch_size,
            "tau_contact": float(tau_contact),
            "penetration_threshold": float(penetration_threshold),
        },
        "space_protocol": RESTORED_PAIR_SPACE,
    }
    return results


def _write_json(path: str, payload: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_local_csv(path: str, payload: dict[str, Any]):
    metrics = [
        "hand_cd",
        "contact_ratio",
        "avg_contact_duration",
        "contact_frequency",
        "region_hand_dist",
        "penetration_rate",
        "penetration_depth",
        "num_valid_sequences",
        "num_contact_segments",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant"] + metrics)
        writer.writeheader()
        for variant in ("gt", "coarse", "refined"):
            row = {"variant": variant}
            row.update(payload[variant])
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage2-lite local restored-space contact metrics.")
    parser.add_argument("--pack", required=True, type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--tau_contact", default=0.10, type=float)
    parser.add_argument("--penetration_threshold", default=0.015, type=float)
    parser.add_argument("--json_out", default="", type=str)
    parser.add_argument("--csv_out", default="", type=str)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    payload = evaluate_local_contact(
        args.pack,
        device=args.device,
        body_model=args.body_model,
        pose_rep=args.pose_rep,
        tau_contact=args.tau_contact,
        penetration_threshold=args.penetration_threshold,
        batch_size=args.batch_size,
    )
    if args.json_out:
        _write_json(args.json_out, payload)
    if args.csv_out:
        _write_local_csv(args.csv_out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
