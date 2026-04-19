"""Global Stage2-lite motion evaluation in stage1-aligned processed space.

Protocol split:

- local_contact.py evaluates restored_pair_space and is the main Stage2 metric.
- this module first inverse-restores the infer pack back to
  stage1_aligned_processed_space, then runs STGCN-style global metrics.

The global protocol is an auxiliary check that the local refiner did not break
overall action distribution. It should not be mixed numerically with local
restored-space metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from eval.a2m.recognition.models.stgcn import STGCN
from eval.a2m.stgcn.diversity import calculate_diversity_multimodality
from eval.a2m.stgcn.fid import calculate_fid
from refine.data.restored_space import RESTORED_PAIR_SPACE
from refine.data.schema import normalize_space_definition


STAGE1_ALIGNED_SPACE = "stage1_aligned_processed_space"
_ACTION_RE = re.compile(r"A(\d+)")


def _load_pack(path: str) -> dict[str, Any]:
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    if path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            return {key: f[key][()] for key in f.keys()}
    raise ValueError(f"Unsupported pack format: {path}")


def _check_global_pack(pack: dict[str, Any]):
    required = (
        "actor_motion",
        "reactor_gt",
        "reactor_coarse",
        "reactor_refined",
        "lengths",
        "dataset_key",
        "space_definition",
        "loader_base_trans",
        "pair_base_trans",
        "ground_offset_y_actor",
        "ground_offset_y_reactor",
    )
    missing = [key for key in required if key not in pack]
    if missing:
        raise KeyError(f"global motion eval pack missing fields: {', '.join(missing)}")
    spaces = {
        normalize_space_definition(x.decode("utf-8") if isinstance(x, bytes) else str(x))
        for x in np.asarray(pack["space_definition"]).reshape(-1)
    }
    if spaces != {RESTORED_PAIR_SPACE}:
        raise ValueError(
            f"global_motion expects infer pack in {RESTORED_PAIR_SPACE} before inverse restore, got {sorted(spaces)}"
        )


def _to_tensor(value, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _inverse_restore_motion(
    motion: torch.Tensor,
    common_shift: torch.Tensor,
    y_shift: torch.Tensor,
) -> torch.Tensor:
    out = motion.clone()
    if out.dim() != 4:
        raise ValueError("motion must have shape [B, J, F, T].")
    if out.shape[1] == 0 or out.shape[2] < 3:
        return out
    transl = out[:, -1, :3, :]
    transl = transl - common_shift.view(-1, 3, 1).to(dtype=out.dtype, device=out.device)
    transl[:, 1, :] = transl[:, 1, :] - y_shift.view(-1, 1).to(dtype=out.dtype, device=out.device)
    out[:, -1, :3, :] = transl
    return out


def _convert_pack_to_stage1_aligned_space(pack: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    common_shift = _to_tensor(pack["loader_base_trans"], device) + _to_tensor(pack["pair_base_trans"], device)
    actor_y = _to_tensor(pack["ground_offset_y_actor"], device)
    reactor_y = _to_tensor(pack["ground_offset_y_reactor"], device)
    actor = _inverse_restore_motion(_to_tensor(pack["actor_motion"], device), common_shift, actor_y)
    return {
        "actor_motion": actor,
        "reactor_gt": _inverse_restore_motion(_to_tensor(pack["reactor_gt"], device), common_shift, reactor_y),
        "reactor_coarse": _inverse_restore_motion(_to_tensor(pack["reactor_coarse"], device), common_shift, reactor_y),
        "reactor_refined": _inverse_restore_motion(_to_tensor(pack["reactor_refined"], device), common_shift, reactor_y),
        "lengths": torch.as_tensor(pack["lengths"], device=device, dtype=torch.long),
    }


def _parse_action_label(dataset_key: Any) -> int:
    if isinstance(dataset_key, bytes):
        dataset_key = dataset_key.decode("utf-8")
    match = _ACTION_RE.search(str(dataset_key))
    if not match:
        raise ValueError(f"Unable to parse InterX action label from dataset_key: {dataset_key}")
    return int(match.group(1))


def _pad_smplx_nodes_if_needed(motion: torch.Tensor) -> torch.Tensor:
    if motion.shape[1] == 56:
        return motion
    if motion.shape[1] == 55:
        pad = torch.zeros(
            (motion.shape[0], 1, motion.shape[2], motion.shape[3]),
            dtype=motion.dtype,
            device=motion.device,
        )
        return torch.cat([motion, pad], dim=1)
    return motion


class PackedMotionDataset(Dataset):
    def __init__(
        self,
        actor_motion: torch.Tensor,
        reactor_motion: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
    ):
        self.output = torch.cat(
            [
                _pad_smplx_nodes_if_needed(actor_motion),
                _pad_smplx_nodes_if_needed(reactor_motion),
            ],
            dim=2,
        ).detach().cpu()
        self.labels = labels.detach().cpu().long()
        self.lengths = lengths.detach().cpu().long()

    def __len__(self):
        return int(self.output.shape[0])

    def __getitem__(self, idx):
        return {
            "output": self.output[idx],
            "y": self.labels[idx],
            "lengths": self.lengths[idx],
        }


def _motion_collate(batch):
    return {
        "output": torch.stack([item["output"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "lengths": torch.stack([item["lengths"] for item in batch], dim=0),
    }


def _load_stgcn(
    *,
    dataset: str,
    body_model: str,
    stgcn_model_path: str,
    num_classes: int,
    in_channels: int,
    device: torch.device,
):
    layout = "smplx" if body_model == "smplx" else "smpl"
    model = STGCN(
        in_channels=in_channels,
        num_class=num_classes,
        num_person=2,
        graph_args={"layout": layout, "strategy": "spatial"},
        edge_importance_weighting=True,
        device=device,
    ).to(device)
    state_dict = torch.load(stgcn_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _compute_features_and_accuracy(stgcn, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    feats = []
    labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                "output": batch["output"].to(device),
                "y": batch["y"].to(device),
                "lengths": batch["lengths"].to(device),
            }
            out = stgcn(batch)
            pred = out["yhat"].argmax(dim=1)
            correct += int((pred == batch["y"]).sum().item())
            total += int(batch["y"].numel())
            feats.append(out["features"].detach().cpu())
            labels.append(batch["y"].detach().cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0), float(correct / max(total, 1))


def _activation_stats(feats: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    arr = feats.detach().cpu().numpy()
    return np.mean(arr, axis=0), np.cov(arr, rowvar=False)


def evaluate_global_motion(
    pack_or_path,
    *,
    dataset: str = "interx",
    stgcn_model_path: str,
    body_model: str = "smplx",
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    pack = _load_pack(pack_or_path) if isinstance(pack_or_path, str) else pack_or_path
    _check_global_pack(pack)

    dev = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    processed = _convert_pack_to_stage1_aligned_space(pack, dev)
    labels = torch.as_tensor([_parse_action_label(x) for x in np.asarray(pack["dataset_key"]).reshape(-1)], dtype=torch.long)
    num_classes = int(labels.max().item()) + 1

    variants = {
        "gt": processed["reactor_gt"],
        "coarse": processed["reactor_coarse"],
        "refined": processed["reactor_refined"],
    }
    actor = processed["actor_motion"]
    in_channels = int(actor.shape[2] * 2)
    stgcn = _load_stgcn(
        dataset=dataset,
        body_model=body_model,
        stgcn_model_path=stgcn_model_path,
        num_classes=num_classes,
        in_channels=in_channels,
        device=dev,
    )

    results = {}
    features = {}
    gt_stats = None
    for name, reactor in variants.items():
        loader = DataLoader(
            PackedMotionDataset(actor, reactor, labels.to(dev), processed["lengths"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_motion_collate,
        )
        feats, y, acc = _compute_features_and_accuracy(stgcn, loader, dev)
        stats = _activation_stats(feats)
        if name == "gt":
            gt_stats = stats
        diversity, multimodality = calculate_diversity_multimodality(
            feats,
            y,
            num_classes,
            seed=seed,
            unconstrained=False,
        )
        features[name] = {"stats": stats}
        results[name] = {
            "accuracy": float(acc),
            "diversity": float(diversity),
            "multimodality": float(multimodality),
        }

    for name in variants:
        results[name]["fid"] = float(calculate_fid(gt_stats, features[name]["stats"]))

    return {
        "gt": results["gt"],
        "coarse": results["coarse"],
        "refined": results["refined"],
        "field_info": {
            "actor_field": "actor_motion",
            "gt_field": "reactor_gt",
            "coarse_field": "reactor_coarse",
            "refined_field": "reactor_refined",
            "dataset_key_field": "dataset_key",
            "num_sequences": int(labels.numel()),
            "num_classes_observed": int(num_classes),
            "inverse_restore": "translation joint subtracts loader_base_trans + pair_base_trans and actor/reactor ground offsets",
        },
        "space_protocol": STAGE1_ALIGNED_SPACE,
        "notes": [
            "Global STGCN eval is an auxiliary Stage2 check.",
            "Pack motions are inverse-restored before recognition to match Stage1 processed-space distribution.",
            "Do not mix these scores numerically with restored-space local contact metrics.",
        ],
    }


def _write_json(path: str, payload: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_global_csv(path: str, payload: dict[str, Any]):
    metrics = ["accuracy", "diversity", "multimodality", "fid"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant"] + metrics)
        writer.writeheader()
        for variant in ("gt", "coarse", "refined"):
            row = {"variant": variant}
            row.update(payload[variant])
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Stage2-lite global motion metrics in Stage1 processed space.")
    parser.add_argument("--pack", required=True, type=str)
    parser.add_argument("--dataset", default="interx", type=str)
    parser.add_argument("--stgcn_model_path", required=True, type=str)
    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--dataset_key_field", default="dataset_key", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--json_out", default="", type=str)
    parser.add_argument("--csv_out", default="", type=str)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.dataset_key_field != "dataset_key":
        raise ValueError("Current Stage2-lite pack uses dataset_key as the dataset key field.")
    payload = evaluate_global_motion(
        args.pack,
        dataset=args.dataset,
        stgcn_model_path=args.stgcn_model_path,
        body_model=args.body_model,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    if args.json_out:
        _write_json(args.json_out, payload)
    if args.csv_out:
        _write_global_csv(args.csv_out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
