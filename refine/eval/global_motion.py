"""Global Stage2-lite motion evaluation in stage1-aligned processed space.

Protocol split:

- local_contact.py evaluates restored_pair_space and is the main Stage2 metric.
- this module first inverse-restores the infer pack back to
  stage1_aligned_processed_space, then runs STGCN-style global metrics.

The global protocol is an auxiliary check that the local refiner did not break
overall action distribution. It should not be mixed numerically with local
restored-space metrics. Inverse restore and STGCN feature extraction are both
batch-wise to match the Stage1 evaluation style and avoid materializing the
entire restored pack on GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from typing import Any

import h5py
import numpy as np
import torch

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


def _slice_array(value, start: int, end: int):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return np.repeat(arr.reshape(1), end - start, axis=0)
    return arr[start:end]


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


def _convert_batch_to_stage1_aligned_space(
    pack: dict[str, Any],
    *,
    reactor_field: str,
    start: int,
    end: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    common_shift = _to_tensor(_slice_array(pack["loader_base_trans"], start, end), device) + _to_tensor(
        _slice_array(pack["pair_base_trans"], start, end), device
    )
    actor_y = _to_tensor(_slice_array(pack["ground_offset_y_actor"], start, end), device)
    reactor_y = _to_tensor(_slice_array(pack["ground_offset_y_reactor"], start, end), device)
    actor = _inverse_restore_motion(_to_tensor(pack["actor_motion"][start:end], device), common_shift, actor_y)
    return {
        "actor_motion": actor,
        "reactor_motion": _inverse_restore_motion(_to_tensor(pack[reactor_field][start:end], device), common_shift, reactor_y),
        "lengths": torch.as_tensor(pack["lengths"][start:end], device=device, dtype=torch.long),
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


def _extract_state_dict(checkpoint) -> dict[str, torch.Tensor]:
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state = checkpoint[key]
                break
    if not isinstance(state, dict):
        raise ValueError("STGCN checkpoint does not contain a state_dict.")
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def _infer_num_classes_from_checkpoint(stgcn_model_path: str) -> int | None:
    checkpoint = torch.load(stgcn_model_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    weight = state_dict.get("fcn.weight")
    if weight is None:
        return None
    return int(weight.shape[0])


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
    checkpoint = torch.load(stgcn_model_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _make_stgcn_batch(
    actor_motion: torch.Tensor,
    reactor_motion: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "output": torch.cat(
            [
                _pad_smplx_nodes_if_needed(actor_motion),
                _pad_smplx_nodes_if_needed(reactor_motion),
            ],
            dim=2,
        ),
        "y": labels,
        "lengths": lengths,
    }


def _compute_features_and_accuracy_for_variant(
    stgcn,
    pack: dict[str, Any],
    *,
    reactor_field: str,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    feats = []
    y_values = []
    correct = 0
    total = 0
    num_sequences = int(labels.numel())
    with torch.no_grad():
        for start in range(0, num_sequences, batch_size):
            end = min(start + batch_size, num_sequences)
            processed = _convert_batch_to_stage1_aligned_space(
                pack,
                reactor_field=reactor_field,
                start=start,
                end=end,
                device=device,
            )
            batch = _make_stgcn_batch(
                processed["actor_motion"],
                processed["reactor_motion"],
                labels[start:end].to(device),
                processed["lengths"],
            )
            out = stgcn(batch)
            pred = out["yhat"].argmax(dim=1)
            correct += int((pred == batch["y"]).sum().item())
            total += int(batch["y"].numel())
            features = out["features"].detach().cpu()
            if features.dim() == 1:
                features = features.unsqueeze(0)
            feats.append(features)
            y_values.append(batch["y"].detach().cpu())
            del processed, batch, out
    return torch.cat(feats, dim=0), torch.cat(y_values, dim=0), float(correct / max(total, 1))


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
    num_classes: int | None = None,
) -> dict[str, Any]:
    pack = _load_pack(pack_or_path) if isinstance(pack_or_path, str) else pack_or_path
    _check_global_pack(pack)

    dev = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    labels = torch.as_tensor([_parse_action_label(x) for x in np.asarray(pack["dataset_key"]).reshape(-1)], dtype=torch.long)
    inferred_num_classes = _infer_num_classes_from_checkpoint(stgcn_model_path)
    if num_classes is None or num_classes <= 0:
        num_classes = inferred_num_classes or int(labels.max().item()) + 1
    num_classes = int(num_classes)
    if int(labels.max().item()) >= num_classes:
        raise ValueError(
            f"Parsed action label {int(labels.max().item())} exceeds num_classes={num_classes}. "
            "Check dataset_key parsing or pass --num_classes."
        )

    variants = {
        "gt": "reactor_gt",
        "coarse": "reactor_coarse",
        "refined": "reactor_refined",
    }
    in_channels = int(np.asarray(pack["actor_motion"]).shape[2] * 2)
    batch_size = max(int(batch_size), 1)
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
    for name, reactor_field in variants.items():
        feats, y, acc = _compute_features_and_accuracy_for_variant(
            stgcn,
            pack,
            reactor_field=reactor_field,
            labels=labels,
            batch_size=batch_size,
            device=dev,
        )
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
            "num_classes": int(num_classes),
            "num_classes_inferred_from_checkpoint": int(inferred_num_classes) if inferred_num_classes else None,
            "batch_size": batch_size,
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
    parser.add_argument("--num_classes", default=0, type=int)
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
        num_classes=args.num_classes,
    )
    if args.json_out:
        _write_json(args.json_out, payload)
    if args.csv_out:
        _write_global_csv(args.csv_out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
