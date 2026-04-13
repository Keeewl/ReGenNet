import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

from eval.contact_eval.contact_evaluator import HandContactEvaluator


METHOD_KEYS = {
    "gt": "gt_reactor_motion",
    "coarse": "coarse_reactor_motion",
    "refined": "refined_reactor_motion",
    "baseline": "baseline_reactor_motion",
}


def _load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    if ext == ".npy":
        return np.load(path, allow_pickle=True)
    raise ValueError(f"Unsupported file type: {path}")


def _extract_tensor(data, key=None, name="tensor"):
    if key is not None:
        if isinstance(data, dict):
            if key not in data:
                return None
            return _extract_tensor(data[key], name=key)
        raise ValueError(f"Expected dict-like for key '{key}'")

    if torch.is_tensor(data):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, dict):
        if len(data) == 1:
            return _extract_tensor(next(iter(data.values())), name=name)
        raise ValueError(f"Ambiguous dict for {name}; provide a key")
    raise ValueError(f"Unsupported data type for {name}: {type(data)}")


def _ensure_batch(tensor, name="tensor"):
    if tensor is None:
        return None
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    if tensor.dim() < 3:
        raise ValueError(f"{name} has unexpected shape: {tuple(tensor.shape)}")
    return tensor


def _ensure_lengths(lengths, batch_size, num_frames):
    if lengths is None:
        return torch.full((batch_size,), num_frames, dtype=torch.long)
    if torch.is_tensor(lengths):
        out = lengths
    else:
        out = torch.as_tensor(lengths, dtype=torch.long)
    if out.dim() == 0:
        out = out.view(1)
    elif out.dim() > 1:
        out = out.view(-1)
    return out


def _format_value(val):
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.6f}"
    if isinstance(val, int):
        return str(val)
    return str(val)


def _print_summary(results):
    for name, metrics in results.items():
        print(f"[{name}]")
        for key in [
            "hand_cd",
            "contact_ratio",
            "avg_contact_duration",
            "contact_frequency",
            "num_valid_sequences",
            "num_contact_segments",
        ]:
            if key in metrics:
                print(f"  {key}: {_format_value(metrics[key])}")


def _write_csv(path, results):
    if not path:
        return
    keys = set()
    for metrics in results.values():
        keys.update(metrics.keys())
    keys = ["method"] + sorted(keys)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for method, metrics in results.items():
            row = {"method": method}
            row.update(metrics)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=str, default=None)
    parser.add_argument("--actor", type=str, default=None)
    parser.add_argument("--lengths", type=str, default=None)
    parser.add_argument("--gt", type=str, default=None)
    parser.add_argument("--coarse", type=str, default=None)
    parser.add_argument("--refined", type=str, default=None)
    parser.add_argument("--baseline", type=str, default=None)

    parser.add_argument("--actor-key", type=str, default="actor_motion")
    parser.add_argument("--lengths-key", type=str, default="lengths")
    parser.add_argument("--gt-key", type=str, default=METHOD_KEYS["gt"])
    parser.add_argument("--coarse-key", type=str, default=METHOD_KEYS["coarse"])
    parser.add_argument("--refined-key", type=str, default=METHOD_KEYS["refined"])
    parser.add_argument("--baseline-key", type=str, default=METHOD_KEYS["baseline"])

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pose-rep", type=str, default="rot6d")
    parser.add_argument("--body-model", type=str, default="smplx")
    parser.add_argument("--tau-contact", type=float, default=0.10)
    parser.add_argument("--tau-near", type=float, default=0.18)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--json-out", type=str, default="hand_contact_metrics.json")
    parser.add_argument("--csv-out", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.pack is None and args.actor is None:
        raise ValueError("Provide --pack or --actor with at least one method")

    if args.pack is not None:
        pack = _load_any(args.pack)
        if not isinstance(pack, dict):
            raise ValueError("--pack must be a dict-like file")
        actor_motion = _extract_tensor(pack, args.actor_key, name="actor_motion")
        lengths = _extract_tensor(pack, args.lengths_key, name="lengths")
        gt_motion = _extract_tensor(pack, args.gt_key, name="gt_reactor_motion")
        coarse_motion = _extract_tensor(pack, args.coarse_key, name="coarse_reactor_motion")
        refined_motion = _extract_tensor(pack, args.refined_key, name="refined_reactor_motion")
        baseline_motion = _extract_tensor(pack, args.baseline_key, name="baseline_reactor_motion")
    else:
        actor_motion = _extract_tensor(_load_any(args.actor), name="actor_motion")
        lengths = _extract_tensor(_load_any(args.lengths), name="lengths") if args.lengths else None
        gt_motion = _extract_tensor(_load_any(args.gt), name="gt_reactor_motion") if args.gt else None
        coarse_motion = _extract_tensor(_load_any(args.coarse), name="coarse_reactor_motion") if args.coarse else None
        refined_motion = _extract_tensor(_load_any(args.refined), name="refined_reactor_motion") if args.refined else None
        baseline_motion = _extract_tensor(_load_any(args.baseline), name="baseline_reactor_motion") if args.baseline else None

    if actor_motion is None:
        raise ValueError("actor_motion not found")

    actor_motion = _ensure_batch(actor_motion, name="actor_motion")
    batch_size, _, _, num_frames = actor_motion.shape
    lengths = _ensure_lengths(lengths, batch_size, num_frames)

    gt_motion = _ensure_batch(gt_motion, name="gt_reactor_motion")
    coarse_motion = _ensure_batch(coarse_motion, name="coarse_reactor_motion")
    refined_motion = _ensure_batch(refined_motion, name="refined_reactor_motion")
    baseline_motion = _ensure_batch(baseline_motion, name="baseline_reactor_motion")

    device = torch.device(args.device)
    actor_motion = actor_motion.to(device)
    lengths = lengths.to(device)
    if gt_motion is not None:
        gt_motion = gt_motion.to(device)

    evaluator = HandContactEvaluator(
        body_model=args.body_model,
        pose_rep=args.pose_rep,
        tau_contact=args.tau_contact,
        tau_near=args.tau_near,
        topk=args.topk,
        device=device,
    )

    method_inputs = {
        "gt": gt_motion,
        "coarse": coarse_motion,
        "refined": refined_motion,
        "baseline": baseline_motion,
    }
    if all(motion is None for motion in method_inputs.values()):
        raise ValueError("No method motions provided")

    results = {}
    for name, motion in method_inputs.items():
        if motion is None:
            continue
        motion = motion.to(device)
        gt_ref = gt_motion
        if gt_ref is None and name == "gt":
            gt_ref = motion
        results[name] = evaluator.evaluate(
            actor_motion,
            motion,
            lengths=lengths,
            gt_reactor_motion=gt_ref,
            return_debug=args.debug,
        )

    _print_summary(results)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
    _write_csv(args.csv_out, results)


if __name__ == "__main__":
    main()
