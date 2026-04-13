import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from tqdm import tqdm

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


def _slice_batch(tensor, start, end):
    if tensor is None:
        return None
    return tensor[start:end]


def _init_accumulator():
    return {
        "hand_cd_sum": 0.0,
        "hand_cd_count": 0.0,
        "hand_cd_topk_sum": 0.0,
        "contact_ratio_sum": 0.0,
        "contact_ratio_count": 0,
        "avg_contact_duration_sum": 0.0,
        "avg_contact_duration_count": 0,
        "contact_frequency_sum": 0.0,
        "contact_frequency_count": 0,
        "num_valid_sequences": 0,
        "num_contact_segments": 0,
        "num_contact_frames": 0,
    }


def _accumulate_metrics(acc, metrics):
    num_valid = int(metrics.get("num_valid_sequences", 0))
    num_segments = int(metrics.get("num_contact_segments", 0))
    num_frames = int(metrics.get("num_contact_frames", 0))

    acc["num_valid_sequences"] += num_valid
    acc["num_contact_segments"] += num_segments
    acc["num_contact_frames"] += num_frames

    contact_ratio = metrics.get("contact_ratio", None)
    if contact_ratio is not None and num_valid > 0:
        acc["contact_ratio_sum"] += float(contact_ratio) * num_valid
        acc["contact_ratio_count"] += num_valid

    avg_duration = metrics.get("avg_contact_duration", None)
    if avg_duration is not None and num_segments > 0:
        acc["avg_contact_duration_sum"] += float(avg_duration) * num_segments
        acc["avg_contact_duration_count"] += num_segments

    contact_freq = metrics.get("contact_frequency", None)
    if contact_freq is not None and num_valid > 0:
        acc["contact_frequency_sum"] += float(contact_freq) * num_valid
        acc["contact_frequency_count"] += num_valid

    hand_cd = metrics.get("hand_cd", None)
    hand_cd_count = metrics.get("hand_cd_count", None)
    if hand_cd is not None and hand_cd_count:
        acc["hand_cd_sum"] += float(hand_cd) * float(hand_cd_count)
        acc["hand_cd_count"] += float(hand_cd_count)

    hand_cd_topk = metrics.get("hand_cd_topk_mean", None)
    if hand_cd_topk is not None and hand_cd_count:
        acc["hand_cd_topk_sum"] += float(hand_cd_topk) * float(hand_cd_count)


def _finalize_metrics(acc, include_debug=False):
    hand_cd = None
    if acc["hand_cd_count"] > 0:
        hand_cd = acc["hand_cd_sum"] / acc["hand_cd_count"]

    contact_ratio = 0.0
    if acc["contact_ratio_count"] > 0:
        contact_ratio = acc["contact_ratio_sum"] / acc["contact_ratio_count"]

    avg_duration = 0.0
    if acc["avg_contact_duration_count"] > 0:
        avg_duration = acc["avg_contact_duration_sum"] / acc["avg_contact_duration_count"]

    contact_freq = 0.0
    if acc["contact_frequency_count"] > 0:
        contact_freq = acc["contact_frequency_sum"] / acc["contact_frequency_count"]

    results = {
        "hand_cd": hand_cd,
        "contact_ratio": contact_ratio,
        "avg_contact_duration": avg_duration,
        "contact_frequency": contact_freq,
        "num_valid_sequences": int(acc["num_valid_sequences"]),
        "num_contact_segments": int(acc["num_contact_segments"]),
        "num_contact_frames": int(acc["num_contact_frames"]),
    }

    if include_debug:
        results["hand_cd_count"] = float(acc["hand_cd_count"])
        if acc["hand_cd_count"] > 0:
            results["hand_cd_topk_mean"] = acc["hand_cd_topk_sum"] / acc["hand_cd_count"]
        else:
            results["hand_cd_topk_mean"] = None

    return results


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
    parser.add_argument("--batch-size", type=int, default=8)
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

    num_samples = actor_motion.shape[0]
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    batch_size = min(int(args.batch_size), num_samples)

    for name, motion in method_inputs.items():
        if motion is None:
            continue
        if motion.shape[0] != num_samples:
            raise ValueError(f"{name} has {motion.shape[0]} samples, expected {num_samples}")
    if gt_motion is not None and gt_motion.shape[0] != num_samples:
        raise ValueError(f"gt has {gt_motion.shape[0]} samples, expected {num_samples}")

    results = {}
    for name, motion in method_inputs.items():
        if motion is None:
            continue
        acc = _init_accumulator()
        total_batches = (num_samples + batch_size - 1) // batch_size
        iterator = range(0, num_samples, batch_size)
        for start in tqdm(iterator, total=total_batches, desc=f"Eval {name}"):
            end = min(start + batch_size, num_samples)
            actor_b = _slice_batch(actor_motion, start, end).to(device)
            motion_b = _slice_batch(motion, start, end).to(device)
            lengths_b = _slice_batch(lengths, start, end).to(device)

            gt_ref = None
            if gt_motion is not None:
                gt_ref = _slice_batch(gt_motion, start, end).to(device)
            elif name == "gt":
                gt_ref = motion_b

            with torch.no_grad():
                metrics = evaluator.evaluate(
                    actor_b,
                    motion_b,
                    lengths=lengths_b,
                    gt_reactor_motion=gt_ref,
                    return_debug=True,
                )
            _accumulate_metrics(acc, metrics)
        results[name] = _finalize_metrics(acc, include_debug=args.debug)

    _print_summary(results)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
    _write_csv(args.csv_out, results)


if __name__ == "__main__":
    main()
