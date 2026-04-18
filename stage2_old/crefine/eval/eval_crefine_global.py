import argparse
import csv
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from eval.a2m.stgcn.accuracy import calculate_accuracy
from eval.a2m.stgcn.diversity import calculate_diversity_multimodality
from eval.a2m.stgcn.evaluate import Evaluation as STGCNEvaluation
from eval.a2m.stgcn.fid import calculate_fid
from stage2_old.common.restored.restored_space import RESTORED_PAIR_SPACE, get_space_definition


INTERX_ACTION_RE = re.compile(r"A(\d+)")

FIELD_INFO = {
    "fid": "GT-referenced FID in STGCN feature space; lower is better.",
    "diversity": "Within-set STGCN feature diversity; higher is more diverse.",
    "multimodality": "Same-label STGCN feature distance; higher indicates conditional variety.",
    "accuracy": "STGCN action recognition accuracy; label source is parsed from dataset_key.",
}


def _load_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    if ext == ".h5":
        import h5py

        with h5py.File(path, "r") as f:
            return {k: f[k][()] for k in f.keys()}
    raise ValueError(f"Unsupported pack format: {path}")


def _load_interx_action_names(data_path):
    candidates = []
    if data_path:
        abs_path = os.path.abspath(data_path)
        dataset_dir = os.path.dirname(os.path.dirname(abs_path))
        candidates.append(os.path.join(dataset_dir, "annots", "action_setting.txt"))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.append(os.path.join(repo_root, "dataset", "interx", "annots", "action_setting.txt"))
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    return []


def _parse_action_label(dataset, dataset_key):
    if isinstance(dataset_key, bytes):
        dataset_key = dataset_key.decode("utf-8")
    dataset_key = str(dataset_key)
    if dataset == "chi3d":
        try:
            return int(dataset_key.split("_")[-1])
        except (IndexError, ValueError):
            return None
    if dataset == "interx":
        match = INTERX_ACTION_RE.search(dataset_key)
        if match:
            return int(match.group(1))
        return None
    raise ValueError(f"Unsupported dataset for STGCN global eval: {dataset}")


def _check_pack(pack):
    if "space_definition" in pack:
        space_definition = get_space_definition(pack["space_definition"]).lower()
        if space_definition != RESTORED_PAIR_SPACE:
            raise ValueError(
                f"Expected pack space_definition='{RESTORED_PAIR_SPACE}', got '{space_definition}'."
            )
    required = ["actor_motion", "gt_reactor_motion", "coarse_reactor_motion", "refined_reactor_motion"]
    missing = [name for name in required if name not in pack]
    if missing:
        raise KeyError(f"Pack is missing required global-eval fields: {', '.join(missing)}")


class PackedMotionDataset(Dataset):
    def __init__(self, actor_motion, reactor_motion, labels):
        self.actor_motion = actor_motion
        self.reactor_motion = reactor_motion
        self.labels = labels

    def __len__(self):
        return int(self.actor_motion.shape[0])

    def __getitem__(self, idx):
        output = torch.cat([self.actor_motion[idx], self.reactor_motion[idx]], dim=1)
        return {
            "output": output,
            "y": self.labels[idx],
        }


def _collate(batch):
    return {
        "output": torch.stack([item["output"] for item in batch], dim=0),
        "y": torch.as_tensor([int(item["y"]) for item in batch], dtype=torch.long),
    }


def _diversity_only(feats):
    if feats.shape[0] <= 1:
        return 0.0
    first = np.random.randint(0, feats.shape[0], 200)
    second = np.random.randint(0, feats.shape[0], 200)
    val = 0.0
    for i, j in zip(first, second):
        val += torch.dist(feats[i], feats[j]).item()
    return val / 200.0


def _write_csv(path, results):
    if not path:
        return
    keys = {"method"}
    for metrics in results.values():
        keys.update(metrics.keys())
    fieldnames = ["method"] + sorted(k for k in keys if k != "method")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in results.items():
            row = {"method": method}
            row.update(metrics)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True, type=str)
    parser.add_argument("--dataset", required=True, choices=["chi3d", "interx"], type=str)
    parser.add_argument("--stgcn-model-path", required=True, type=str)
    parser.add_argument("--body-model", default="smplx", type=str)
    parser.add_argument("--data-path", default="", type=str)
    parser.add_argument("--dataset-key-field", default="dataset_key", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--json-out", default="crefine_global_metrics.json", type=str)
    parser.add_argument("--csv-out", default="", type=str)
    args = parser.parse_args()

    pack = _load_any(args.pack)
    _check_pack(pack)

    actor_motion = torch.from_numpy(np.asarray(pack["actor_motion"])).float()
    gt_motion = torch.from_numpy(np.asarray(pack["gt_reactor_motion"])).float()
    coarse_motion = torch.from_numpy(np.asarray(pack["coarse_reactor_motion"])).float()
    refined_motion = torch.from_numpy(np.asarray(pack["refined_reactor_motion"])).float()
    dataset_keys = pack.get(args.dataset_key_field, None)
    if dataset_keys is None:
        raise KeyError(
            f"Pack {args.pack} is missing {args.dataset_key_field}; global STGCN eval needs dataset_key to derive action labels."
        )

    labels = []
    for key in np.asarray(dataset_keys).reshape(-1):
        label = _parse_action_label(args.dataset, key)
        if label is None:
            raise ValueError(f"Unable to parse action label from dataset_key '{key}' for dataset={args.dataset}.")
        labels.append(label)
    labels = torch.as_tensor(labels, dtype=torch.long)

    num_classes = 8 if args.dataset == "chi3d" else len(_load_interx_action_names(args.data_path))
    if num_classes <= 0:
        raise ValueError("Unable to resolve num_classes for STGCN evaluation.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    parameters = {
        "nfeats": 12,
        "num_classes": num_classes,
        "num_person": 2,
        "model_path": args.stgcn_model_path,
    }
    evaluator = STGCNEvaluation(args.dataset, args.body_model, parameters, device)

    method_motions = {
        "gt": gt_motion,
        "coarse": coarse_motion,
        "refined": refined_motion,
    }
    gt_dataset = PackedMotionDataset(actor_motion, gt_motion, labels)
    gt_loader = DataLoader(gt_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
    gt_feats, gt_labels = evaluator.compute_features(None, gt_loader)
    gt_stats = evaluator.calculate_activation_statistics(gt_feats)

    results = {}
    for name, reactor_motion in method_motions.items():
        dataset = PackedMotionDataset(actor_motion, reactor_motion, labels)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)
        accuracy, _ = calculate_accuracy(None, loader, num_classes, evaluator.model, device)
        feats, feat_labels = evaluator.compute_features(None, loader)
        stats = evaluator.calculate_activation_statistics(feats)
        diversity, multimodality = calculate_diversity_multimodality(feats, feat_labels, num_classes, seed=10)
        results[name] = {
            "fid": float(calculate_fid(gt_stats, stats)),
            "diversity": float(diversity if np.isfinite(diversity) else _diversity_only(feats)),
            "multimodality": float(multimodality),
            "accuracy": float(accuracy),
        }

    payload = {
        "results": results,
        "field_info": FIELD_INFO,
        "dataset": args.dataset,
        "num_classes": num_classes,
    }
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
    _write_csv(args.csv_out, results)

    for name, metrics in results.items():
        print(f"[{name}] fid={metrics['fid']:.6f} diversity={metrics['diversity']:.6f} multimodality={metrics['multimodality']:.6f} accuracy={metrics['accuracy']:.6f}")


if __name__ == "__main__":
    main()
