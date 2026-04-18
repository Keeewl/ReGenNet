"""Small CLI for debugging deterministic Stage2-lite windows."""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from refine.data.cache_dataset import ReactionDataDataset
from refine.data.collate import reaction_data_collate
from refine.data.restored_space import extract_restoration_metadata
from refine.eval.window_audit import audit_windows
from refine.model.windows import DeterministicWindowSelector, WindowConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reaction_data", required=True, type=str)
    parser.add_argument("--num_samples", default=2, type=int)
    parser.add_argument("--json_out", default="", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument("--body_model", default="smplx", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = ReactionDataDataset(args.reaction_data)
    count = min(len(dataset), max(1, int(args.num_samples)))
    batch = reaction_data_collate([dataset[idx] for idx in range(count)])
    restoration_meta = extract_restoration_metadata(batch)
    selector = DeterministicWindowSelector(
        config=WindowConfig(),
        body_model=args.body_model,
        pose_rep=args.pose_rep,
    )
    result = selector.build_windows_for_batch(
        actor_motion=batch["actor_motion"],
        coarse_motion=batch["coarse_motion"],
        lengths=batch["lengths"],
        restoration_meta=restoration_meta,
    )
    stats = audit_windows(
        window_items=result["window_items"],
        actor_motion=batch["actor_motion"],
        coarse_motion=batch["coarse_motion"],
        lengths=batch["lengths"],
        restoration_meta=restoration_meta,
        gt_motion=batch.get("gt_motion", None),
        selector=selector,
    )
    payload = {
        "num_samples": count,
        "num_windows": len(result["window_items"]),
        "audit": stats,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
