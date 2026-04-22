"""CLI for window-level evaluation of a trained refine_v2 refiner."""

from __future__ import annotations

import argparse
import json
import os


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate refine_v2 window residual refiner.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reaction_data_path", required=True)
    parser.add_argument("--contact_labels_path", required=True)
    parser.add_argument("--subset_manifest_path", required=True)
    parser.add_argument("--selector_windows_path", required=True)
    parser.add_argument("--include_buckets", nargs="+", default=["GT+ / Pred+"])
    parser.add_argument("--selected_action_types", nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--output_json", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    import torch
    from torch.utils.data import DataLoader

    from refine_v2.data.schema import to_jsonable
    from refine_v2.model.losses_v2 import RefineV2Loss, RefineV2LossConfig
    from refine_v2.model.refiner_v2 import RefineV2WindowRefiner, RefineV2WindowRefinerConfig
    from refine_v2.refiner_data.window_dataset import RefineV2WindowDataset
    from refine_v2.refiner_data.window_loader import collate_refine_v2_window_batch
    from refine_v2.train.eval_window import evaluate_model

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    dataset = RefineV2WindowDataset(
        args.reaction_data_path,
        args.contact_labels_path,
        args.subset_manifest_path,
        args.selector_windows_path,
        include_buckets=args.include_buckets,
        selected_action_types=args.selected_action_types,
        strict_checks=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_refine_v2_window_batch,
    )
    model_cfg = RefineV2WindowRefinerConfig(**state["model_config"])
    model = RefineV2WindowRefiner(model_cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    train_cfg = state.get("config", {})
    loss_fn = RefineV2Loss(
        RefineV2LossConfig(
            lambda_motion=float(train_cfg.get("lambda_motion", 1.0)),
            lambda_contact=float(train_cfg.get("lambda_contact", 1.0)),
            lambda_smooth=float(train_cfg.get("lambda_smooth", 0.05)),
            lambda_region_dist=0.0,
            lambda_boundary_trans=float(train_cfg.get("lambda_boundary_trans", 0.0)),
            boundary_trans_frames=int(train_cfg.get("boundary_trans_frames", 2)),
            contact_frame_weight=float(train_cfg.get("contact_frame_weight", 2.0)),
            smooth_l1_beta=float(train_cfg.get("smooth_l1_beta", 0.05)),
        )
    ).to(device)
    result = evaluate_model(model, loader, loss_fn, device=device, max_batches=int(args.max_batches))
    print("refine_v2 eval")
    print(json.dumps(to_jsonable(result["metrics"]), indent=2, sort_keys=True))
    if args.output_json:
        out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(result), f, indent=2, sort_keys=True)
        print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
