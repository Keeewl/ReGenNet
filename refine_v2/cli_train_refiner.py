"""CLI for training the first refine_v2 residual refiner."""

from __future__ import annotations

import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Train refine_v2 window residual refiner.")
    parser.add_argument("--reaction_data_path", required=True)
    parser.add_argument("--contact_labels_path", required=True)
    parser.add_argument("--subset_manifest_path", required=True)
    parser.add_argument("--selector_windows_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--include_buckets", nargs="+", default=["GT+ / Pred+"])
    parser.add_argument("--selected_action_types", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=1234)
    parser.add_argument("--overfit_num_windows", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--max_window_size", type=int, default=256)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--lambda_motion", type=float, default=1.0)
    parser.add_argument("--lambda_contact", type=float, default=1.0)
    parser.add_argument("--lambda_smooth", type=float, default=0.05)
    parser.add_argument("--lambda_region_dist", type=float, default=0.0)
    parser.add_argument("--contact_frame_weight", type=float, default=2.0)
    parser.add_argument("--smooth_l1_beta", type=float, default=0.05)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from refine_v2.train.trainer import RefineV2Trainer, RefineV2TrainerConfig

    config = RefineV2TrainerConfig(**vars(args))
    trainer = RefineV2Trainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
