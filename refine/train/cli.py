"""CLI for Stage2-lite training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from refine.model.features import FeatureBuilderConfig
from refine.model.losses import JointRefinementLossConfig
from refine.model.network import JointLocalRefinerConfig
from refine.model.windows import WindowConfig


@dataclass(frozen=True)
class Stage2LiteTrainConfig:
    reaction_data_path: str
    save_dir: str
    batch_size: int
    num_workers: int
    shuffle: bool
    drop_last: bool
    body_model: str
    pose_rep: str
    window_strict_score_threshold: float
    window_near_score_threshold_pre: float
    window_near_score_threshold_post: float
    window_raw_L_min: int
    window_raw_L_max: int
    window_model_W: int
    window_gap_merge: int
    window_pre_max: int
    window_post_max: int
    window_per_hand_max_windows: int
    window_per_seq_max_windows: int
    window_target_smooth_k: int
    hidden_dim: int
    num_heads: int
    num_blocks: int
    dropout: float
    mlp_ratio: float
    delta_scale: float
    residual_loss_type: str
    lambda_res: float
    lambda_smooth: float
    lambda_contact: float
    lambda_identity: float
    core_weight: float
    support_weight: float
    identity_core_weight: float
    identity_support_weight: float
    lr: float
    weight_decay: float
    grad_clip: float
    num_steps: int
    max_epochs: int
    log_interval: int
    save_interval: int
    device: str
    seed: int
    resume_checkpoint: str
    mixed_precision: bool


def build_parser() -> argparse.ArgumentParser:
    window_defaults = WindowConfig()
    network_defaults = JointLocalRefinerConfig()
    loss_defaults = JointRefinementLossConfig()
    feature_defaults = FeatureBuilderConfig()

    parser = argparse.ArgumentParser(description="Train Stage2-lite local refiner.")

    parser.add_argument("--reaction_data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_last", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--window_strict_score_threshold", type=float, default=window_defaults.strict_score_threshold)
    parser.add_argument("--window_near_score_threshold_pre", type=float, default=window_defaults.near_score_threshold_pre)
    parser.add_argument("--window_near_score_threshold_post", type=float, default=window_defaults.near_score_threshold_post)
    parser.add_argument("--window_raw_L_min", type=int, default=window_defaults.raw_L_min)
    parser.add_argument("--window_raw_L_max", type=int, default=window_defaults.raw_L_max)
    parser.add_argument("--window_model_W", type=int, default=feature_defaults.model_window_size)
    parser.add_argument("--window_gap_merge", type=int, default=window_defaults.gap_merge)
    parser.add_argument("--window_pre_max", type=int, default=window_defaults.pre_max)
    parser.add_argument("--window_post_max", type=int, default=window_defaults.post_max)
    parser.add_argument("--window_per_hand_max_windows", type=int, default=window_defaults.per_hand_max_windows)
    parser.add_argument("--window_per_seq_max_windows", type=int, default=window_defaults.per_seq_max_windows)
    parser.add_argument("--window_target_smooth_k", type=int, default=window_defaults.target_smooth_k)

    parser.add_argument("--body_model", type=str, default="smplx")
    parser.add_argument("--pose_rep", type=str, default="rot6d")
    parser.add_argument("--hidden_dim", type=int, default=network_defaults.hidden_dim)
    parser.add_argument("--num_heads", type=int, default=network_defaults.num_heads)
    parser.add_argument("--num_blocks", type=int, default=network_defaults.num_blocks)
    parser.add_argument("--dropout", type=float, default=network_defaults.dropout)
    parser.add_argument("--mlp_ratio", type=float, default=network_defaults.mlp_ratio)
    parser.add_argument("--delta_scale", type=float, default=network_defaults.delta_scale)

    parser.add_argument("--residual_loss_type", type=str, default=loss_defaults.residual_loss_type)
    parser.add_argument("--lambda_res", type=float, default=loss_defaults.lambda_res)
    parser.add_argument("--lambda_smooth", type=float, default=loss_defaults.lambda_smooth)
    parser.add_argument("--lambda_contact", type=float, default=loss_defaults.lambda_contact)
    parser.add_argument("--lambda_identity", type=float, default=loss_defaults.lambda_identity)
    parser.add_argument("--core_weight", type=float, default=loss_defaults.core_weight)
    parser.add_argument("--support_weight", type=float, default=loss_defaults.support_weight)
    parser.add_argument("--identity_core_weight", type=float, default=loss_defaults.identity_core_weight)
    parser.add_argument("--identity_support_weight", type=float, default=loss_defaults.identity_support_weight)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=False)
    return parser


def build_train_config(args: argparse.Namespace) -> Stage2LiteTrainConfig:
    return Stage2LiteTrainConfig(
        reaction_data_path=args.reaction_data_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        drop_last=args.drop_last,
        body_model=args.body_model,
        pose_rep=args.pose_rep,
        window_strict_score_threshold=args.window_strict_score_threshold,
        window_near_score_threshold_pre=args.window_near_score_threshold_pre,
        window_near_score_threshold_post=args.window_near_score_threshold_post,
        window_raw_L_min=args.window_raw_L_min,
        window_raw_L_max=args.window_raw_L_max,
        window_model_W=args.window_model_W,
        window_gap_merge=args.window_gap_merge,
        window_pre_max=args.window_pre_max,
        window_post_max=args.window_post_max,
        window_per_hand_max_windows=args.window_per_hand_max_windows,
        window_per_seq_max_windows=args.window_per_seq_max_windows,
        window_target_smooth_k=args.window_target_smooth_k,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        delta_scale=args.delta_scale,
        residual_loss_type=args.residual_loss_type,
        lambda_res=args.lambda_res,
        lambda_smooth=args.lambda_smooth,
        lambda_contact=args.lambda_contact,
        lambda_identity=args.lambda_identity,
        core_weight=args.core_weight,
        support_weight=args.support_weight,
        identity_core_weight=args.identity_core_weight,
        identity_support_weight=args.identity_support_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_steps=args.num_steps,
        max_epochs=args.max_epochs,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=args.device,
        seed=args.seed,
        resume_checkpoint=args.resume_checkpoint,
        mixed_precision=args.mixed_precision,
    )


def parse_args(argv: list[str] | None = None) -> Stage2LiteTrainConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    return build_train_config(args)


def main(argv: list[str] | None = None):
    config = parse_args(argv)
    from refine.train.loop import Stage2LiteTrainer

    trainer = Stage2LiteTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
