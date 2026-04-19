"""CLI for Stage2-lite inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Stage2LiteInferConfig:
    reaction_data_path: str
    checkpoint_path: str
    output_dir: str
    device: str
    batch_size: int
    num_workers: int
    sample_mode: str
    num_samples: int
    seed: int
    subset_indices_path: str
    per_action: int
    output_name: str
    save_manifest: bool
    save_coverage_report: bool
    save_debug_stats: bool
    body_model: str
    pose_rep: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage2-lite local refinement inference.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=0, type=int)

    parser.add_argument("--sample_mode", default="fixed", choices=["fixed", "random", "stratified"], type=str)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--subset_indices_path", default="", type=str)
    parser.add_argument("--per_action", default=0, type=int)

    parser.add_argument("--output_name", default="refined_pack.npz", type=str)
    parser.add_argument("--save_manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_coverage_report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_debug_stats", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    return parser


def parse_args(argv: list[str] | None = None) -> Stage2LiteInferConfig:
    args = build_parser().parse_args(argv)
    return Stage2LiteInferConfig(
        reaction_data_path=args.reaction_data_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_mode=args.sample_mode,
        num_samples=args.num_samples,
        seed=args.seed,
        subset_indices_path=args.subset_indices_path,
        per_action=args.per_action,
        output_name=args.output_name,
        save_manifest=args.save_manifest,
        save_coverage_report=args.save_coverage_report,
        save_debug_stats=args.save_debug_stats,
        body_model=args.body_model,
        pose_rep=args.pose_rep,
    )


def main(argv: list[str] | None = None):
    from refine.infer.runner import Stage2LiteInferRunner

    runner = Stage2LiteInferRunner(parse_args(argv))
    runner.run()


if __name__ == "__main__":
    main()
