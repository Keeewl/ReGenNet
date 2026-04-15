import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import argparse
from torch.utils.data import DataLoader

from model.contact.contact_defs import default_refiner_joint_ids
from model.crefine.crefine_inputs import (
    DiffusionRefinerCacheDataset,
    diffusion_refiner_collate,
)
from model.crefine.crefine_model import MeshConditionalDiffusionRefiner
from model.crefine.crefine_training_loop import ContactDiffusionRefinerTrainLoop
from model.crefine.restored_space import SUPPORTED_BODY_MODEL_TYPE
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform
from utils.fixseed import fixseed



def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", required=True, type=str)
    parser.add_argument("--blueprint_cache_path", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_steps", default=50_000, type=int)
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--save_interval", default=2_000, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--resume_checkpoint", default="", type=str)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--max_batches", default=-1, type=int)
    parser.add_argument(
        "--train_platform_type",
        default="NoPlatform",
        choices=["NoPlatform", "ClearmlPlatform", "TensorboardPlatform"],
        type=str,
    )

    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument("--window_size", default=12, type=int)
    parser.add_argument("--window_pad", default=2, type=int)
    parser.add_argument("--include_buffer", action="store_true")
    parser.add_argument("--density", default="medium", choices=["small", "medium"], type=str)
    parser.add_argument("--crefine_version", default="crefine_v3", type=str)
    parser.add_argument("--use_shape_condition", action="store_true")
    parser.add_argument("--shape_dim", default=10, type=int)
    parser.add_argument("--use_restored_shape", action="store_true")
    parser.add_argument("--gender_num_embeddings", default=3, type=int)

    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--num_temporal_blocks", default=2, type=int)
    parser.add_argument("--num_cross_blocks", default=2, type=int)
    parser.add_argument("--num_spatial_blocks", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    parser.add_argument("--diffusion_steps", default=1000, type=int)
    parser.add_argument("--sampling_steps", default=50, type=int)
    parser.add_argument("--noise_schedule", default="cosine", choices=["linear", "cosine"], type=str)

    parser.add_argument("--strict_near_ratio", default=0.7, type=float)
    parser.add_argument("--max_windows_per_batch", default=64, type=int)
    parser.add_argument("--teacher_warmup_steps", default=1000, type=int)
    parser.add_argument("--alignment_only_steps", default=2000, type=int)
    parser.add_argument("--cleanup_ramp_steps", default=3000, type=int)

    parser.add_argument("--softmin_beta", default=30.0, type=float)
    parser.add_argument("--strict_contact_target", default=0.008, type=float)
    parser.add_argument("--near_contact_margin", default=0.03, type=float)
    parser.add_argument("--aux_alpha_min", default=0.05, type=float)
    parser.add_argument("--delta_clip", default=0.5, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--blueprint_conf_min", default=0.3, type=float)
    parser.add_argument("--max_nontarget_vertices", default=256, type=int)

    parser.add_argument("--lambda_contact_strict", default=1.0, type=float)
    parser.add_argument("--lambda_penetration", default=0.5, type=float)
    parser.add_argument("--lambda_contact_near", default=0.1, type=float)
    parser.add_argument("--lambda_identity", default=0.02, type=float)
    parser.add_argument("--lambda_smooth", default=0.01, type=float)
    parser.add_argument("--penetration_margin", default=0.005, type=float)
    parser.add_argument("--nontarget_margin", default=0.02, type=float)
    parser.add_argument("--penalize_target_penetration", action="store_true")
    parser.add_argument("--log_events", action="store_true")
    parser.add_argument("--lambda_contact_normal", default=0.0, type=float)
    parser.add_argument("--lambda_clearance", default=0.0, type=float)

    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    return parser.parse_args()


def main():
    args = _parse_args()
    fixseed(args.seed)
    if str(args.body_model).lower() != SUPPORTED_BODY_MODEL_TYPE:
        raise ValueError(
            f"stage2 crefine training requires body_model={SUPPORTED_BODY_MODEL_TYPE}, got {args.body_model}."
        )
    if args.use_restored_shape and not args.use_shape_condition:
        raise ValueError(
            "use_restored_shape=True requires use_shape_condition=True so restored-shape tokens are active in training."
        )

    if args.cuda and torch.cuda.is_available():
        args.device_str = f"cuda:{args.device}"
    else:
        args.device_str = "cpu"

    if os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError(f"save_dir [{args.save_dir}] already exists.")
    os.makedirs(args.save_dir, exist_ok=True)

    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    dataset = DiffusionRefinerCacheDataset(args.cache_path, args.blueprint_cache_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=diffusion_refiner_collate,
        persistent_workers=args.num_workers > 0,
    )

    joint_ids = default_refiner_joint_ids(include_buffer=args.include_buffer)
    model = MeshConditionalDiffusionRefiner(
        joint_ids=joint_ids,
        hidden_dim=args.hidden_dim,
        num_temporal_blocks=args.num_temporal_blocks,
        num_cross_blocks=args.num_cross_blocks,
        num_spatial_blocks=args.num_spatial_blocks,
        dropout=args.dropout,
        cond_dim=18,
        actor_dim=6,
        mesh_dim=6,
        mesh_rel_dim=15,
        mesh_type_vocab=16,
        time_embed_dim=args.hidden_dim,
        use_spatial_attn=args.num_spatial_blocks > 0,
        shape_dim=args.shape_dim,
        gender_num_embeddings=args.gender_num_embeddings,
        use_shape_condition=args.use_shape_condition,
    )
    model.config = {
        "stage2": "crefine_residual_diffusion_refiner",
        "crefine_version": args.crefine_version,
        "joint_ids": joint_ids,
        "hidden_dim": args.hidden_dim,
        "num_temporal_blocks": args.num_temporal_blocks,
        "num_cross_blocks": args.num_cross_blocks,
        "num_spatial_blocks": args.num_spatial_blocks,
        "dropout": args.dropout,
        "diffusion_steps": args.diffusion_steps,
        "noise_schedule": args.noise_schedule,
        "sampling_steps": args.sampling_steps,
        "window_size": args.window_size,
        "window_pad": args.window_pad,
        "density": args.density,
        "aux_alpha_min": args.aux_alpha_min,
        "delta_clip": args.delta_clip,
        "grad_clip": args.grad_clip,
        "alignment_only_steps": args.alignment_only_steps,
        "cleanup_ramp_steps": args.cleanup_ramp_steps,
        "strict_contact_target": args.strict_contact_target,
        "near_contact_margin": args.near_contact_margin,
        "blueprint_conf_min": args.blueprint_conf_min,
        "penalize_target_penetration": args.penalize_target_penetration,
        "teacher_warmup_steps": args.teacher_warmup_steps,
        "strict_near_ratio": args.strict_near_ratio,
        "use_shape_condition": args.use_shape_condition,
        "shape_dim": args.shape_dim,
        "use_restored_shape": args.use_restored_shape,
        "gender_num_embeddings": args.gender_num_embeddings,
        "lambda_contact_strict": args.lambda_contact_strict,
        "lambda_penetration": args.lambda_penetration,
        "lambda_contact_near": args.lambda_contact_near,
        "lambda_identity": args.lambda_identity,
        "lambda_smooth": args.lambda_smooth,
        "lambda_contact_normal": args.lambda_contact_normal,
        "lambda_clearance": args.lambda_clearance,
        "lr": args.lr,
    }

    train_platform_type = eval(args.train_platform_type)
    if train_platform_type is TensorboardPlatform:
        tb_root = os.path.join(args.save_dir, "tb")
        os.makedirs(tb_root, exist_ok=True)
        existing = []
        for name in os.listdir(tb_root):
            if not name.startswith("run_"):
                continue
            parts = name.split("_", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                existing.append(int(parts[1]))
        next_idx = max(existing) + 1 if existing else 0
        run_name = f"run_{next_idx:03d}"
        tb_dir = os.path.join(tb_root, run_name)
        train_platform = train_platform_type(tb_dir)
    else:
        train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name="Args")

    loop = ContactDiffusionRefinerTrainLoop(args, model, loader, train_platform)
    loop.run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
