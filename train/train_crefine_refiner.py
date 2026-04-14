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
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform
from utils.fixseed import fixseed


def main():
    args = _parse_args()
    fixseed(args.seed)

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
    )
    model.config = {
        "stage2": "crefine_residual_diffusion_refiner",
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
