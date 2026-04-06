import json
import os

import torch
from torch.utils.data import DataLoader

from data.refine_dataset import RefineCacheDataset, refine_collate
from model.refine.refine_model import RNetV1, RNetV2
from train.refine_training_loop import RefineTrainLoop
from utils.fixseed import fixseed
from utils.parser_util import refine_train_args


def build_model_from_args(args):
    if args.rnet_version == "v2":
        model = RNetV2(
            njoints=56,
            nfeats=6,
            body_model="smplx",
            pose_rep="rot6d",
            top_k=args.top_k,
            window_size=args.window_size,
            vel_threshold=args.vel_threshold,
            geom_sigma=args.geom_sigma,
            selector_sigma=args.selector_sigma,
            selector_alpha=args.selector_alpha,
            selector_beta=args.selector_beta,
            selector_gamma=args.selector_gamma,
            hidden_dim=args.hidden_dim,
            num_temporal_blocks=args.num_temporal_blocks,
            dropout=args.dropout,
        )
        model.config = {
            "version": "v2",
            "top_k": args.top_k,
            "window_size": args.window_size,
            "vel_threshold": args.vel_threshold,
            "geom_sigma": args.geom_sigma,
            "selector_sigma": args.selector_sigma,
            "selector_alpha": args.selector_alpha,
            "selector_beta": args.selector_beta,
            "selector_gamma": args.selector_gamma,
            "hidden_dim": args.hidden_dim,
            "num_temporal_blocks": args.num_temporal_blocks,
            "dropout": args.dropout,
        }
        return model

    model = RNetV1(
        njoints=56,
        nfeats=6,
        body_model="smplx",
        pose_rep="rot6d",
        top_k=args.top_k,
        window_size=args.window_size,
        vel_threshold=args.vel_threshold,
        geom_sigma=args.geom_sigma,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model.config = {
        "version": "v1",
        "top_k": args.top_k,
        "window_size": args.window_size,
        "vel_threshold": args.vel_threshold,
        "geom_sigma": args.geom_sigma,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    return model


def main():
    args = refine_train_args()
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

    dataset = RefineCacheDataset(args.cache_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=refine_collate,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model_from_args(args)
    loop = RefineTrainLoop(args, model, loader)
    loop.run_loop()


if __name__ == "__main__":
    main()
