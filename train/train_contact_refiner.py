import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

from data.refine_dataset import RefineCacheDataset, refine_collate
from model.contact.refiner_model import HandContactRefiner
from model.contact.proposal_model import HandContactProposal
from model.contact.contact_defs import default_refiner_joint_ids
from train.contact_refiner_training_loop import ContactRefinerTrainLoop
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform
from utils.fixseed import fixseed
from utils.parser_util import contact_refiner_train_args


def _build_proposal_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    hand_dim = int(cfg.get("hand_dim", 31))
    part_dim = int(cfg.get("part_dim", 13))
    relation_dim = int(cfg.get("relation_dim", 8))
    hidden_dim = int(cfg.get("hidden_dim", 64))
    num_temporal_blocks = int(cfg.get("num_temporal_blocks", 2))
    dropout = float(cfg.get("dropout", 0.1))
    model = HandContactProposal(
        hand_dim=hand_dim,
        part_dim=part_dim,
        relation_dim=relation_dim,
        hidden_dim=hidden_dim,
        num_temporal_blocks=num_temporal_blocks,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()
    return model


def main():
    args = contact_refiner_train_args()
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

    joint_ids = default_refiner_joint_ids(include_buffer=args.include_buffer)
    model = HandContactRefiner(
        joint_ids=joint_ids,
        hidden_dim=args.hidden_dim,
        num_temporal_blocks=args.num_temporal_blocks,
        num_cross_blocks=args.num_cross_blocks,
        num_spatial_blocks=args.num_spatial_blocks,
        dropout=args.dropout,
        delta_max=args.delta_max,
        use_spatial_attn=args.num_spatial_blocks > 0,
    )
    model.config = {
        "stage2": "hcr_contact_refiner",
        "joint_ids": joint_ids,
        "hidden_dim": args.hidden_dim,
        "num_temporal_blocks": args.num_temporal_blocks,
        "num_cross_blocks": args.num_cross_blocks,
        "num_spatial_blocks": args.num_spatial_blocks,
        "dropout": args.dropout,
        "delta_max": args.delta_max,
    }

    proposal_model = None
    if args.proposal_ckpt and not args.proposal_checkpoint:
        args.proposal_checkpoint = args.proposal_ckpt
    requires_pred = False
    if args.window_source_debug in ("predict", "mix"):
        requires_pred = True
    if args.window_source_debug == "":
        requires_pred = (args.mix_stage_ratio > 0.0) or (args.predict_stage_ratio > 0.0)
    if requires_pred:
        if not args.proposal_checkpoint:
            raise ValueError("--proposal_checkpoint/--proposal_ckpt is required for predicted windows")
        proposal_model = _build_proposal_model(args.proposal_checkpoint, args.device_str)

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

    loop = ContactRefinerTrainLoop(args, model, loader, train_platform, proposal_model=proposal_model)
    loop.run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
