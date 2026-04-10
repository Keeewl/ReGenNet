import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader

from data.refine_dataset import RefineCacheDataset, refine_collate
from model.contact.proposal_model import HandContactProposal
from train.contact_proposal_training_loop import ContactProposalTrainLoop
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform
from utils.fixseed import fixseed
from utils.parser_util import contact_proposal_train_args


def main():
    args = contact_proposal_train_args()
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

    hand_dim = 31
    part_dim = 13
    model = HandContactProposal(
        hand_dim=hand_dim,
        part_dim=part_dim,
        relation_dim=8,
        hidden_dim=args.hidden_dim,
        num_temporal_blocks=args.num_temporal_blocks,
        dropout=args.dropout,
    )
    model.config = {
        "stage2": "hcr_contact_proposal",
        "hand_dim": hand_dim,
        "part_dim": part_dim,
        "relation_dim": 8,
        "hidden_dim": args.hidden_dim,
        "num_temporal_blocks": args.num_temporal_blocks,
        "dropout": args.dropout,
        "topk": args.topk,
        "sigma": args.sigma,
        "tau_contact": args.tau_contact,
        "tau_near": args.tau_near,
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

    loop = ContactProposalTrainLoop(args, model, loader, train_platform)
    loop.run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
