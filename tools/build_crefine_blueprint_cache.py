import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.refine_dataset import RefineCacheDataset, refine_collate
from model.crefine.crefine_windows import DiffusionWindowBuilder, logits_to_frame_labels
from model.contact.proposal_features import HandContactFeatureBuilder
from model.contact.proposal_model import HandContactProposal
from model.crefine.restored_space import (
    RESTORED_PAIR_SPACE,
    SUPPORTED_BODY_MODEL_TYPE,
    extract_restoration_metadata,
    validate_restoration_metadata,
)


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


def _save_npz(path, arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", required=True, type=str)
    parser.add_argument("--proposal_checkpoint", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)

    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument("--proposal_density", default="small", choices=["small", "medium"], type=str)
    parser.add_argument("--proposal_softmin_beta", default=30.0, type=float)
    parser.add_argument("--window_size", default=12, type=int)
    parser.add_argument("--window_pad", default=2, type=int)
    parser.add_argument("--window_buffer", default=0, type=int)
    parser.add_argument("--active_threshold", default=0.5, type=float)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--max_batches", default=-1, type=int)
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()
    if str(args.body_model).lower() != SUPPORTED_BODY_MODEL_TYPE:
        raise ValueError(
            f"stage2 blueprint building requires body_model={SUPPORTED_BODY_MODEL_TYPE}, got {args.body_model}."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = RefineCacheDataset(args.cache_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=refine_collate,
        persistent_workers=args.num_workers > 0,
    )

    proposal = _build_proposal_model(args.proposal_checkpoint, device=device)
    feature_builder = HandContactFeatureBuilder(
        body_model=args.body_model,
        pose_rep=args.pose_rep,
        translation=True,
        glob=True,
        density=args.proposal_density,
        softmin_beta=args.proposal_softmin_beta,
        device=device,
    )
    window_builder = DiffusionWindowBuilder(window_size=args.window_size, pad=args.window_pad)

    active_list = []
    target_list = []
    band_list = []
    phase_list = []
    conf_list = []
    strict_windows_list = []
    near_windows_list = []
    lengths_list = []
    indices_list = []
    dataset_key_list = []

    total = 0
    total_batches = len(loader)
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=total_batches, desc="Build blueprint cache")
        for batch_idx, batch in pbar:
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if args.num_samples > 0 and total >= args.num_samples:
                break

            actor = batch["actor_motion"].to(device)
            coarse = batch["coarse_motion"].to(device)
            lengths = batch["lengths"].to(device)
            sample_index = batch["sample_index"].to("cpu")
            restoration_meta = extract_restoration_metadata(batch, device=device)
            validate_restoration_metadata(restoration_meta, context="stage2 blueprint restoration metadata")

            hand_feat, part_feat, rel_feat = feature_builder.build(
                actor,
                coarse,
                lengths=lengths,
                restoration_meta=restoration_meta,
            )
            logits = proposal(hand_feat, part_feat, rel_feat)
            labels = logits_to_frame_labels(logits, active_threshold=args.active_threshold)

            strict_windows, near_windows = window_builder.build_from_labels(labels, lengths=lengths)
            if args.window_buffer > 0:
                strict_windows = window_builder.expand_windows_batch(
                    strict_windows,
                    lengths,
                    buffer_frames=args.window_buffer,
                )
                near_windows = window_builder.expand_windows_batch(
                    near_windows,
                    lengths,
                    buffer_frames=args.window_buffer,
                )

            active_prob = torch.sigmoid(logits["active"]).squeeze(-1)

            keep = actor.shape[0]
            if args.num_samples > 0 and total + keep > args.num_samples:
                keep = args.num_samples - total

            active_list.append(labels["active"][:keep].cpu().numpy())
            target_list.append(labels["target_part"][:keep].cpu().numpy())
            band_list.append(labels["band"][:keep].cpu().numpy())
            phase_list.append(labels["phase"][:keep].cpu().numpy())
            conf_list.append(active_prob[:keep].cpu().numpy())
            lengths_list.append(lengths[:keep].cpu().numpy())
            indices_list.append(sample_index[:keep].numpy())
            if "dataset_key" in batch:
                dataset_key_list.append(np.asarray(batch["dataset_key"][:keep], dtype=object))

            strict_windows_list.extend(strict_windows[:keep])
            near_windows_list.extend(near_windows[:keep])

            total += keep
            pbar.set_postfix(samples=total)

    output = {
        "active": np.concatenate(active_list, axis=0).astype(np.int64),
        "target_part": np.concatenate(target_list, axis=0).astype(np.int64),
        "band": np.concatenate(band_list, axis=0).astype(np.int64),
        "phase": np.concatenate(phase_list, axis=0).astype(np.int64),
        "active_prob": np.concatenate(conf_list, axis=0).astype(np.float32),
        "lengths": np.concatenate(lengths_list, axis=0).astype(np.int64),
        "sample_indices": np.concatenate(indices_list, axis=0).astype(np.int64),
        "strict_windows": np.array(strict_windows_list, dtype=object),
        "near_windows": np.array(near_windows_list, dtype=object),
        "config_json": json.dumps(vars(args)).encode("utf-8"),
        "space_definition": np.asarray(RESTORED_PAIR_SPACE),
        "proposal_mode": np.asarray("fully_restored_shape"),
    }
    if dataset_key_list:
        output["dataset_key"] = np.concatenate(dataset_key_list, axis=0)

    _save_npz(args.output_path, output)
    print(f"Saved blueprint cache to {args.output_path} (samples={output['active'].shape[0]})")


if __name__ == "__main__":
    main()
