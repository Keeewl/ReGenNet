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
from model.contact.contact_defs import default_refiner_joint_ids
from model.contact.refiner_inputs import ContactWindowSampler
from model.contact.refiner_model import HandContactRefiner
from model.contact.proposal_model import HandContactProposal


def _load_refiner_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu")
    config = ckpt.get("config", {})
    model = HandContactRefiner(
        joint_ids=config.get("joint_ids"),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_temporal_blocks=int(config.get("num_temporal_blocks", 2)),
        num_cross_blocks=int(config.get("num_cross_blocks", 2)),
        num_spatial_blocks=int(config.get("num_spatial_blocks", 1)),
        dropout=float(config.get("dropout", 0.1)),
        delta_max=float(config.get("delta_max", 0.15)),
        use_spatial_attn=int(config.get("num_spatial_blocks", 1)) > 0,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, config


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


def _mix_windows(teacher_windows, teacher_labels, pred_windows, pred_labels, ratio, mix_mode, device):
    batch = len(teacher_windows)
    if mix_mode == "per_batch":
        use_teacher = torch.rand((), device=device) > float(ratio)
        if use_teacher:
            return teacher_windows, teacher_labels
        return pred_windows, pred_labels

    choices = torch.rand(batch, device=device) > float(ratio)
    windows = []
    labels = {k: teacher_labels[k].clone() for k in teacher_labels.keys()}
    for b in range(batch):
        if choices[b]:
            windows.append(teacher_windows[b])
        else:
            windows.append(pred_windows[b])
            for key in labels.keys():
                labels[key][b] = pred_labels[key][b]
    return windows, labels


def _accumulate_windows(delta_full, window_items, lengths, joint_ids, num_joints):
    batch = int(lengths.shape[0])
    max_len = int(lengths.max().item())
    device = delta_full.device
    delta_sum = torch.zeros(batch, num_joints, delta_full.shape[2], max_len, device=device)
    delta_count = torch.zeros(batch, num_joints, max_len, device=device)

    joint_ids_t = torch.as_tensor(joint_ids, device=device, dtype=torch.long)

    for idx, item in enumerate(window_items):
        b = int(item["batch_index"])
        start = int(item["start"])
        end = int(item["end"])
        if start > end:
            continue
        length = end - start + 1
        delta_slice = delta_full[idx, :, :, :length]
        delta_sum[b, :, :, start : end + 1] += delta_slice
        delta_count[b, joint_ids_t, start : end + 1] += 1.0

    count = delta_count.clamp(min=1.0).unsqueeze(2)
    delta_avg = delta_sum / count
    delta_avg = delta_avg * (delta_count.unsqueeze(2) > 0)
    return delta_avg


def _save_npz(path, arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def _save_h5(path, arrays):
    import h5py

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", required=True, type=str)
    parser.add_argument("--refiner_checkpoint", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)

    parser.add_argument("--window_source", default="predicted", choices=["teacher", "predicted", "mixed"], type=str)
    parser.add_argument("--proposal_checkpoint", default="", type=str)
    parser.add_argument("--active_threshold", default=0.5, type=float)
    parser.add_argument("--pred_window_ratio", default=0.5, type=float)
    parser.add_argument("--mix_mode", default="per_sample", choices=["per_sample", "per_batch"], type=str)

    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument("--window_size", default=12, type=int)
    parser.add_argument("--window_pad", default=2, type=int)
    parser.add_argument("--include_buffer", action="store_true")
    parser.add_argument("--topk", default=3, type=int)
    parser.add_argument("--sigma", default=0.1, type=float)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--max_batches", default=-1, type=int)
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--save_coarse", action="store_true")
    parser.add_argument("--save_config", action="store_true")
    args = parser.parse_args()

    if args.window_source in ("predicted", "mixed") and not args.proposal_checkpoint:
        raise ValueError("proposal_checkpoint is required for predicted windows")

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

    refiner, refiner_config = _load_refiner_checkpoint(args.refiner_checkpoint, device=device)
    joint_ids = refiner_config.get("joint_ids")
    if not joint_ids:
        joint_ids = default_refiner_joint_ids(include_buffer=args.include_buffer)

    proposal_model = None
    if args.window_source in ("predicted", "mixed"):
        proposal_model = _build_proposal_model(args.proposal_checkpoint, device=device)

    sampler = ContactWindowSampler(
        body_model=args.body_model,
        pose_rep=args.pose_rep,
        translation=True,
        glob=True,
        window_size=args.window_size,
        window_pad=args.window_pad,
        include_buffer=args.include_buffer,
        topk=args.topk,
        sigma=args.sigma,
        device=device,
    )

    refined_list = []
    actor_list = []
    gt_list = []
    coarse_list = []
    lengths_list = []
    indices_list = []

    total = 0
    total_batches = len(loader)
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=total_batches, desc="Generate refined motion")
        for batch_idx, batch in pbar:
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if args.num_samples > 0 and total >= args.num_samples:
                break

            actor = batch["actor_motion"].to(device)
            coarse = batch["coarse_motion"].to(device)
            gt = batch["gt_motion"].to(device)
            lengths = batch["lengths"].to(device)
            sample_index = batch["sample_index"].to("cpu")

            if args.window_source == "teacher":
                windows, labels = sampler.build_teacher_windows(actor, gt, lengths=lengths)
            elif args.window_source == "predicted":
                windows, labels = sampler.build_predicted_windows(
                    actor,
                    coarse,
                    lengths=lengths,
                    proposal_model=proposal_model,
                    active_threshold=args.active_threshold,
                )
            else:
                teacher_windows, teacher_labels = sampler.build_teacher_windows(actor, gt, lengths=lengths)
                pred_windows, pred_labels = sampler.build_predicted_windows(
                    actor,
                    coarse,
                    lengths=lengths,
                    proposal_model=proposal_model,
                    active_threshold=args.active_threshold,
                )
                windows, labels = _mix_windows(
                    teacher_windows,
                    teacher_labels,
                    pred_windows,
                    pred_labels,
                    ratio=args.pred_window_ratio,
                    mix_mode=args.mix_mode,
                    device=device,
                )

            window_batch = sampler.build_window_batch(actor, coarse, gt, lengths, windows, labels)
            if window_batch is None:
                refined = coarse
            else:
                delta_local = refiner(
                    window_batch["coarse_local"],
                    window_batch["actor_patch_feat"],
                    window_batch["relation_feat"],
                    window_batch["cond_feat"],
                    time_mask=window_batch["time_mask"],
                    actor_patch_mask=window_batch["actor_patch_mask"],
                )

                joint_ids_t = torch.as_tensor(window_batch["joint_ids"], device=device, dtype=torch.long)
                delta_full = torch.zeros_like(window_batch["coarse_full"])
                delta_full.index_copy_(1, joint_ids_t, delta_local.permute(0, 2, 3, 1))

                delta_avg = _accumulate_windows(
                    delta_full,
                    window_batch["window_items"],
                    lengths=lengths,
                    joint_ids=window_batch["joint_ids"],
                    num_joints=coarse.shape[1],
                )
                refined = coarse + delta_avg

            keep = refined.shape[0]
            if args.num_samples > 0 and total + keep > args.num_samples:
                keep = args.num_samples - total

            refined_list.append(refined[:keep].cpu().numpy())
            lengths_list.append(lengths[:keep].cpu().numpy())
            indices_list.append(sample_index[:keep].numpy())

            actor_list.append(actor[:keep].cpu().numpy())
            if args.save_gt:
                gt_list.append(gt[:keep].cpu().numpy())
            if args.save_coarse:
                coarse_list.append(coarse[:keep].cpu().numpy())

            total += keep
            pbar.set_postfix(samples=total)

    refined_motion = np.concatenate(refined_list, axis=0).astype(np.float32)
    lengths_out = np.concatenate(lengths_list, axis=0).astype(np.int64)
    indices_out = np.concatenate(indices_list, axis=0).astype(np.int64)

    output = {
        "refined_reactor_motion": refined_motion,
        "lengths": lengths_out,
        "sample_indices": indices_out,
    }
    output["actor_motion"] = np.concatenate(actor_list, axis=0).astype(np.float32)
    if args.save_gt:
        output["gt_reactor_motion"] = np.concatenate(gt_list, axis=0).astype(np.float32)
    if args.save_coarse:
        output["coarse_reactor_motion"] = np.concatenate(coarse_list, axis=0).astype(np.float32)
    if args.save_config:
        output["refiner_config_json"] = json.dumps(refiner_config).encode("utf-8")

    if args.output_path.endswith(".h5"):
        _save_h5(args.output_path, output)
    else:
        _save_npz(args.output_path, output)

    print(f"Saved refined motion to {args.output_path} (samples={refined_motion.shape[0]})")


if __name__ == "__main__":
    main()
