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
from model.crefine.crefine_inputs import DiffusionRefinerInputBuilder
from model.crefine.crefine_model import MeshConditionalDiffusionRefiner, create_spaced_diffusion
from model.crefine.restored_space import (
    OPTIONAL_CACHE_FIELDS,
    REQUIRED_CACHE_FIELDS,
    extract_restoration_metadata,
)
from model.crefine.mesh_regions import get_mesh_region_provider
from model.crefine.restored_body_model import RestoredBodyModelForward


def _load_refiner_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = MeshConditionalDiffusionRefiner(
        joint_ids=cfg.get("joint_ids"),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_temporal_blocks=int(cfg.get("num_temporal_blocks", 2)),
        num_cross_blocks=int(cfg.get("num_cross_blocks", 2)),
        num_spatial_blocks=int(cfg.get("num_spatial_blocks", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
        cond_dim=18,
        actor_dim=6,
        mesh_dim=6,
        mesh_rel_dim=15,
        mesh_type_vocab=16,
        time_embed_dim=int(cfg.get("hidden_dim", 128)),
        use_spatial_attn=int(cfg.get("num_spatial_blocks", 1)) > 0,
        shape_dim=int(cfg.get("shape_dim", 10)),
        gender_num_embeddings=int(cfg.get("gender_num_embeddings", 3)),
        use_shape_condition=bool(cfg.get("use_shape_condition", False)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


def _accumulate_windows(delta_sum, weight_sum, delta_full, window_items, joint_ids):
    device = delta_full.device
    joint_ids_t = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
    for idx, item in enumerate(window_items):
        b = int(item["batch_index"])
        start = int(item["start_frame"])
        end = int(item["end_frame"])
        if start > end:
            continue
        length = end - start + 1
        center = (length - 1) / 2.0
        weights = 1.0 - (torch.abs(torch.arange(length, device=device) - center) / max(center, 1.0))
        weights = weights.clamp(min=0.1)
        weights = weights.view(1, 1, -1)

        delta_slice = delta_full[idx, :, :, :length]
        delta_sum[b, :, :, start : end + 1] += delta_slice * weights
        weight_sum[b, joint_ids_t, start : end + 1] += weights.squeeze(0)




def _softmin_distance(a_xyz, b_xyz, beta=30.0):
    if a_xyz.numel() == 0 or b_xyz.numel() == 0:
        return a_xyz.new_full((a_xyz.shape[0],), 1e6)
    dist = torch.linalg.norm(a_xyz[:, :, None, :] - b_xyz[:, None, :, :], dim=-1)
    dist = dist.reshape(dist.shape[0], -1)
    beta = float(beta)
    return -torch.logsumexp(-beta * dist, dim=-1) / max(beta, 1e-6)


def _post_cleanup(
    delta_avg,
    refined,
    actor,
    joint_ids,
    strength,
    margin,
    body_model,
    pose_rep,
    density,
    actor_betas=None,
    reactor_betas=None,
    actor_gender_id=None,
    reactor_gender_id=None,
    body_model_type=None,
):
    if strength <= 0.0:
        return delta_avg
    device = refined.device
    body_forward = RestoredBodyModelForward(
        body_model=body_model,
        pose_rep=pose_rep,
        translation=True,
        glob=True,
        device=device,
    )
    num_frames = refined.shape[-1]
    mask = torch.ones(refined.shape[0], num_frames, device=device, dtype=torch.bool)
    actor_verts = body_forward.motion_to_xyz(
        actor,
        mask=mask,
        jointstype="vertices",
        betas=actor_betas,
        gender_id=actor_gender_id,
        body_model_type=body_model_type,
    )
    refined_verts = body_forward.motion_to_xyz(
        refined,
        mask=mask,
        jointstype="vertices",
        betas=reactor_betas,
        gender_id=reactor_gender_id,
        body_model_type=body_model_type,
    )

    provider = get_mesh_region_provider(density=density, body_model=body_model, pose_rep=pose_rep)
    hand_ids = []
    for side in ("left", "right"):
        for ids in provider.reactor_hand_patch_ids(side).values():
            hand_ids.extend(ids)
    actor_ids = []
    for part in provider.actor_parts.values():
        for ids in part.values():
            actor_ids.extend(ids)

    if not hand_ids or not actor_ids:
        return delta_avg

    hand_ids_t = torch.as_tensor(sorted(set(hand_ids)), device=device, dtype=torch.long)
    actor_ids_t = torch.as_tensor(sorted(set(actor_ids)), device=device, dtype=torch.long)

    delta = delta_avg.clone()
    for b in range(refined.shape[0]):
        hand_xyz = refined_verts[b].index_select(0, hand_ids_t).permute(2, 0, 1)
        actor_xyz = actor_verts[b].index_select(0, actor_ids_t).permute(2, 0, 1)
        dist = _softmin_distance(hand_xyz, actor_xyz, beta=30.0)
        mask_close = dist < float(margin)
        if mask_close.any():
            scale = torch.ones_like(dist)
            scale = torch.where(mask_close, 1.0 - float(strength), scale)
            scale = scale.view(1, 1, -1)
            joint_ids_t = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
            delta_sel = delta[b].index_select(0, joint_ids_t) * scale
            delta[b].index_copy_(0, joint_ids_t, delta_sel)
    return delta


def _save_npz(path, arrays):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **arrays)


def _save_h5(path, arrays):
    import h5py

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in arrays.items():
            arr = np.asarray(value)
            if arr.dtype.kind in {"U", "S", "O"}:
                flat = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr.reshape(-1).tolist()]
                dt = h5py.string_dtype(encoding="utf-8")
                arr = np.asarray(flat, dtype=dt).reshape(arr.shape)
                f.create_dataset(key, data=arr, dtype=dt)
            else:
                f.create_dataset(key, data=arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", required=True, type=str)
    parser.add_argument("--blueprint_cache_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)

    parser.add_argument("--sampling_steps", default=50, type=int)
    parser.add_argument("--delta_clip", default=0.5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--max_batches", default=-1, type=int)
    parser.add_argument("--num_samples", default=-1, type=int)
    parser.add_argument("--max_windows_per_batch", default=128, type=int)

    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument("--window_size", default=12, type=int)
    parser.add_argument("--window_pad", default=2, type=int)
    parser.add_argument("--include_buffer", action="store_true")
    parser.add_argument("--density", default="medium", choices=["small", "medium"], type=str)
    parser.add_argument("--post_cleanup", action="store_true")
    parser.add_argument("--post_cleanup_strength", default=0.2, type=float)

    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_coarse", action="store_true")
    parser.add_argument("--save_config", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_path) and not args.overwrite:
        raise FileExistsError(f"output_path [{args.output_path}] already exists.")

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

    blueprint = np.load(args.blueprint_cache_path, allow_pickle=True)
    strict_windows_all = blueprint["strict_windows"]
    near_windows_all = blueprint["near_windows"]
    blueprint_band = blueprint["band"]
    blueprint_phase = blueprint["phase"]
    blueprint_conf = blueprint.get("active_prob", None)

    refiner, refiner_config = _load_refiner_checkpoint(args.model_path, device=device)
    joint_ids = refiner_config.get("joint_ids")
    if not joint_ids:
        joint_ids = default_refiner_joint_ids(include_buffer=args.include_buffer)

    diffusion_steps = int(refiner_config.get("diffusion_steps", 1000))
    noise_schedule = refiner_config.get("noise_schedule", "cosine")
    sampling_diffusion = create_spaced_diffusion(
        diffusion_steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=f"ddim{int(args.sampling_steps)}",
    )

    builder = DiffusionRefinerInputBuilder(
        body_model=args.body_model,
        pose_rep=args.pose_rep,
        translation=True,
        glob=True,
        window_size=args.window_size,
        window_pad=args.window_pad,
        include_buffer=args.include_buffer,
        density=args.density,
        device=device,
    )

    refined_list = []
    actor_list = []
    lengths_list = []
    indices_list = []
    coarse_list = []
    gt_list = []
    meta_lists = {key: [] for key in REQUIRED_CACHE_FIELDS + OPTIONAL_CACHE_FIELDS}

    total = 0
    total_batches = len(loader)
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=total_batches, desc="Run diffusion refiner")
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
            restoration_meta = extract_restoration_metadata(batch, device=device)
            actor_restored, coarse_restored = builder.restore_pair_batch(actor, coarse, restoration_meta)
            _, gt_restored = builder.restore_pair_batch(actor, gt, restoration_meta)

            strict_windows = strict_windows_all[batch_idx * args.batch_size : batch_idx * args.batch_size + actor.shape[0]]
            near_windows = near_windows_all[batch_idx * args.batch_size : batch_idx * args.batch_size + actor.shape[0]]
            band = torch.from_numpy(blueprint_band[batch_idx * args.batch_size : batch_idx * args.batch_size + actor.shape[0]]).to(device)
            phase = torch.from_numpy(blueprint_phase[batch_idx * args.batch_size : batch_idx * args.batch_size + actor.shape[0]]).to(device)
            conf = None
            if blueprint_conf is not None:
                conf = torch.from_numpy(blueprint_conf[batch_idx * args.batch_size : batch_idx * args.batch_size + actor.shape[0]]).to(device)

            window_items = []
            for b, items in enumerate(strict_windows):
                for win in items:
                    item = dict(win)
                    item["batch_index"] = b
                    window_items.append(item)
            for b, items in enumerate(near_windows):
                for win in items:
                    item = dict(win)
                    item["batch_index"] = b
                    window_items.append(item)

            if not window_items:
                refined = coarse_restored
            else:
                max_len = int(lengths.max().item())
                delta_sum = torch.zeros(actor.shape[0], coarse.shape[1], coarse.shape[2], max_len, device=device)
                weight_sum = torch.zeros(actor.shape[0], coarse.shape[1], max_len, device=device)

                for start_idx in range(0, len(window_items), args.max_windows_per_batch):
                    chunk_items = window_items[start_idx : start_idx + args.max_windows_per_batch]
                    window_batch = builder.build_window_batch(
                        actor,
                        coarse,
                        coarse,
                        lengths,
                        chunk_items,
                        {"band": band, "phase": phase},
                        blueprint_conf=conf,
                        restoration_meta=restoration_meta,
                    )
                    if window_batch is None:
                        continue

                    noise = torch.randn_like(window_batch["coarse_local"])
                    shape_tokens = refiner.encode_shape_tokens(
                        window_batch["actor_betas"],
                        window_batch["reactor_betas"],
                        window_batch["actor_gender_id"],
                        window_batch["reactor_gender_id"],
                    )
                    window_batch.update(shape_tokens)
                    model_kwargs = {
                        "coarse_local": window_batch["coarse_local"],
                        "actor_tokens": window_batch["actor_local_motion"],
                        "actor_mask": window_batch["actor_local_mask"],
                        "mesh_tokens": window_batch["mesh_token_feat"],
                        "mesh_token_type": window_batch["mesh_token_type"],
                        "mesh_mask": window_batch["mesh_token_mask"],
                        "cond_feat": window_batch["cond_feat"],
                        "mesh_relation_feat": window_batch["mesh_relation_features"],
                        "time_mask": window_batch["time_mask"],
                        "actor_shape_tokens": window_batch.get("actor_shape_tokens"),
                        "reactor_shape_tokens": window_batch.get("reactor_shape_tokens"),
                        "relative_shape_tokens": window_batch.get("relative_shape_tokens"),
                        "shape_mask": window_batch.get("shape_mask"),
                    }

                    residual = sampling_diffusion.p_sample_loop(
                        refiner,
                        noise.shape,
                        noise=noise,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        device=device,
                        progress=False,
                    )
                    residual = residual.clamp(-args.delta_clip, args.delta_clip)

                    joint_ids_t = torch.as_tensor(window_batch["joint_ids"], device=device, dtype=torch.long)
                    delta_full = torch.zeros_like(window_batch["coarse_full"])
                    delta_full.index_copy_(1, joint_ids_t, residual.permute(0, 2, 3, 1))

                    _accumulate_windows(
                        delta_sum,
                        weight_sum,
                        delta_full,
                        window_batch["window_items"],
                        window_batch["joint_ids"],
                    )

                weight = weight_sum.clamp(min=1.0).unsqueeze(2)
                delta_avg = delta_sum / weight
                delta_avg = delta_avg * (weight_sum.unsqueeze(2) > 0)
                if args.post_cleanup:
                    delta_avg = _post_cleanup(
                        delta_avg,
                        coarse_restored + delta_avg,
                        actor_restored,
                        joint_ids,
                        strength=args.post_cleanup_strength,
                        margin=0.02,
                        body_model=args.body_model,
                        pose_rep=args.pose_rep,
                        density=args.density,
                        actor_betas=restoration_meta["actor_betas"],
                        reactor_betas=restoration_meta["reactor_betas"],
                        actor_gender_id=restoration_meta["actor_gender_id"],
                        reactor_gender_id=restoration_meta["reactor_gender_id"],
                        body_model_type="smplx",
                    )
                refined = coarse_restored + delta_avg

            keep = refined.shape[0]
            if args.num_samples > 0 and total + keep > args.num_samples:
                keep = args.num_samples - total

            refined_list.append(refined[:keep].cpu().numpy())
            lengths_list.append(lengths[:keep].cpu().numpy())
            indices_list.append(sample_index[:keep].numpy())
            actor_list.append(actor_restored[:keep].cpu().numpy())
            coarse_list.append(coarse_restored[:keep].cpu().numpy())
            gt_list.append(gt_restored[:keep].cpu().numpy())
            for key in meta_lists.keys():
                value = batch[key]
                if torch.is_tensor(value):
                    value = value[:keep].cpu().numpy()
                elif isinstance(value, np.ndarray):
                    value = value[:keep]
                else:
                    value = value[:keep]
                meta_lists[key].append(value)

            total += keep
            pbar.set_postfix(samples=total)

    refined_motion = np.concatenate(refined_list, axis=0).astype(np.float32)
    lengths_out = np.concatenate(lengths_list, axis=0).astype(np.int64)
    indices_out = np.concatenate(indices_list, axis=0).astype(np.int64)

    output = {
        "refined_reactor_motion": refined_motion,
        "coarse_reactor_motion": np.concatenate(coarse_list, axis=0).astype(np.float32),
        "gt_reactor_motion": np.concatenate(gt_list, axis=0).astype(np.float32),
        "lengths": lengths_out,
        "sample_indices": indices_out,
        "actor_motion": np.concatenate(actor_list, axis=0).astype(np.float32),
    }
    for key, chunks in meta_lists.items():
        if not chunks:
            continue
        first = chunks[0]
        if isinstance(first, np.ndarray) and first.dtype.kind in {"U", "S", "O"}:
            output[key] = np.concatenate([np.asarray(x, dtype=object) for x in chunks], axis=0)
        elif isinstance(first, np.ndarray):
            output[key] = np.concatenate(chunks, axis=0)
        else:
            output[key] = np.asarray(sum([list(x) for x in chunks], []), dtype=object)
    if args.save_config:
        output["refiner_config_json"] = json.dumps(refiner_config).encode("utf-8")

    if args.output_path.endswith(".h5"):
        _save_h5(args.output_path, output)
    else:
        _save_npz(args.output_path, output)

    print(f"Saved refined motion to {args.output_path} (samples={refined_motion.shape[0]})")


if __name__ == "__main__":
    main()
