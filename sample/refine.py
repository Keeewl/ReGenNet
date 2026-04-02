import argparse
import json
import os

import numpy as np
import torch

from data_loaders.get_data import get_dataset_loader
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.cnet.cnet_v5 import CNetV5
from model.refine.refine_model import RNetV1
from model.rotation2xyz import Rotation2xyz_x
from utils.fixseed import fixseed
from utils.parser_util import refine_sample_args


def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = args.diffusion_steps
    scale_beta = 1.0
    timestep_respacing = args.timestep_respacing
    if getattr(args, "use_ddim", False) and not timestep_respacing:
        timestep_respacing = "ddim5"
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            gd.ModelVarType.FIXED_LARGE
            if not args.sigma_small
            else gd.ModelVarType.FIXED_SMALL
        )
        if not learn_sigma
        else gd.ModelVarType.LEARNED_RANGE,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_orient=args.lambda_orient,
        lambda_body=args.lambda_body,
        lambda_transl=args.lambda_transl,
        data_rep=args.pose_rep,
        num_person=1,
        body_model=args.body_model,
        vel_threshold=args.vel_threshold,
    )


def load_model_args(model_path):
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.exists(args_path):
        return {}
    with open(args_path, "r") as f:
        return json.load(f)


def merge_args(base, override):
    for key, val in override.items():
        if val is not None and val != "":
            base[key] = val
    return base


def build_stage1_model(args, num_actions):
    if args.body_model == "smpl":
        njoints = 25
    else:
        njoints = 56
    nfeats = 6 if args.pose_rep == "rot6d" else 3
    model = CNetV5(
        modeltype="",
        njoints=njoints,
        nfeats=nfeats,
        num_actions=num_actions,
        translation=True,
        pose_rep=args.pose_rep,
        glob=True,
        glob_rot=True,
        num_frames=args.num_frames,
        latent_dim=args.latent_dim,
        ff_size=1024,
        num_layers=args.layers,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        data_rep=args.pose_rep,
        cond_mode="no_cond" if args.unconstrained else "action",
        cond_mask_prob=args.cond_mask_prob,
        arch=args.arch,
        cm_mode=args.cm_mode,
        body_model=args.body_model,
        wo_pos_emb=args.wo_pos_emb,
        clip_version="ViT-B/32",
        dataset=args.dataset,
    )
    return model


def load_stage1_weights(model, model_path):
    state_dict = torch.load(model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:3]} ...")
    if missing:
        print(f"[warn] missing keys: {missing[:3]} ...")


def load_rnet_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = RNetV1(
        njoints=56,
        nfeats=6,
        body_model="smplx",
        pose_rep="rot6d",
        top_k=config.get("top_k", 5),
        window_size=config.get("window_size", 5),
        vel_threshold=config.get("vel_threshold", None),
        geom_sigma=config.get("geom_sigma", 0.1),
        hidden_dim=config.get("hidden_dim", 256),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def to_device(cond, device):
    if "y" not in cond:
        return cond
    cond = {"y": {k: v.to(device) if torch.is_tensor(v) else v for k, v in cond["y"].items()}}
    return cond


def local_pose_error(pred, gt, joint_ids, mask):
    joint_ids = torch.as_tensor(joint_ids, device=pred.device, dtype=torch.long)
    pred_local = pred.index_select(1, joint_ids).permute(0, 3, 1, 2)
    gt_local = gt.index_select(1, joint_ids).permute(0, 3, 1, 2)
    diff = pred_local - gt_local
    mask = mask.float()
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    extra = diff[0, 0].numel() if diff.dim() > 2 else 1
    denom = mask.sum() * extra
    denom = denom.clamp(min=1.0)
    return (diff * diff * mask).sum() / denom


def save_results(
    path,
    motion_xyz,
    output_rot6d,
    cmotion_rot6d,
    text,
    lengths,
    num_samples,
    num_repetitions,
    extra=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    root, ext = os.path.splitext(path)
    if ext != ".npy":
        path = root + ".npy"
    payload = {
        "motion": motion_xyz,
        "output": output_rot6d,
        "cmotion": cmotion_rot6d,
        "text": text,
        "lengths": lengths,
        "num_samples": num_samples,
        "num_repetitions": num_repetitions,
    }
    if extra:
        payload.update(extra)
    np.save(path, payload)
    return path


def to_xyz(rot2xyz, motion, lengths):
    batch_size, _, _, num_frames = motion.shape
    mask = torch.arange(num_frames, device=motion.device).view(1, -1) < lengths.view(-1, 1)
    return rot2xyz(
        x=motion,
        mask=mask,
        pose_rep="rot6d",
        glob=True,
        translation=True,
        jointstype="smplx",
        vertstrans=True,
        betas=None,
        beta=0,
        glob_rot=None,
        num_person=1,
    )


def main():
    args_cli = refine_sample_args()
    fixseed(args_cli.seed)

    if args_cli.cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{args_cli.device}")
    else:
        device = torch.device("cpu")

    if args_cli.cgenerate_results:
        data = np.load(args_cli.cgenerate_results, allow_pickle=True).item()
        actor_all = data["cmotion"]
        coarse_all = data["output"]
        lengths_all = data["lengths"]
        text_all = list(data.get("text", [""] * len(lengths_all)))
        num_samples = data.get("num_samples", len(lengths_all))
        num_repetitions = data.get("num_repetitions", 1)

        rnet = load_rnet_checkpoint(args_cli.stage2_model_path, device)
        rot2xyz = Rotation2xyz_x(device=device)

        refined_list = []
        motion_xyz_list = []
        batch_size = args_cli.batch_size
        for start in range(0, len(lengths_all), batch_size):
            end = min(len(lengths_all), start + batch_size)
            actor = torch.from_numpy(actor_all[start:end]).to(device)
            coarse = torch.from_numpy(coarse_all[start:end]).to(device)
            lengths = torch.as_tensor(lengths_all[start:end], device=device)
            refined, _ = rnet(actor, coarse, lengths=lengths)
            refined_list.append(refined.detach().cpu().numpy())
            motion_xyz = to_xyz(rot2xyz, refined, lengths)
            motion_xyz_list.append(motion_xyz.detach().cpu().numpy())

        refined_all = np.concatenate(refined_list, axis=0)
        motion_xyz_all = np.concatenate(motion_xyz_list, axis=0)
        saved_path = save_results(
            args_cli.output_path,
            motion_xyz_all,
            refined_all,
            actor_all,
            text_all,
            lengths_all,
            num_samples=num_samples,
            num_repetitions=num_repetitions,
        )
        print(f"Saved refined results to {saved_path}")
        return

    if args_cli.coarse_cache:
        data_cache = np.load(args_cli.coarse_cache, allow_pickle=True)
        actor_all = data_cache["actor_motion"]
        coarse_all = data_cache["reactor_coarse"]
        gt_all = data_cache.get("reactor_gt", None)
        lengths_all = data_cache["lengths"]
        sample_indices_all = data_cache.get("sample_indices", np.arange(len(lengths_all)))

        rnet = load_rnet_checkpoint(args_cli.stage2_model_path, device)

        refined_list = []
        motion_xyz_list = []
        rot2xyz = Rotation2xyz_x(device=device)
        for i in range(len(lengths_all)):
            actor = torch.from_numpy(actor_all[i:i+1]).to(device)
            coarse = torch.from_numpy(coarse_all[i:i+1]).to(device)
            lengths = torch.as_tensor([lengths_all[i]], device=device)
            refined, _ = rnet(actor, coarse, lengths=lengths)
            refined_list.append(refined.detach().cpu().numpy())
            motion_xyz = to_xyz(rot2xyz, refined, lengths)
            motion_xyz_list.append(motion_xyz.detach().cpu().numpy())

        refined_all = np.concatenate(refined_list, axis=0)
        motion_xyz_all = np.concatenate(motion_xyz_list, axis=0)
        text_all = [""] * len(lengths_all)
        saved_path = save_results(
            args_cli.output_path,
            motion_xyz_all,
            refined_all,
            actor_all,
            text_all,
            lengths_all,
            num_samples=len(lengths_all),
            num_repetitions=1,
        )
        print(f"Saved refined results to {saved_path}")
        return

    if not args_cli.stage1_model_path:
        raise ValueError("stage1_model_path is required unless --coarse_cache is provided.")

    stage1_args = load_model_args(args_cli.stage1_model_path)
    overrides = {
        "data_path": args_cli.data_path,
        "dataset": args_cli.dataset,
        "split": args_cli.split,
    }
    merged = merge_args(stage1_args, overrides)
    args = argparse.Namespace(**merged)
    args.use_ddim = args_cli.use_ddim
    if args_cli.timestep_respacing:
        args.timestep_respacing = args_cli.timestep_respacing
    args.output_path = args_cli.output_path
    args.max_batches = args_cli.max_batches
    args.num_samples = args_cli.num_samples
    args.stage1_model_path = args_cli.stage1_model_path

    print("Loading dataset...")
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_person=args.num_person,
        data_path=args.data_path,
        pose_rep=args.pose_rep,
        body_model=args.body_model,
        setting=args.setting,
        split=args.split,
    )

    num_actions = getattr(data.dataset, "num_actions", 1)
    stage1 = build_stage1_model(args, num_actions=num_actions)
    diffusion = create_gaussian_diffusion(args)
    stage1.to(device)
    stage1.eval()
    load_stage1_weights(stage1, args.stage1_model_path)

    rnet = load_rnet_checkpoint(args_cli.stage2_model_path, device)

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    actor_list = []
    coarse_list = []
    refined_list = []
    gt_list = []
    lengths_list = []
    indices_list = []
    text_list = []
    motion_xyz_list = []

    total = 0
    total_coarse_err = 0.0
    total_refined_err = 0.0
    total_frames = 0.0

    with torch.no_grad():
        for batch_idx, (motion, cond) in enumerate(data):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            if args.num_samples > 0 and total >= args.num_samples:
                break

            motion = motion.to(device)
            cond = to_device(cond, device)

            coarse = sample_fn(
                stage1,
                motion.shape,
                clip_denoised=False,
                model_kwargs=cond,
                progress=False,
                noise=None,
            )

            actor = cond["y"]["cmotion"]
            lengths = cond["y"]["lengths"]
            refined, aux = rnet(actor, coarse, lengths=lengths)
            rot2xyz = stage1.rot2xyz
            motion_xyz = to_xyz(rot2xyz, refined, lengths)

            bs = motion.shape[0]
            keep = bs
            if args.num_samples > 0 and total + bs > args.num_samples:
                keep = args.num_samples - total

            actor_list.append(actor[:keep].cpu().numpy())
            coarse_list.append(coarse[:keep].cpu().numpy())
            refined_list.append(refined[:keep].detach().cpu().numpy())
            gt_list.append(motion[:keep].cpu().numpy())
            lengths_list.append(lengths[:keep].cpu().numpy())
            indices_list.append(np.arange(total, total + keep, dtype=np.int64))
            if "action_text" in cond["y"]:
                text_list += cond["y"]["action_text"][:keep]
            else:
                text_list += [""] * keep
            motion_xyz_list.append(motion_xyz[:keep].detach().cpu().numpy())

            num_frames = motion.shape[-1]
            mask = torch.arange(num_frames, device=device).view(1, -1) < lengths.view(-1, 1)
            coarse_err = local_pose_error(coarse, motion, rnet.refine_joint_ids, mask)
            refined_err = local_pose_error(refined, motion, rnet.refine_joint_ids, mask)

            total_coarse_err += coarse_err.item() * keep
            total_refined_err += refined_err.item() * keep
            total_frames += keep

            total += keep

    actor_all = np.concatenate(actor_list, axis=0)
    coarse_all = np.concatenate(coarse_list, axis=0)
    refined_all = np.concatenate(refined_list, axis=0)
    gt_all = np.concatenate(gt_list, axis=0)
    lengths_all = np.concatenate(lengths_list, axis=0)
    sample_indices_all = np.concatenate(indices_list, axis=0)
    motion_xyz_all = np.concatenate(motion_xyz_list, axis=0)

    saved_path = save_results(
        args.output_path,
        motion_xyz_all,
        refined_all,
        actor_all,
        text_list,
        lengths_all,
        num_samples=len(lengths_all),
        num_repetitions=1,
    )

    if total_frames > 0:
        print(
            f"local_pose_err coarse={total_coarse_err/total_frames:.6f} "
            f"refined={total_refined_err/total_frames:.6f}"
        )
    print(f"Saved refined results to {saved_path}")


if __name__ == "__main__":
    main()
