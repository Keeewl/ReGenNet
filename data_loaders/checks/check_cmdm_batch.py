import argparse
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from data_loaders.get_data import get_dataset_loader
from diffusion.resample import UniformSampler
from utils.model_util import create_gaussian_diffusion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load one CMDM/MDM batch and print tensor shapes/statistics."
    )
    parser.add_argument("--dataset", default="chi3d", choices=["chi3d"], type=str)
    parser.add_argument("--setting", default="cmdm", choices=["mdm", "cmdm"], type=str)
    parser.add_argument(
        "--body_model", default="smplx", choices=["smpl", "smplx"], type=str
    )
    parser.add_argument(
        "--arch",
        default="online",
        choices=["trans_enc", "trans_dec", "gru", "mlp", "online", "offline"],
        type=str,
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_frames", default=150, type=int)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument(
        "--data_path",
        default="dataset/chi3d/smplx/conditioned/chi3d_smplx_train.h5",
        type=str,
    )
    parser.add_argument("--num_person", default=None, type=int)
    parser.add_argument("--vel_threshold", default=0.01, type=float)

    return parser.parse_args()


def _tensor_min_max(flat):
    if flat.numel() == 0:
        return "empty", "empty", False
    if torch.is_floating_point(flat):
        nan_mask = torch.isnan(flat)
        has_nan = nan_mask.any().item()
        if has_nan:
            valid = flat[~nan_mask]
            if valid.numel() == 0:
                return "all_nan", "all_nan", True
            return valid.min().item(), valid.max().item(), True
        return flat.min().item(), flat.max().item(), False
    return flat.min().item(), flat.max().item(), False


def print_tensor_info(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return
    if not torch.is_tensor(tensor):
        print(f"{name}: not a tensor (type={type(tensor)})")
        return
    flat = tensor.detach().cpu().reshape(-1)
    tmin, tmax, has_nan = _tensor_min_max(flat)
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={tmin}, max={tmax}, has_nan={has_nan}"
    )


def infer_rep_info(njoints, nfeats, body_model, pose_rep):
    base_joints = 25 if body_model == "smpl" else 56
    parts = []
    if pose_rep == "rot6d" or nfeats % 6 == 0:
        mult = nfeats // 6
        parts.append(f"nfeats % 6 == 0 (nfeats/6 = {mult})")
    if njoints == base_joints:
        parts.append(f"njoints == base ({base_joints})")
    if njoints == base_joints + 1:
        parts.append(f"njoints == base+1 ({base_joints + 1}), translation may be extra joint")
    if not parts:
        return "no obvious rot6d/translation pattern detected"
    return "; ".join(parts)


def build_diffusion_args(args, num_person):
    return SimpleNamespace(
        timestep_respacing="",
        noise_schedule="cosine",
        sigma_small=True,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
        lambda_orient=1.0,
        lambda_body=1.0,
        lambda_transl=1.0,
        pose_rep=args.pose_rep,
        num_person=num_person,
        body_model=args.body_model,
        vel_threshold=args.vel_threshold,
    )


def main():
    args = parse_args()
    if args.num_person is None:
        num_person = 2 if args.setting == "cmdm" else 1
    else:
        num_person = args.num_person

    if not os.path.exists(args.data_path):
        print(f"data_path not found: {args.data_path}")
        print("Please pass --data_path to an existing .h5 file.")
        return

    print("creating data loader...")
    loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_person=num_person,
        data_path=args.data_path,
        pose_rep=args.pose_rep,
        body_model=args.body_model,
        setting=args.setting,
        ar_shuffle=False,
    )

    motion, cond = next(iter(loader))
    y = cond.get("y", {}) if isinstance(cond, dict) else {}
    cmotion = y.get("cmotion")

    print("\n== batch tensors ==")
    print_tensor_info("x", motion)
    print_tensor_info("y['cmotion']", cmotion)

    if torch.is_tensor(motion):
        bs, njoints, nfeats, nframes = motion.shape
        print(f"\n== parsed dims ==\nbs={bs}, njoints={njoints}, nfeats={nfeats}, nframes={nframes}")
        print(f"rep inference: {infer_rep_info(njoints, nfeats, args.body_model, args.pose_rep)}")

        if torch.is_tensor(cmotion):
            same_shape = tuple(motion.shape) == tuple(cmotion.shape)
            print(f"x vs y['cmotion'] same shape: {same_shape}")
            if not same_shape:
                print(f"x shape: {tuple(motion.shape)}")
                print(f"y['cmotion'] shape: {tuple(cmotion.shape)}")
                print("Note: shape mismatch can mean different conditioning format.")

    diffusion_args = build_diffusion_args(args, num_person)
    diffusion = create_gaussian_diffusion(diffusion_args)
    sampler = UniformSampler(diffusion)
    timesteps, _ = sampler.sample(motion.shape[0], device="cpu")
    print(
        f"\n== timesteps ==\nshape={tuple(timesteps.shape)}, dtype={timesteps.dtype}, "
        f"range=[{timesteps.min().item()}, {timesteps.max().item()}], "
        f"num_timesteps={diffusion.num_timesteps}"
    )


if __name__ == "__main__":
    main()
