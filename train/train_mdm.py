# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
import time
import torch as th
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop, parse_resume_step_from_filename
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from mpi4py import MPI

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    
    # Separate TensorBoard runs per resume to avoid step stitching.
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
        if args.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
            run_name = f"run_{next_idx:03d}_resume_from_{resume_step}"
        else:
            run_name = (
                "run_000_initial"
                if next_idx == 0
                else f"run_{next_idx:03d}_initial"
            )
        tb_dir = os.path.join(tb_root, run_name)
        train_platform = train_platform_type(tb_dir)
    else:
        train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist()

    print("creating data loader...")
    if args.arch == 'trans_enc' or args.arch == 'mlp' or args.arch == 'gru' or args.arch == 'offline':
        arch_mode = 'offline'
    elif args.arch == 'trans_dec' or args.arch == 'online':
        arch_mode = 'online'
    if args.unconstrained:
        action_conditioned = False
    else:
        action_conditioned = True
    print("Setting:", args.setting, "| Dataset:", args.dataset, "| Arch:", arch_mode, "| Action conditioned:", action_conditioned)
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, 
                              num_person=args.num_person, data_path = args.data_path, pose_rep = args.pose_rep, body_model=args.body_model, setting=args.setting, ar_shuffle=args.shuffle,
                              shard=MPI.COMM_WORLD.Get_rank(), num_shards=MPI.COMM_WORLD.Get_size())

    # dist util debug
    if os.environ.get("REGENNET_DEBUG_RANKS", "0") == "1":
        rank = MPI.COMM_WORLD.Get_rank()
        world = MPI.COMM_WORLD.Get_size()
        local_rank = os.environ.get("LOCAL_RANK", "unknown")
        device = th.cuda.current_device() if th.cuda.is_available() else "cpu"
        data_len = len(data)
        dataset_len = len(data.dataset) if hasattr(data, "dataset") else "n/a"
        print(f"[debug] rank={rank}/{world} local_rank={local_rank} device={device} len(data)={data_len} len(dataset)={dataset_len}", flush=True)
        time.sleep(10)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
