import copy
import os
import re
import functools

import torch
from tqdm import tqdm

from data_loaders.get_data import get_dataset_loader
from data_loaders.tensors import collate, ccollate
from eval.a2m.stgcn.evaluate import Evaluation as STGCNEvaluation
from eval.a2m.stgcn_eval import _load_interx_action_names
from eval.a2m.tools import save_metrics, format_metrics
from model.refine.refine_model import RNetV1
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import refine_evaluation_parser


class RefineDataloader:
    def __init__(
        self,
        mode,
        stage1_model,
        diffusion,
        rnet,
        dataiterator,
        device,
        dataset,
        num_samples,
        num_person,
        body_model,
        sample_fn,
    ):
        assert mode in ["gt", "coarse", "refined"]
        self.batches = []

        with torch.no_grad():
            for motions, model_kwargs in tqdm(
                dataiterator, desc=f"Construct dataloader: {mode}.."
            ):
                motions = motions.to(device)
                if num_samples != -1 and len(self.batches) * dataiterator.batch_size > num_samples:
                    continue

                batch = dict()
                if mode == "gt":
                    batch["output"] = motions
                else:
                    for _k in model_kwargs["y"].keys():
                        if isinstance(model_kwargs["y"][_k], torch.Tensor):
                            model_kwargs["y"][_k] = model_kwargs["y"][_k].to(device)
                    coarse = sample_fn(
                        stage1_model,
                        motions.shape,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                    )
                    if mode == "refined":
                        lengths = model_kwargs["y"]["lengths"]
                        refined, _ = rnet(model_kwargs["y"]["cmotion"], coarse, lengths=lengths)
                        output_reactor = refined
                    else:
                        output_reactor = coarse

                    batch["output"] = torch.cat((model_kwargs["y"]["cmotion"], output_reactor), axis=2)
                    batch["text"] = model_kwargs["y"]["action_text"]

                max_n_frames = model_kwargs["y"]["lengths"].max()
                mask = model_kwargs["y"]["mask"].reshape(dataiterator.batch_size, max_n_frames).bool()

                batch["output_xyz"] = stage1_model.rot2xyz(
                    x=batch["output"],
                    mask=mask,
                    pose_rep="rot6d",
                    glob=True,
                    translation=True,
                    jointstype=body_model,
                    vertstrans=True,
                    betas=None,
                    beta=0,
                    glob_rot=None,
                    get_rotations_back=False,
                    num_person=num_person,
                )
                batch["lengths"] = model_kwargs["y"]["lengths"].to(device)
                batch["y"] = model_kwargs["y"]["action"].squeeze().long().cpu()
                self.batches.append(batch)

            num_samples_last_batch = num_samples % dataiterator.batch_size
            if num_samples_last_batch > 0 and self.batches:
                for k, v in self.batches[-1].items():
                    self.batches[-1][k] = v[:num_samples_last_batch]

    def __iter__(self):
        return iter(self.batches)


def build_rnet(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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


def build_stgcn_eval(args, device):
    if args.dataset == "chi3d":
        num_classes = 8
    elif args.dataset == "interx":
        action_names = _load_interx_action_names(args.data_path)
        if not action_names:
            raise ValueError("InterX action_setting.txt not found or empty.")
        num_classes = len(action_names)
    else:
        raise NotImplementedError("This dataset is not supported.")

    params = args.__dict__.copy()
    params["pose_rep"] = args.pose_rep
    params["nfeats"] = 6 * 2
    params["model_path"] = args.rec_model_path
    params["num_person"] = 2
    params["num_classes"] = num_classes

    return STGCNEvaluation(args.dataset, args.body_model, params, device)


def evaluate_refine(args, stage1_model, diffusion, rnet, data, acc_only=False):
    bs = args.batch_size
    device = dist_util.dev()
    stage1_model.eval()
    rnet.eval()

    stgcn_eval = build_stgcn_eval(args, device)

    data_types = ["train", "test"]
    datasetGT = {"train": [data], "test": [copy.deepcopy(data)]}
    for key in data_types:
        datasetGT[key][0].split = key

    allseeds = list(range(args.num_seeds))
    stgcn_metrics = {}

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    for index, seed in enumerate(allseeds):
        print(f"Evaluation number: {index + 1}/{args.num_seeds}")
        fixseed(seed)
        for key in data_types:
            for dataset in datasetGT[key]:
                dataset.reset_shuffle()
                dataset.shuffle()

        dataiterator = {
            key: [
                torch.utils.data.DataLoader(
                    datasetGT[key][0],
                    batch_size=bs,
                    shuffle=False,
                    num_workers=8,
                    drop_last=True,
                    collate_fn=collate,
                )
            ]
            for key in data_types
        }
        dataiterator_con = {
            key: [
                torch.utils.data.DataLoader(
                    datasetGT[key][0],
                    batch_size=bs,
                    shuffle=False,
                    num_workers=8,
                    drop_last=True,
                    collate_fn=ccollate,
                )
            ]
            for key in data_types
        }

        loader_builder = functools.partial(
            RefineDataloader,
            stage1_model=stage1_model,
            diffusion=diffusion,
            rnet=rnet,
            device=device,
            dataset=args.dataset,
            num_samples=args.num_samples,
            num_person=2,
            body_model=args.body_model,
            sample_fn=sample_fn,
        )

        gt_loaders = {
            key: loader_builder(mode="gt", dataiterator=dataiterator[key][0])
            for key in data_types
        }
        coarse_loaders = {
            key: loader_builder(mode="coarse", dataiterator=dataiterator_con[key][0])
            for key in data_types
        }
        refined_loaders = {
            key: loader_builder(mode="refined", dataiterator=dataiterator_con[key][0])
            for key in data_types
        }

        loaders = {"gt": gt_loaders, "coarse": coarse_loaders, "refined": refined_loaders}
        if acc_only:
            stgcn_metrics[seed] = stgcn_eval.evaluate_acc(stage1_model, loaders, setting="cmdm")
        else:
            stgcn_metrics[seed] = stgcn_eval.evaluate(stage1_model, loaders, setting="cmdm")
        del loaders

    metrics = {
        "feats": {
            key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds]
            for key in stgcn_metrics[allseeds[0]]
        }
    }
    return metrics


def main():
    args = refine_evaluation_parser()
    fixseed(args.seed)
    dist_util.setup_dist()

    print(f"Eval mode [{args.eval_mode}]")
    assert args.eval_mode in ["debug", "full"]
    if args.eval_mode == "debug":
        args.num_samples = 100
        args.num_seeds = 1
        acc_only = True
    else:
        args.num_samples = 1000
        args.num_seeds = 20
        acc_only = False

    num_frames = 150
    data_loader = get_dataset_loader(
        name=args.dataset,
        num_frames=num_frames,
        batch_size=args.batch_size,
        num_person=args.num_person,
        data_path=args.data_path,
        pose_rep=args.pose_rep,
        body_model="smplx",
        setting="cmdm",
    )

    print("creating model and diffusion...")
    stage1_model, diffusion = create_model_and_diffusion(args, data_loader)

    print(f"Loading stage1 checkpoint from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    load_model_wo_clip(stage1_model, state_dict)

    device = dist_util.dev()
    stage1_model.to(device)
    stage1_model.eval()

    print(f"Loading stage2 checkpoint from [{args.stage2_model_path}]...")
    rnet = build_rnet(args.stage2_model_path, device)

    eval_results = evaluate_refine(args, stage1_model, diffusion, rnet, data_loader.dataset, acc_only=acc_only)

    ckpt_name = os.path.basename(args.stage2_model_path)
    iter_nums = re.findall(r"\d+", ckpt_name)
    iter_id = int(iter_nums[0]) if iter_nums else 0
    auto = "auto" if args.auto_regressive else "not_auto"
    metricname = (
        "evaluation_results_refine_iter{}_samp{}_{}_{}_{}.yaml".format(
            iter_id, args.num_samples, auto, args.timestep_respacing, args.eval_mode
        )
    )
    evalpath = os.path.join(os.path.dirname(args.stage2_model_path), metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, eval_results)

    fid_to_print = {
        k: sum([float(vv) for vv in v]) / len(v)
        for k, v in eval_results["feats"].items()
        if "fid" in k and ("coarse" in k or "refined" in k)
    }
    print(fid_to_print)


if __name__ == "__main__":
    main()
