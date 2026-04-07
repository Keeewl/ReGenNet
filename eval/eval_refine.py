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
from eval.metrics_contact import contact_distance, contact_distance_semantic
from model.refine.refine_model import RNetV1, RNetV2, RNetV3
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import refine_evaluation_parser



def split_actor_reactor_xyz(output_xyz):
    """
    output_xyz: [B, J, C, T] with C in {3, 6}
    returns actor_xyz/reactor_xyz: [B, J, 3, T]
    """
    if output_xyz.shape[2] == 6:
        actor_xyz = output_xyz[:, :, :3, :]
        reactor_xyz = output_xyz[:, :, 3:, :]
        return actor_xyz, reactor_xyz
    if output_xyz.shape[2] == 3 and output_xyz.shape[1] % 2 == 0:
        half = output_xyz.shape[1] // 2
        actor_xyz = output_xyz[:, :half, :, :]
        reactor_xyz = output_xyz[:, half:, :, :]
        return actor_xyz, reactor_xyz
    raise ValueError(f"Unexpected output_xyz shape: {tuple(output_xyz.shape)}")



def _accumulate_cd(stats, key, value, count):
    stats[f"{key}_sum"] += value * count
    stats[f"{key}_count"] += count



def _finalize_cd(stats, key):
    denom = stats[f"{key}_count"].clamp(min=1.0)
    return (stats[f"{key}_sum"] / denom).item()



def evaluate_contact_distance(coarse_loader, refined_loader, rnet, tau_contact=0.1):
    """
    Aggregate CD metrics with count-weighted averaging.
    """
    pair_mode = getattr(rnet, "pair_mode", "same_index")
    use_semantic = pair_mode == "semantic_nearest"
    if use_semantic:
        candidate_pairs = rnet.candidate_contact_pairs
        part_joint_ids = rnet.part_joint_ids
        topk_pairs = rnet.topk_pairs
    else:
        pair_ids = rnet.refine_joint_ids

    stats = {}

    def _init_stats(device):
        for name in ["coarse", "refined", "active_coarse", "active_refined"]:
            stats[f"{name}_sum"] = torch.zeros((), device=device)
            stats[f"{name}_count"] = torch.zeros((), device=device)

    def _cd_metrics(actor_xyz, pred_xyz, gt_xyz, lengths, active_mask=None):
        if use_semantic:
            return contact_distance_semantic(
                actor_xyz,
                pred_xyz,
                gt_xyz,
                candidate_pairs,
                part_joint_ids,
                topk_pairs,
                tau_contact=tau_contact,
                lengths=lengths,
                active_mask=active_mask,
            )
        return contact_distance(
            actor_xyz,
            pred_xyz,
            gt_xyz,
            pair_ids,
            pair_ids,
            tau_contact=tau_contact,
            lengths=lengths,
            active_mask=active_mask,
        )

    for mode, loader in [("coarse", coarse_loader), ("refined", refined_loader)]:
        for batch in loader:
            output_xyz = batch["output_xyz"]
            gt_xyz = batch["gt_xyz"]
            lengths = batch["lengths"]
            active_mask = batch.get("active_mask", None)

            actor_xyz, pred_xyz = split_actor_reactor_xyz(output_xyz)
            _, gt_reactor_xyz = split_actor_reactor_xyz(gt_xyz)

            metrics = _cd_metrics(actor_xyz, pred_xyz, gt_reactor_xyz, lengths, active_mask=None)

            if not stats:
                _init_stats(metrics["cd"].device)

            _accumulate_cd(stats, mode, metrics["cd"], metrics["count"])

            if active_mask is not None:
                metrics_active = _cd_metrics(
                    actor_xyz,
                    pred_xyz,
                    gt_reactor_xyz,
                    lengths,
                    active_mask=active_mask,
                )
                _accumulate_cd(stats, f"active_{mode}", metrics_active["cd"], metrics_active["count"])

    results = {
        "cd_coarse": _finalize_cd(stats, "coarse"),
        "cd_refined": _finalize_cd(stats, "refined"),
    }
    results["cd_improve"] = results["cd_coarse"] - results["cd_refined"]
    results["cd_active_coarse"] = _finalize_cd(stats, "active_coarse")
    results["cd_active_refined"] = _finalize_cd(stats, "active_refined")
    results["cd_active_improve"] = results["cd_active_coarse"] - results["cd_active_refined"]
    return results


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
                        refined, aux = rnet(model_kwargs["y"]["cmotion"], coarse, lengths=lengths)
                        output_reactor = refined
                        active_mask = aux["active_mask"]
                    else:
                        output_reactor = coarse
                        active_mask = None

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

                if mode != "gt":
                    gt_output = torch.cat((model_kwargs["y"]["cmotion"], motions), axis=2)
                    batch["gt_xyz"] = stage1_model.rot2xyz(
                        x=gt_output,
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
                    actor_xyz, reactor_xyz = split_actor_reactor_xyz(batch["output_xyz"])
                    if active_mask is None:
                        active_mask, _, _ = rnet.active_selector.select(
                            actor_xyz, reactor_xyz, lengths=batch["lengths"]
                        )
                    batch["active_mask"] = active_mask

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
    version = config.get("rnet_version", config.get("version", "v1"))

    if version == "v3":
        model = RNetV3(
            njoints=56,
            nfeats=6,
            body_model="smplx",
            pose_rep="rot6d",
            top_k=config.get("top_k", 5),
            window_size=config.get("window_size", 7),
            train_window_size=config.get("train_window_size", 10),
            vel_threshold=config.get("vel_threshold", None),
            geom_sigma=config.get("geom_sigma", 0.1),
            selector_sigma=config.get("selector_sigma", 0.1),
            selector_alpha=config.get("selector_alpha", 1.0),
            selector_beta=config.get("selector_beta", 0.5),
            selector_gamma=config.get("selector_gamma", 0.5),
            hidden_dim=config.get("hidden_dim", 256),
            num_temporal_blocks=config.get("num_temporal_blocks", 2),
            dropout=config.get("dropout", 0.1),
            pair_mode=config.get("pair_mode", "semantic_nearest"),
            topk_pairs=config.get("topk_pairs", 3),
            pair_reduce=config.get("pair_reduce", "mean"),
            use_contact_feature_aug=config.get("use_contact_feature_aug", True),
            pair_feature_topk=config.get("pair_feature_topk", 3),
            use_closing_speed=config.get("use_closing_speed", True),
            use_part_contact_summary=config.get("use_part_contact_summary", True),
            tau_contact=config.get("tau_contact", 0.1),
            tau_near=config.get("tau_near", 0.18),
            contact_error_margin=config.get("contact_error_margin", 0.05),
            gate_level=config.get("gate_level", "joint"),
            gate_init_bias=config.get("gate_init_bias", -2.0),
            bound_mode=config.get("bound_mode", "tanh"),
            delta_max=config.get("delta_max", 0.15),
        )
    elif version == "v2":
        model = RNetV2(
            njoints=56,
            nfeats=6,
            body_model="smplx",
            pose_rep="rot6d",
            top_k=config.get("top_k", 5),
            window_size=config.get("window_size", 5),
            vel_threshold=config.get("vel_threshold", None),
            geom_sigma=config.get("geom_sigma", 0.1),
            selector_sigma=config.get("selector_sigma", 0.1),
            selector_alpha=config.get("selector_alpha", 1.0),
            selector_beta=config.get("selector_beta", 0.5),
            selector_gamma=config.get("selector_gamma", 0.5),
            hidden_dim=config.get("hidden_dim", 256),
            num_temporal_blocks=config.get("num_temporal_blocks", 2),
            dropout=config.get("dropout", 0.1),
        )
    else:
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

    cd_metrics = {}

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

        cd_metrics[seed] = {}
        for split in data_types:
            tau_contact = getattr(rnet, "tau_contact", 0.1)
            cd_result = evaluate_contact_distance(
                coarse_loaders[split],
                refined_loaders[split],
                rnet,
                tau_contact=tau_contact,
            )
            for key, val in cd_result.items():
                cd_metrics[seed][f"{key}_{split}"] = val

    combined = {}
    for seed in allseeds:
        merged = {}
        merged.update(stgcn_metrics[seed])
        merged.update(cd_metrics[seed])
        combined[seed] = merged

    metrics = {
        "feats": {
            key: [format_metrics(combined[seed])[key] for seed in allseeds]
            for key in combined[allseeds[0]]
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
