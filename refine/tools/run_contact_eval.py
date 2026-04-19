"""Run the fixed Inter-X contact_dataset Stage2-lite protocol.

This is an orchestration wrapper for `interx_contact_dataset_v1`:

1. build or reuse a contact subset JSON from `reaction_data`
2. run Stage2-lite inference on that subset
3. run existing local/contact eval
4. optionally run existing global/STGCN eval
5. write the existing summary plus an explicit dataset protocol field

It does not change the refiner, windows, losses, training, inference algorithm,
or metric definitions.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from refine.eval.global_motion import evaluate_global_motion, _write_global_csv
from refine.eval.local_contact import evaluate_local_contact, _write_local_csv
from refine.eval.summary import build_summary, write_markdown, write_summary
from refine.infer.cli import Stage2LiteInferConfig
from refine.infer.runner import Stage2LiteInferRunner
from refine.protocols.interx_actions import (
    CONTACT_ACTION_LABELS,
    CONTACT_ACTION_NAMES,
    CONTACT_DATASET_PROTOCOL_NAME,
)
from refine.tools.build_contact_subset import build_contact_subset


def _write_json(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _load_json(path: str) -> dict[str, Any] | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dataset_protocol_payload() -> dict[str, Any]:
    return {
        "name": CONTACT_DATASET_PROTOCOL_NAME,
        "action_labels": sorted(CONTACT_ACTION_LABELS),
        "action_names": list(CONTACT_ACTION_NAMES),
    }


def _resolve_subset(args: argparse.Namespace) -> tuple[str, str]:
    os.makedirs(args.output_dir, exist_ok=True)
    subset_json = args.subset_json or os.path.join(args.output_dir, "contact_subset_indices.json")
    stats_json = os.path.join(args.output_dir, "contact_subset_stats.json")
    if args.subset_json:
        print(f"[contact_eval] Using provided subset: {subset_json}", flush=True)
        return subset_json, stats_json

    if args.reuse_existing_subset and os.path.exists(subset_json):
        print(f"[contact_eval] Reusing existing subset: {subset_json}", flush=True)
        return subset_json, stats_json

    subset, stats = build_contact_subset(
        args.reaction_data_path,
        allow_unknown_action=args.allow_unknown_action,
        sort_indices=True,
    )
    _write_json(subset_json, subset)
    _write_json(stats_json, stats)
    print(
        f"[contact_eval] Built contact subset: selected={subset['num_selected']} "
        f"total={subset['num_total']} -> {subset_json}",
        flush=True,
    )
    return subset_json, stats_json


def _run_infer(args: argparse.Namespace, subset_json: str):
    config = Stage2LiteInferConfig(
        reaction_data_path=args.reaction_data_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_mode="fixed",
        num_samples=-1,
        seed=args.seed,
        subset_indices_path=subset_json,
        per_action=0,
        output_name=args.output_name,
        save_manifest=True,
        save_coverage_report=True,
        save_debug_stats=args.save_debug_stats,
        body_model=args.body_model,
        pose_rep=args.pose_rep,
    )
    print("[contact_eval] Running Stage2-lite inference on contact_dataset.", flush=True)
    return Stage2LiteInferRunner(config).run()


def _run_local_eval(args: argparse.Namespace, pack_path: str) -> dict[str, Any]:
    local_json = os.path.join(args.output_dir, "local_eval.json")
    local_csv = os.path.join(args.output_dir, "local_eval.csv")
    local_batch_size = args.local_batch_size if args.local_batch_size > 0 else args.batch_size
    print(f"[contact_eval] Running local contact eval: batch_size={local_batch_size}", flush=True)
    payload = evaluate_local_contact(
        pack_path,
        device=args.device,
        body_model=args.body_model,
        pose_rep=args.pose_rep,
        batch_size=local_batch_size,
    )
    _write_json(local_json, payload)
    _write_local_csv(local_csv, payload)
    return payload


def _run_global_eval(args: argparse.Namespace, pack_path: str) -> dict[str, Any] | None:
    if args.skip_global:
        print("[contact_eval] Skipping global STGCN eval.", flush=True)
        return None
    if not args.stgcn_model_path:
        raise ValueError("--stgcn_model_path is required unless --skip_global is set.")

    global_json = os.path.join(args.output_dir, "global_eval.json")
    global_csv = os.path.join(args.output_dir, "global_eval.csv")
    global_batch_size = args.global_batch_size if args.global_batch_size > 0 else args.batch_size
    print(f"[contact_eval] Running global STGCN eval: batch_size={global_batch_size}", flush=True)
    payload = evaluate_global_motion(
        pack_path,
        dataset="interx",
        stgcn_model_path=args.stgcn_model_path,
        body_model=args.body_model,
        batch_size=global_batch_size,
        device=args.device,
        seed=args.seed,
    )
    _write_json(global_json, payload)
    _write_global_csv(global_csv, payload)
    return payload


def _write_eval_summary(
    args: argparse.Namespace,
    *,
    local_eval: dict[str, Any],
    global_eval: dict[str, Any] | None,
):
    manifest_path = os.path.join(args.output_dir, "subset_manifest.json")
    coverage_path = os.path.join(args.output_dir, "coverage_report.json")
    payload = build_summary(
        local_eval=local_eval,
        global_eval=global_eval,
        manifest=_load_json(manifest_path),
        coverage=_load_json(coverage_path),
    )
    payload["dataset_protocol"] = _dataset_protocol_payload()
    payload.setdefault("notes", []).append(
        "dataset_protocol = interx_contact_dataset_v1; this is a fixed contact-heavy Inter-X subset."
    )

    json_path = os.path.join(args.output_dir, "eval_summary.json")
    md_path = os.path.join(args.output_dir, "eval_summary.md")
    write_summary(json_path, payload)
    write_markdown(md_path, payload)
    print(f"[contact_eval] Wrote summary: {json_path}", flush=True)
    return payload


def run_contact_eval(args: argparse.Namespace) -> dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    subset_json, stats_json = _resolve_subset(args)
    pack_path = os.path.join(args.output_dir, args.output_name)

    infer_result = None
    if args.skip_infer:
        if not os.path.exists(pack_path):
            raise FileNotFoundError(f"--skip_infer requested but pack does not exist: {pack_path}")
        print(f"[contact_eval] Skipping inference; using existing pack: {pack_path}", flush=True)
    else:
        infer_result = _run_infer(args, subset_json)
        pack_path = infer_result["paths"]["refined_pack"]

    local_eval = _run_local_eval(args, pack_path)
    global_eval = _run_global_eval(args, pack_path)
    summary = _write_eval_summary(args, local_eval=local_eval, global_eval=global_eval)
    return {
        "pack_path": pack_path,
        "subset_json": subset_json,
        "stats_json": stats_json,
        "infer": infer_result,
        "local_eval": local_eval,
        "global_eval": global_eval,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed Inter-X contact_dataset Stage2-lite eval.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--checkpoint_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--stgcn_model_path", default="", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--local_batch_size", default=8, type=int)
    parser.add_argument("--global_batch_size", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument("--output_name", default="contact_refined_pack.npz", type=str)
    parser.add_argument("--subset_json", default="", type=str)
    parser.add_argument("--reuse_existing_subset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_unknown_action", action="store_true")
    parser.add_argument("--skip_infer", action="store_true")
    parser.add_argument("--skip_global", action="store_true")
    parser.add_argument("--save_debug_stats", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    result = run_contact_eval(args)
    print(
        json.dumps(
            {
                "dataset_protocol": _dataset_protocol_payload(),
                "pack_path": result["pack_path"],
                "subset_json": result["subset_json"],
                "stats_json": result["stats_json"],
                "summary_json": os.path.join(args.output_dir, "eval_summary.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
