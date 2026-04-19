"""Summary builder for Stage2-lite evaluation results."""

from __future__ import annotations

import argparse
import json
from typing import Any


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _variant_delta(payload: dict[str, Any] | None, metric: str) -> dict[str, float] | None:
    if not payload or "coarse" not in payload or "refined" not in payload:
        return None
    if metric not in payload["coarse"] or metric not in payload["refined"]:
        return None
    coarse = float(payload["coarse"][metric])
    refined = float(payload["refined"][metric])
    return {
        "coarse": coarse,
        "refined": refined,
        "refined_minus_coarse": refined - coarse,
        "coarse_minus_refined": coarse - refined,
    }


def build_summary(
    *,
    local_eval: dict[str, Any] | None = None,
    global_eval: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_changes = {}
    for metric in (
        "hand_cd",
        "contact_ratio",
        "avg_contact_duration",
        "contact_frequency",
        "region_hand_dist",
        "penetration_rate",
        "penetration_depth",
    ):
        delta = _variant_delta(local_eval, metric)
        if delta is not None:
            local_changes[metric] = delta

    global_changes = {}
    for metric in ("accuracy", "fid", "diversity", "multimodality"):
        delta = _variant_delta(global_eval, metric)
        if delta is not None:
            global_changes[metric] = delta

    subset_protocol = {}
    if manifest:
        subset_protocol = {
            "sample_mode": manifest.get("sample_mode"),
            "seed": manifest.get("seed"),
            "num_samples_requested": manifest.get("num_samples_requested"),
            "num_samples_selected": manifest.get("num_samples_selected"),
            "reaction_data_path": manifest.get("reaction_data_path"),
            "checkpoint_path": manifest.get("checkpoint_path"),
        }

    return {
        "subset_protocol": subset_protocol,
        "coverage": coverage or {},
        "local_contact_eval": local_eval or {},
        "global_motion_eval": global_eval or {},
        "local_refined_vs_coarse": local_changes,
        "global_refined_vs_coarse": global_changes,
        "notes": [
            "local_contact_eval protocol = restored_pair_space; this is the main Stage2 hand/contact evaluation.",
            "global_motion_eval protocol = stage1_aligned_processed_space; this is an auxiliary recognition/distribution check.",
            "Do not directly mix local restored-space metrics with global processed-space STGCN metrics.",
            "Summary binds metrics to the infer subset manifest and coverage report for reproducibility.",
        ],
    }


def write_summary(path: str, payload: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_markdown(path: str, payload: dict[str, Any]):
    lines = [
        "# Stage2-Lite Eval Summary",
        "",
        "## Protocols",
        "",
        "- Local/contact: `restored_pair_space`.",
        "- Global/STGCN: `stage1_aligned_processed_space` after inverse restore.",
        "",
        "## Subset",
        "",
    ]
    subset = payload.get("subset_protocol", {})
    for key, value in subset.items():
        lines.append(f"- `{key}`: {value}")
    coverage = payload.get("coverage", {})
    lines.extend(["", "## Coverage", ""])
    for key in (
        "num_sequences",
        "num_actions_covered",
        "num_zero_window_sequences",
        "num_sequences_with_windows",
        "total_windows",
        "avg_windows_per_seq",
        "avg_covered_frame_ratio",
    ):
        if key in coverage:
            lines.append(f"- `{key}`: {coverage[key]}")
    lines.extend(["", "## Local Refined vs Coarse", ""])
    for metric, delta in payload.get("local_refined_vs_coarse", {}).items():
        lines.append(
            f"- `{metric}`: coarse={delta['coarse']:.6g}, refined={delta['refined']:.6g}, "
            f"refined_minus_coarse={delta['refined_minus_coarse']:.6g}"
        )
    lines.extend(["", "## Global Refined vs Coarse", ""])
    for metric, delta in payload.get("global_refined_vs_coarse", {}).items():
        lines.append(
            f"- `{metric}`: coarse={delta['coarse']:.6g}, refined={delta['refined']:.6g}, "
            f"refined_minus_coarse={delta['refined_minus_coarse']:.6g}"
        )
    lines.extend(["", "## Notes", ""])
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage2-lite eval summary.")
    parser.add_argument("--local_json", default="", type=str)
    parser.add_argument("--global_json", default="", type=str)
    parser.add_argument("--manifest_json", default="", type=str)
    parser.add_argument("--coverage_json", default="", type=str)
    parser.add_argument("--json_out", required=True, type=str)
    parser.add_argument("--md_out", default="", type=str)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    payload = build_summary(
        local_eval=_load_json(args.local_json),
        global_eval=_load_json(args.global_json),
        manifest=_load_json(args.manifest_json),
        coverage=_load_json(args.coverage_json),
    )
    write_summary(args.json_out, payload)
    if args.md_out:
        write_markdown(args.md_out, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
