"""Writers for Stage2-lite inference outputs."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ensure_output_dir(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)


def write_refined_pack(output_dir: str, output_name: str, pack: dict[str, Any]) -> str:
    ensure_output_dir(output_dir)
    path = os.path.join(output_dir, output_name)
    np.savez_compressed(path, **pack)
    return path


def write_json(output_dir: str, filename: str, payload: dict[str, Any]) -> str:
    ensure_output_dir(output_dir)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
        f.write("\n")
    return path


def write_inference_outputs(
    output_dir: str,
    output_name: str,
    pack: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
    coverage_report: dict[str, Any] | None = None,
    debug_stats: dict[str, Any] | None = None,
    save_manifest: bool = True,
    save_coverage_report: bool = True,
    save_debug_stats: bool = False,
) -> dict[str, str]:
    paths = {
        "refined_pack": write_refined_pack(output_dir, output_name, pack),
    }
    if save_manifest and manifest is not None:
        paths["subset_manifest"] = write_json(output_dir, "subset_manifest.json", manifest)
    if save_coverage_report and coverage_report is not None:
        paths["coverage_report"] = write_json(output_dir, "coverage_report.json", coverage_report)
    if save_debug_stats and debug_stats is not None:
        paths["debug_stats"] = write_json(output_dir, "debug_stats.json", debug_stats)
    return paths
