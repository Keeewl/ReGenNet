"""Aggregate one-seed table1 HiReact dry-run summaries into the final table row."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Any

import numpy as np


TABLE1_METRICS = ("fid", "accuracy", "diversity", "multimodality")
TABLE1_METRICS_DISPLAY = (
    ("fid", "FID"),
    ("accuracy", "Acc."),
    ("diversity", "Div."),
    ("multimodality", "Multimod."),
)


def _valformat(val: float, power: int = 3) -> str:
    p = float(pow(10, power))
    return str(np.round(p * val).astype(int) / p).ljust(4, "0")


def _format_interval(values: np.ndarray, key: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(np.mean(values))
    interval = float(1.96 * np.var(values))
    return {
        "mean": mean,
        "interval": interval,
        "formatted": f"{_valformat(mean, 3)} +/- {_valformat(interval, 4)}",
        "latex": rf"${_valformat(mean, 3)}^{{\pm{_valformat(interval, 4)}}}$",
        "metric": key,
    }


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _seed_from_path(path: str) -> int:
    m = re.search(r"seed(\d+)", path)
    return int(m.group(1)) if m else -1


def aggregate_summaries(summary_paths: list[str]) -> dict[str, Any]:
    if not summary_paths:
        raise ValueError("No summary files provided.")
    rows = []
    for path in sorted(summary_paths, key=_seed_from_path):
        payload = _read_json(path)
        rows.append(
            {
                "seed": int(_seed_from_path(path)),
                "path": path,
                "train": dict(payload["train"]["stgcn_metrics"]["refined"]),
                "test": dict(payload["test"]["stgcn_metrics"]["refined"]),
            }
        )

    out: dict[str, Any] = {
        "artifact": "table1_hireact_aggregate",
        "num_seeds": int(len(rows)),
        "seeds": [int(item["seed"]) for item in rows],
        "summary_paths": [item["path"] for item in rows],
        "raw": {
            "train_conditioned": {metric: [float(item["train"][metric]) for item in rows] for metric in TABLE1_METRICS},
            "test_conditioned": {metric: [float(item["test"][metric]) for item in rows] for metric in TABLE1_METRICS},
        },
    }
    out["train_conditioned"] = {
        metric: _format_interval(np.asarray(out["raw"]["train_conditioned"][metric], dtype=np.float64), metric)
        for metric in TABLE1_METRICS
    }
    out["test_conditioned"] = {
        metric: _format_interval(np.asarray(out["raw"]["test_conditioned"][metric], dtype=np.float64), metric)
        for metric in TABLE1_METRICS
    }
    out["table_row_preview"] = {
        "method": "HiReact",
        "train_conditioned": {display: out["train_conditioned"][metric]["formatted"] for metric, display in TABLE1_METRICS_DISPLAY},
        "test_conditioned": {display: out["test_conditioned"][metric]["formatted"] for metric, display in TABLE1_METRICS_DISPLAY},
    }
    return out


def _write_json(path: str, payload: dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_csv(path: str, payload: dict[str, Any]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "metric", "mean", "interval", "formatted"],
        )
        writer.writeheader()
        for split in ("train_conditioned", "test_conditioned"):
            block = payload[split]
            for metric in TABLE1_METRICS:
                item = block[metric]
                writer.writerow(
                    {
                        "split": split,
                        "metric": metric,
                        "mean": item["mean"],
                        "interval": item["interval"],
                        "formatted": item["formatted"],
                    }
                )


def _write_md(path: str, payload: dict[str, Any]):
    train = payload["table_row_preview"]["train_conditioned"]
    test = payload["table_row_preview"]["test_conditioned"]
    lines = [
        "# Table1 HiReact Aggregate",
        "",
        f"- num_seeds: `{payload['num_seeds']}`",
        f"- seeds: `{payload['seeds']}`",
        "",
        "| Method | Train FID↓ | Train Acc.↑ | Train Div.→ | Train Multimod.→ | Test FID↓ | Test Acc.↑ | Test Div.→ | Test Multimod.→ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| HiReact | {train['FID']} | {train['Acc.']} | {train['Div.']} | {train['Multimod.']} | {test['FID']} | {test['Acc.']} | {test['Div.']} | {test['Multimod.']} |",
        "",
        "Notes:",
        "- Interval formatting matches `eval/easy_table.py`.",
        "- Current interval is `1.96 * var(values)` to stay consistent with the existing table code.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate one-seed table1 HiReact dry-run summaries.")
    parser.add_argument(
        "--summary_glob",
        default="refine_v2/save/table1/hireact_seed*/hireact_dryrun/table1_hireact_dryrun_summary.json",
        type=str,
    )
    parser.add_argument("--json_out", required=True, type=str)
    parser.add_argument("--csv_out", required=True, type=str)
    parser.add_argument("--md_out", required=True, type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary_paths = glob.glob(args.summary_glob)
    payload = aggregate_summaries(summary_paths)
    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    _write_json(args.json_out, payload)
    _write_csv(args.csv_out, payload)
    _write_md(args.md_out, payload)
    print(f"aggregated seeds: {payload['seeds']}")
    print(json.dumps(payload["table_row_preview"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
