"""Small JSON/CSV/Markdown reporting helpers for refine_v2 subsets."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from refine_v2.data.schema import to_jsonable


def ensure_dir(path: str):
    os.makedirs(os.path.abspath(path), exist_ok=True)


def write_json(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def markdown_table(rows: list[dict[str, Any]], fieldnames: list[str], *, max_rows: int | None = None) -> str:
    use_rows = rows[:max_rows] if max_rows is not None else rows
    if not fieldnames:
        return ""
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in use_rows:
        values = []
        for key in fieldnames:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
