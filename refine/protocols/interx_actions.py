"""Inter-X action labels for the fixed Stage2-lite contact_dataset protocol.

`interx_contact_dataset_v1` is a fixed hand/contact-focused subset used for
Stage2-lite development and reporting. It does not replace the existing
all-action infer/eval protocol; it only defines a reproducible subset protocol.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch


INTERX_ACTION_ID_TO_NAME = {
    "A000": "Hug",
    "A001": "Handshake",
    "A002": "Wave",
    "A003": "Grab",
    "A004": "Hit",
    "A005": "Kick",
    "A006": "Posing",
    "A007": "Push",
    "A008": "Pull",
    "A009": "Sit on leg",
    "A010": "Slap",
    "A011": "Pat on back",
    "A012": "Point finger at",
    "A013": "Walk towards",
    "A014": "Knock over",
    "A015": "Step on foot",
    "A016": "High-five",
    "A017": "Chase",
    "A018": "Whisper in ear",
    "A019": "Support with hand",
    "A020": "Rock-paper-scissors",
    "A021": "Dance",
    "A022": "Link arms",
    "A023": "Shoulder to shoulder",
    "A024": "Bend",
    "A025": "Carry on back",
    "A026": "Massaging shoulder",
    "A027": "Massaging leg",
    "A028": "Hand wrestling",
    "A029": "Chat",
    "A030": "Pat on cheek",
    "A031": "Thumb up",
    "A032": "Touch head",
    "A033": "Imitate",
    "A034": "Kiss on cheek",
    "A035": "Help up",
    "A036": "Cover mouth",
    "A037": "Look back",
    "A038": "Block",
    "A039": "Fly kiss",
}

INTERX_ACTION_NAME_TO_ID = {name: label for label, name in INTERX_ACTION_ID_TO_NAME.items()}

CONTACT_ACTION_LABELS = {
    "A000",
    "A001",
    "A003",
    "A007",
    "A008",
    "A011",
    "A016",
    "A019",
    "A020",
    "A022",
    "A028",
    "A030",
    "A034",
    "A035",
    "A036",
}

CONTACT_ACTION_NAMES = tuple(INTERX_ACTION_ID_TO_NAME[label] for label in sorted(CONTACT_ACTION_LABELS))
CONTACT_DATASET_PROTOCOL_NAME = "interx_contact_dataset_v1"

_ACTION_PATTERNS = (
    re.compile(r"A(\d+)"),
    re.compile(r"action[_-]?(\d+)", re.IGNORECASE),
)


def as_jsonable_scalar(value: Any):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_action_label_from_dataset_key(dataset_key: Any) -> tuple[str, str]:
    """Parse an Inter-X `Axxx` label from a dataset key.

    The behavior intentionally matches the original Stage2-lite infer fallback:
    parse `A<number>` first, then `action<number>`, otherwise group by a stable
    prefix before common separators, and finally return `unknown`.
    """

    key = str(as_jsonable_scalar(dataset_key))
    for pattern in _ACTION_PATTERNS:
        match = pattern.search(key)
        if match:
            return f"A{int(match.group(1)):03d}", "parsed_from_dataset_key"

    for sep in ("|", "/", "\\", ":", "_"):
        if sep in key:
            prefix = key.split(sep)[0]
            if prefix:
                return prefix, "fallback_dataset_key_prefix"
    return "unknown", "fallback_unknown"


def action_name_for_label(label: str) -> str:
    return INTERX_ACTION_ID_TO_NAME.get(str(label), "unknown")
