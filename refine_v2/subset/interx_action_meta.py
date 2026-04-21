"""Inter-X action metadata helpers with a lightweight fallback."""

from __future__ import annotations

import re
from typing import Any

try:
    from refine.protocols.interx_actions import (  # type: ignore
        INTERX_ACTION_ID_TO_NAME,
        INTERX_ACTION_NAME_TO_ID,
        action_name_for_label,
        parse_action_label_from_dataset_key,
    )
except Exception:
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
    _ACTION_PATTERNS = (
        re.compile(r"A(\d+)"),
        re.compile(r"action[_-]?(\d+)", re.IGNORECASE),
    )

    def _as_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def parse_action_label_from_dataset_key(dataset_key: Any) -> tuple[str, str]:
        key = _as_text(dataset_key)
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
