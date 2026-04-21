"""Tiny stderr progress display with ETA, avoiding extra dependencies."""

from __future__ import annotations

import sys
import time


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds == float("inf"):
        return "--:--:--"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressBar:
    def __init__(
        self,
        label: str,
        total: int | None,
        *,
        unit: str = "items",
        enabled: bool = True,
        stream=None,
    ):
        self.label = str(label)
        self.total = int(total) if total is not None else None
        self.unit = str(unit)
        self.enabled = bool(enabled)
        self.stream = stream or sys.stderr
        self.count = 0
        self.start_time: float | None = None
        self._last_len = 0

    def start(self):
        if not self.enabled:
            return self
        self.start_time = time.time()
        self._render()
        return self

    def update(self, step: int = 1):
        if not self.enabled:
            return
        if self.start_time is None:
            self.start()
        self.count += int(step)
        if self.total is not None:
            self.count = min(self.count, self.total)
        self._render()

    def finish(self):
        if not self.enabled:
            return
        if self.start_time is None:
            self.start()
        if self.total is not None:
            self.count = self.total
        self._render(final=True)
        self.stream.write("\n")
        self.stream.flush()

    def _render(self, final: bool = False):
        now = time.time()
        start = self.start_time or now
        elapsed = max(0.0, now - start)
        if self.total:
            pct = 100.0 * self.count / max(self.total, 1)
            rate = self.count / elapsed if elapsed > 0 else 0.0
            remaining = max(self.total - self.count, 0)
            eta = remaining / rate if rate > 0 else None
            progress = f"{self.count}/{self.total} {self.unit} | {pct:5.1f}%"
        else:
            eta = None
            progress = f"{self.count} {self.unit}"
        status = "done" if final else "ETA"
        text = (
            f"[{self.label}] {progress} | elapsed {_format_seconds(elapsed)} "
            f"| {status} {_format_seconds(eta)}"
        )
        padding = " " * max(0, self._last_len - len(text))
        self.stream.write("\r" + text + padding)
        self.stream.flush()
        self._last_len = len(text)

