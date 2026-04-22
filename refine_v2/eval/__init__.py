"""Evaluation and audit helpers for refine_v2.

Keep this package initializer lightweight. Some audit/report CLIs do not need
torch or SMPL-X body models, so heavyweight eval modules are imported by their
own CLI entrypoints instead of here.
"""

__all__: list[str] = []
