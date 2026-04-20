"""CLI: strict audit for refine_v2 windows against direct GT contact labels."""

from __future__ import annotations

import argparse
import os

from .audit_v2 import audit_windows, save_audit_json


def build_parser():
    parser = argparse.ArgumentParser(description="Audit refine_v2 selector windows against GT contact labels.")
    parser.add_argument("--contact_labels_path", required=True, type=str)
    parser.add_argument("--selector_windows_path", required=True, type=str)
    parser.add_argument("--output_json", required=True, type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = audit_windows(args.contact_labels_path, args.selector_windows_path)
    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_audit_json(args.output_json, payload)
    print(f"saved refine_v2 audit: {args.output_json}")
    for key, value in payload["metrics"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

