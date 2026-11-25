"""Unified per-run report hub: one offline index.html with Overview / Bodily Map / Session Report / Validation.

Usage: python scripts/report_hub.py <run_dir> [--out <file>]
Reads existing run outputs only; writes <run_dir>/index.html by default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.bodily_map.hub import build_hub  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help="output file (default <run_dir>/index.html)")
    a = ap.parse_args()
    path = build_hub(a.run_dir, a.out)
    print("hub written to", path)


if __name__ == "__main__":
    main()
