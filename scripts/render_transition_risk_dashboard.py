#!/usr/bin/env python3
"""Render an offline ATTR dashboard from a summary JSON file."""

from __future__ import annotations
import argparse
from pathlib import Path
from aletheion_state_models.benchmarks.transition_risk.render import render_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    for path in render_smoke(args.summary):
        print(path)


if __name__ == "__main__":
    main()
