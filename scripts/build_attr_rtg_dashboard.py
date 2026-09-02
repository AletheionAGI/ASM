#!/usr/bin/env python3
"""Build the final ATTR-RTG dashboard from its registered summary only."""
from __future__ import annotations

import argparse
from pathlib import Path

from attr_rtg_dashboard.charts import render_all
from attr_rtg_dashboard.data import dashboard_data, export_tables, load_summary
from attr_rtg_dashboard.html import render_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/benchmarks/attr_rtg/registered_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/attr_rtg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = load_summary(args.input)
    tables = dashboard_data(payload)
    export_tables(args.output, tables)
    render_all(tables, args.output)
    render_html(payload, tables, args.output)
    print(f"ATTR-RTG dashboard written to {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
