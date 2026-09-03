"""Render portable summary artifacts from already-computed scalar rows."""

from __future__ import annotations

import csv
import html
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .charts import render_rtg_charts
from .png import write_bar_png

DISCLAIMER = "LOCAL-ONLY, SINGLE-ADMINISTRATOR, NOT INDEPENDENTLY ATTESTED"


def render_summary(
    rows: Iterable[dict[str, Any]], output_dir: Path, *, synthetic: bool | None = None
) -> list[Path]:
    data = [dict(row) for row in rows]
    if synthetic is None:
        synthetic = all(
            row.get("status") == "SYNTHETIC" or row.get("regime") == "synthetic"
            for row in data
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg = output_dir / "summary.svg"
    png = output_dir / "summary.png"
    html_path = output_dir / "summary.html"
    json_path = output_dir / "manifest.json"
    csv_path = output_dir / "manifest.csv"
    eligible = [row for row in data if _finite_number(row.get("h8_nll"))]
    labels = [
        f"{row.get('arm', index)}/{row.get('seed', '-')}"
        for index, row in enumerate(eligible)
    ]
    values = [float(row["h8_nll"]) for row in eligible]
    svg_text = _svg(labels, values, synthetic=synthetic)
    svg.write_text(svg_text, encoding="utf-8")
    write_bar_png(png, values, labels)
    chart_paths = render_rtg_charts(data, output_dir)
    embedded_svgs = []
    for path in chart_paths:
        if path.suffix == ".svg":
            source = path.read_text(encoding="utf-8")
            embedded_svgs.append(
                f"<h2>{path.stem}</h2>" + source[source.index("<svg") :]
            )
    html_path.write_text(
        _html(data, [svg_text, *embedded_svgs], synthetic=synthetic), encoding="utf-8"
    )
    fields = sorted({key for row in data for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)
    artifact_names = [svg.name, png.name, html_path.name, csv_path.name, json_path.name]
    artifact_names.extend(path.name for path in chart_paths)
    manifest = {
        "schema_version": 1,
        "disclaimer": DISCLAIMER,
        "synthetic": synthetic,
        "artifacts": artifact_names,
        "rows": data,
    }
    json_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return [png, svg, html_path, json_path, csv_path, *chart_paths]


def _svg(labels: list[str], values: list[float], *, synthetic: bool) -> str:
    maximum = max(values, default=1.0) or 1.0
    title = "Synthetic H8 NLL summary" if synthetic else "Official H8 NLL summary"
    slot = 630 / max(1, len(values))
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        x = 65 + index * slot
        height = 220 * max(0.0, value) / maximum
        bars.append(
            f'<rect x="{x:.2f}" y="{290 - height:.2f}" width="{slot * 0.68:.2f}" height="{height:.2f}" fill="#2a6fbb"/>'
        )
        bars.append(
            f'<text transform="translate({x + slot * 0.34:.2f},305) rotate(55)" font-size="8">{html.escape(label)}</text>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="380" viewBox="0 0 720 380">'
        f'<rect width="100%" height="100%" fill="white"/><text x="20" y="28" font-size="18">{title}</text>'
        '<line x1="55" y1="60" x2="55" y2="290" stroke="black"/><line x1="55" y1="290" x2="705" y2="290" stroke="black"/>'
        + "".join(bars)
        + '<rect x="20" y="345" width="10" height="10" fill="#2a6fbb"/>'
        + f'<text x="35" y="354" font-size="10">H8 NLL; labels arm/seed</text><text x="250" y="370" font-size="10">{DISCLAIMER}</text></svg>'
    )


def _html(rows: list[dict[str, Any]], svgs: list[str], *, synthetic: bool) -> str:
    payload = html.escape(json.dumps(rows, indent=2, sort_keys=True))
    heading = (
        "ATTR-RTG-RCMZ synthetic summary"
        if synthetic
        else "ATTR-RTG-RCMZ official summary"
    )
    figures = "".join(f"<section>{svg}</section>" for svg in svgs)
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>ATTR-RTG-RCMZ summary</title>"
        "<style>body{font-family:system-ui;max-width:900px;margin:2rem auto}pre{background:#eee;padding:1rem}</style>"
        f"</head><body><h1>{heading}</h1><p>{DISCLAIMER}</p>{figures}<pre>{payload}</pre></body></html>"
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
