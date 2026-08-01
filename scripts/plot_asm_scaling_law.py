from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Callable


LABELS = {
    "J_NO_DIRECTION": "ASM-R",
    "J_METRIC_ORTHONORMAL_DIRECTION": "ASM-F",
    "J": "ASM-X",
    "J_DIRECT_CONTROL_MATCHED": "ASM-S",
    "TRANSFORMER": "Transformer",
    "ASM_R_STREAM": "ASM-R",
    "ASM-R": "ASM-R",
    "ASM-C": "ASM-C",
    "Transformer": "Transformer",
    "Chance (1/64)": "Chance (1/64)",
    "ASM-C / ASM-R": "ASM-C / ASM-R",
    "MQAR_CHANCE": "MQAR chance",
}
COLORS = {
    "J_NO_DIRECTION": "#0f766e",
    "J_METRIC_ORTHONORMAL_DIRECTION": "#be123c",
    "J": "#d97706",
    "J_DIRECT_CONTROL_MATCHED": "#2563eb",
    "TRANSFORMER": "#111827",
    "ASM_R_STREAM": "#0f766e",
    "ASM-R": "#0f766e",
    "ASM-C": "#7c3aed",
    "Transformer": "#dc2626",
    "Chance (1/64)": "#9ca3af",
    "ASM-C / ASM-R": "#7c3aed",
    "MQAR_CHANCE": "#9ca3af",
}


def fmt_tokens(value: float) -> str:
    return f"{value / 1_000_000:g}M"


def svg_line_chart(
    path: Path,
    series: dict[str, list[dict]],
    x_key: str,
    y_key: str,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    x_format: Callable[[float], str],
    log_x: bool = False,
) -> None:
    width, height = 1200, 720
    left, right, top, bottom = 105, 55, 105, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [row for rows in series.values() for row in rows]
    xs = [float(row[x_key]) for row in values]
    ys = [float(row[y_key]) for row in values]
    xmap_values = [math.log10(value) for value in xs] if log_x else xs
    xmin, xmax = min(xmap_values), max(xmap_values)
    ymin, ymax = min(ys), max(ys)
    ypad = max((ymax - ymin) * 0.10, 0.01)
    ymin, ymax = ymin - ypad, ymax + ypad

    def xy(x: float, y: float) -> tuple[float, float]:
        mapped_x = math.log10(x) if log_x else x
        px = left + (mapped_x - xmin) / max(xmax - xmin, 1e-12) * plot_w
        py = top + (ymax - y) / max(ymax - ymin, 1e-12) * plot_h
        return px, py

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="38" font-family="Arial" font-size="25" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="{left}" y="66" font-family="Arial" font-size="14" fill="#4b5563">{html.escape(subtitle)}</text>',
    ]
    for index in range(6):
        value = ymin + (ymax - ymin) * index / 5
        y = xy(xs[0], value)[1]
        chunks.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.3f}</text>',
            ]
        )
    unique_x = sorted(set(xs))
    tick_xs = unique_x if len(unique_x) <= 9 else unique_x[:: max(1, len(unique_x) // 8)]
    for value in tick_xs:
        x = xy(value, ys[0])[0]
        chunks.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#f3f4f6"/>',
                f'<text x="{x:.2f}" y="{height-bottom+25}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{html.escape(x_format(value))}</text>',
            ]
        )
    for variant, rows in series.items():
        points = " ".join(f"{xy(float(row[x_key]), float(row[y_key]))[0]:.2f},{xy(float(row[x_key]), float(row[y_key]))[1]:.2f}" for row in rows)
        color = COLORS.get(variant, "#111827")
        chunks.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in rows:
            x, y = xy(float(row[x_key]), float(row[y_key]))
            chunks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
    legend_x = left + 15
    for index, variant in enumerate(series):
        x = legend_x + index * 170
        color = COLORS.get(variant, "#111827")
        chunks.append(f'<line x1="{x}" y1="88" x2="{x+28}" y2="88" stroke="{color}" stroke-width="4"/>')
        chunks.append(f'<text x="{x+36}" y="93" font-family="Arial" font-size="14" fill="#111827">{html.escape(LABELS.get(variant, variant))}</text>')
    chunks.extend(
        [
            f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">{html.escape(x_label)}</text>',
            f'<text x="25" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 25 {top+plot_h/2})" font-family="Arial" font-size="14" fill="#111827">{html.escape(y_label)}</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def svg_final_bar(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: float(row["validation_ce"]))
    width, height = 1000, 620
    left, right, top, bottom = 120, 60, 110, 90
    plot_h = height - top - bottom
    ymin = min(float(row["validation_ce"]) for row in rows) - 0.005
    ymax = max(float(row["validation_ce"]) for row in rows) + 0.005
    bar_w = 135
    gap = 75
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="40" font-family="Arial" font-size="25" font-weight="700" fill="#111827">Frozen validation CE at 100M tokens</text>',
        f'<text x="{left}" y="68" font-family="Arial" font-size="14" fill="#4b5563">Seed 1; lower is better. Axis is truncated to make the small differences visible.</text>',
    ]
    for index in range(5):
        value = ymin + (ymax - ymin) * index / 4
        y = top + (ymax - value) / (ymax - ymin) * plot_h
        chunks.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        chunks.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.3f}</text>')
    for index, row in enumerate(rows):
        variant = row["variant"]
        value = float(row["validation_ce"])
        x = left + 85 + index * (bar_w + gap)
        y = top + (ymax - value) / (ymax - ymin) * plot_h
        chunks.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_w}" height="{top+plot_h-y:.2f}" rx="5" fill="{COLORS.get(variant, "#111827")}"/>')
        chunks.append(f'<text x="{x+bar_w/2}" y="{y-10:.2f}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#111827">{value:.6f}</text>')
        chunks.append(f'<text x="{x+bar_w/2}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">{html.escape(LABELS.get(variant, variant))}</text>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot frozen ASM 100M scaling-law results.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["variant"] in LABELS]
    series = {
        variant: sorted((row for row in rows if row["variant"] == variant), key=lambda row: row["milestone_tokens"])
        for variant in LABELS
    }
    series = {variant: variant_rows for variant, variant_rows in series.items() if variant_rows}
    args.output_root.mkdir(parents=True, exist_ok=True)
    svg_line_chart(args.output_root / "validation_ce_by_tokens.svg", series, "milestone_tokens", "validation_ce", "ASM scaling law: validation CE by training tokens", "Frozen rescoring over the same 4,834,787-token validation stream; seed 1; lower is better.", "Training tokens (log scale)", "Cross-entropy", fmt_tokens, log_x=True)
    svg_line_chart(args.output_root / "validation_ppl_by_tokens.svg", series, "milestone_tokens", "validation_ppl", "ASM scaling law: perplexity by training tokens", "Frozen rescoring over the same validation stream; seed 1; lower is better.", "Training tokens (log scale)", "Perplexity", fmt_tokens, log_x=True)
    svg_line_chart(args.output_root / "validation_ce_by_gpu_hours.svg", series, "training_gpu_hours", "validation_ce", "ASM compute efficiency: validation CE by GPU time", "RTX 4090 training time recorded at each milestone; seed 1; lower and further left is better.", "Training GPU hours", "Cross-entropy", lambda value: f"{value:.2f}h")
    final_rows = [row for row in rows if int(row["milestone_tokens"]) == 100_000_000]
    svg_final_bar(args.output_root / "validation_ce_at_100m.svg", final_rows)
    with (args.output_root / "chart_data.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["variant", "asm_name", "milestone_tokens", "validation_ce", "validation_ppl", "training_gpu_hours"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: LABELS.get(row["variant"], row["variant"]) if field == "asm_name" else row[field] for field in fields})
    print(f"saved={args.output_root}")


if __name__ == "__main__":
    main()
