from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


COLORS = {
    "ASM-C2-FW-LM": "#6f2dbd",
    "ASM-R": "#00897b",
    "Transformer": "#ef6c00",
}


def line_chart(rows: list[dict]) -> str:
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 45, 75
    values = [
        float(row[key])
        for row in rows
        for key in ("candidate_ce", "asm_r_ce", "transformer_ce")
    ]
    low, high = min(values), max(values)
    padding = max((high - low) * 0.08, 0.01)
    low, high = low - padding, high + padding

    def x(index: int) -> float:
        return left + index * (width - left - right) / max(len(rows) - 1, 1)

    def y(value: float) -> float:
        return top + (high - value) * (height - top - bottom) / (high - low)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="26" text-anchor="middle" font-family="sans-serif" font-size="20">Frozen validation CE by seed lineage</text>',
    ]
    for tick in range(6):
        value = low + tick * (high - low) / 5
        position = y(value)
        parts.append(f'<line x1="{left}" y1="{position:.2f}" x2="{width-right}" y2="{position:.2f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-10}" y="{position+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{value:.3f}</text>')
    series = (
        ("ASM-C2-FW-LM", "candidate_ce"),
        ("ASM-R", "asm_r_ce"),
        ("Transformer", "transformer_ce"),
    )
    for series_index, (label, key) in enumerate(series):
        points = " ".join(f"{x(index):.2f},{y(float(row[key])):.2f}" for index, row in enumerate(rows))
        color = COLORS[label]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
        for index, row in enumerate(rows):
            parts.append(f'<circle cx="{x(index):.2f}" cy="{y(float(row[key])):.2f}" r="5" fill="{color}"/>')
        legend_x = left + series_index * 235
        parts.append(f'<rect x="{legend_x}" y="{height-34}" width="18" height="5" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+26}" y="{height-27}" font-family="sans-serif" font-size="14">{html.escape(label)}</text>')
    for index, row in enumerate(rows):
        parts.append(f'<text x="{x(index):.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="sans-serif" font-size="14">seed {row["seed"]}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def bar_chart(rows: list[dict]) -> str:
    width, height = 900, 520
    left, right, top, bottom = 90, 30, 45, 70
    plot_height = height - top - bottom
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="26" text-anchor="middle" font-family="sans-serif" font-size="20">MQAR accuracy at 32K by seed lineage</text>',
    ]
    for tick in range(0, 101, 20):
        position = top + (100 - tick) * plot_height / 100
        parts.append(f'<line x1="{left}" y1="{position:.2f}" x2="{width-right}" y2="{position:.2f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left-10}" y="{position+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{tick}%</text>')
    gate_y = top + 0.2 * plot_height
    parts.append(f'<line x1="{left}" y1="{gate_y:.2f}" x2="{width-right}" y2="{gate_y:.2f}" stroke="#c62828" stroke-width="2" stroke-dasharray="8 6"/>')
    slot = (width - left - right) / len(rows)
    for index, row in enumerate(rows):
        accuracy = 100 * float(row["mqar_32768_accuracy"])
        bar_height = accuracy * plot_height / 100
        bar_x = left + index * slot + slot * 0.2
        bar_width = slot * 0.6
        bar_y = top + plot_height - bar_height
        parts.append(f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{COLORS["ASM-C2-FW-LM"]}"/>')
        parts.append(f'<text x="{bar_x+bar_width/2:.2f}" y="{bar_y-8:.2f}" text-anchor="middle" font-family="sans-serif" font-size="14">{accuracy:.2f}%</text>')
        parts.append(f'<text x="{bar_x+bar_width/2:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="sans-serif" font-size="14">seed {row["seed"]}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot independent ASM-C2-FW-LM confirmation.")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.decision.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "language_ce_by_seed.svg").write_text(line_chart(data["seeds"]), encoding="utf-8")
    (args.output_root / "mqar_32k_by_seed.svg").write_text(bar_chart(data["seeds"]), encoding="utf-8")
    print(f"saved={args.output_root}")


if __name__ == "__main__":
    main()
