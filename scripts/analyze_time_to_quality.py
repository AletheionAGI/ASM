from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def model_family(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "gpt2" in name:
        return "gpt2"
    if "drm" in name:
        return "drm"
    return "unknown"


def seed_from_name(run_dir: Path) -> int | None:
    marker = "seed_"
    name = run_dir.name.lower()
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        else:
            break
    return int("".join(digits)) if digits else None


def load_monitor_rows(run_dir: Path) -> list[dict[str, Any]]:
    monitor = run_dir / "monitor.jsonl"
    rows: list[dict[str, Any]] = []
    if monitor.exists():
        for line in monitor.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    else:
        payload = load_json(run_dir / "metrics_latest.json")
        for row in payload.get("history", []):
            item = dict(row)
            item["elapsed_sec_cumulative"] = row.get("elapsed_sec")
            rows.append(item)
    rows = [row for row in rows if row.get("tokens_seen") is not None]
    rows.sort(key=lambda row: (int(row["tokens_seen"]), float(row.get("elapsed_sec_cumulative") or row.get("elapsed_sec") or 0.0)))
    deduped: dict[int, dict[str, Any]] = {}
    for row in rows:
        deduped[int(row["tokens_seen"])] = row
    return [deduped[token] for token in sorted(deduped)]


def plateau_status(
    rows: list[dict[str, Any]],
    *,
    window: int,
    min_improvement_per_million: float,
) -> tuple[bool, float | None]:
    points = [
        (int(row["tokens_seen"]), float(row["best_val_ce"]))
        for row in rows
        if row.get("best_val_ce") is not None
    ]
    if len(points) < window + 1:
        return False, None
    start_tokens, start_best = points[-window - 1]
    end_tokens, end_best = points[-1]
    token_delta_m = max((end_tokens - start_tokens) / 1_000_000.0, 1e-8)
    improvement_per_million = max(start_best - end_best, 0.0) / token_delta_m
    return improvement_per_million < min_improvement_per_million, improvement_per_million


def first_reach(
    rows: list[dict[str, Any]],
    target_ce: float,
) -> dict[str, Any] | None:
    for row in rows:
        best_val_ce = row.get("best_val_ce")
        if best_val_ce is not None and float(best_val_ce) <= target_ce:
            return row
    return None


def collect_runs(root: Path, target_ce: float | None, plateau_window: int, min_improvement_per_million: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    point_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        rows = load_monitor_rows(run_dir)
        summary = load_json(run_dir / "summary.json")
        run_config = load_json(run_dir / "run_config.json")
        family = model_family(run_dir)
        seed = seed_from_name(run_dir)
        for row in rows:
            point = dict(row)
            point["run"] = run_dir.name
            point["family"] = family
            point["seed"] = seed
            point_rows.append(point)
        plateau, improvement_rate = plateau_status(
            rows,
            window=plateau_window,
            min_improvement_per_million=min_improvement_per_million,
        )
        reached = first_reach(rows, target_ce) if target_ce is not None else None
        last = rows[-1] if rows else {}
        run_rows.append(
            {
                "run": run_dir.name,
                "family": family,
                "seed": seed,
                "parameter_count": summary.get("parameter_count") or run_config.get("parameter_count"),
                "tokens_seen": summary.get("tokens_seen") or last.get("tokens_seen"),
                "best_val_ce": summary.get("best_val_ce") or last.get("best_val_ce"),
                "elapsed_sec_cumulative": last.get("elapsed_sec_cumulative") or last.get("elapsed_sec"),
                "tokens_per_sec": last.get("tokens_per_sec"),
                "plateau_detected": plateau,
                "last_improvement_per_million": improvement_rate,
                "target_ce": target_ce,
                "target_reached": reached is not None,
                "tokens_to_target": reached.get("tokens_seen") if reached else None,
                "seconds_to_target": reached.get("elapsed_sec_cumulative") or reached.get("elapsed_sec") if reached else None,
            }
        )
    return point_rows, run_rows


def aggregate_runs(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["family"])].append(row)
    out = []
    metrics = ["best_val_ce", "tokens_per_sec", "elapsed_sec_cumulative", "tokens_to_target", "seconds_to_target"]
    for family, rows in sorted(grouped.items()):
        item: dict[str, Any] = {"family": family, "n": len(rows)}
        item["target_reached_count"] = sum(1 for row in rows if row.get("target_reached"))
        item["plateau_detected_count"] = sum(1 for row in rows if row.get("plateau_detected"))
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            if values:
                item[f"{metric}_mean"] = mean(values)
                item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
                item[f"{metric}_median"] = sorted(values)[len(values) // 2]
        out.append(item)
    return out


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def svg_line(path: Path, point_rows: list[dict[str, Any]], y_key: str, title: str, target_ce: float | None = None) -> None:
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in point_rows:
        if row.get(y_key) is None:
            continue
        label = f"{row['family']}_seed_{row.get('seed')}"
        series[label].append((float(row["tokens_seen"]) / 1_000_000.0, float(row[y_key])))
    width, height, margin = 1100, 620, 74
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    max_x = max(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_y = max(y for _, y in all_points)
    if target_ce is not None:
        min_y = min(min_y, target_ce)
        max_y = max(max_y, target_ce)
    pad_y = max((max_y - min_y) * 0.12, 0.05)
    min_y -= pad_y
    max_y += pad_y
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    palette = ["#0f766e", "#14b8a6", "#065f46", "#b91c1c", "#ef4444", "#7f1d1d", "#2563eb"]
    elems = []
    for idx, label in enumerate(sorted(series)):
        color = palette[idx % len(palette)]
        points = sorted(series[label])
        coords = []
        for x_value, y_value in points:
            x = margin + x_value / max(max_x, 1e-8) * plot_w
            y = margin + (max_y - y_value) / max(max_y - min_y, 1e-8) * plot_h
            coords.append(f"{x:.1f},{y:.1f}")
        elems.append(f"<polyline points='{' '.join(coords)}' fill='none' stroke='{color}' stroke-width='3'/>")
        legend_y = 90 + idx * 21
        elems.append(f"<line x1='{width - 300}' y1='{legend_y}' x2='{width - 260}' y2='{legend_y}' stroke='{color}' stroke-width='3'/>")
        elems.append(f"<text x='{width - 250}' y='{legend_y + 4}' font-size='12'>{html.escape(label)}</text>")
    if target_ce is not None:
        y = margin + (max_y - target_ce) / max(max_y - min_y, 1e-8) * plot_h
        elems.append(f"<line x1='{margin}' y1='{y:.1f}' x2='{width - margin}' y2='{y:.1f}' stroke='#111827' stroke-width='2' stroke-dasharray='8 6'/>")
        elems.append(f"<text x='{margin + 8}' y='{y - 8:.1f}' font-size='12'>target CE {target_ce:.4f}</text>")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">{html.escape(title)}</text>
<text x="{margin}" y="56" font-size="13" font-family="Arial" fill="#4b5563">x-axis: million tokens. Lower CE is better.</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111827"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111827"/>
{''.join(elems)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def svg_bar(path: Path, run_rows: list[dict[str, Any]], key: str, title: str, *, higher_is_better: bool = False) -> None:
    items = [(f"{row['family']}_s{row.get('seed')}", row.get(key)) for row in run_rows if row.get(key) is not None]
    width, height = 1100, 580
    if not items:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    values = [float(value) for _, value in items]
    max_v = max(values) * 1.15 + 1e-8
    x0, y0, bar_w, gap = 70, 470, 90, 24
    elems = []
    for idx, (label, value) in enumerate(items):
        number = float(value)
        h = number / max_v * 340
        x = x0 + idx * (bar_w + gap)
        y = y0 - h
        color = "#0f766e" if label.startswith("drm") else "#b91c1c"
        elems.append(f"<rect x='{x}' y='{y:.1f}' width='{bar_w}' height='{h:.1f}' fill='{color}'/>")
        elems.append(f"<text x='{x}' y='{y - 8:.1f}' font-size='11'>{number:.3g}</text>")
        elems.append(f"<text x='{x}' y='{y0 + 18}' font-size='10'>{html.escape(label)}</text>")
    direction = "Higher is better." if higher_is_better else "Lower is better."
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="60" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">{html.escape(title)}</text>
<text x="60" y="56" font-size="13" font-family="Arial" fill="#4b5563">{direction}</text>
<line x1="50" y1="{y0}" x2="{width - 40}" y2="{y0}" stroke="#111827"/>
{''.join(elems)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def svg_time_to_quality(path: Path, run_rows: list[dict[str, Any]], target_ce: float | None) -> None:
    items = []
    for row in run_rows:
        reached = bool(row.get("target_reached"))
        seconds = row.get("seconds_to_target") if reached else row.get("elapsed_sec_cumulative")
        if seconds is None:
            continue
        label = f"{row['family']}_s{row.get('seed')}"
        items.append((label, float(seconds), reached))
    width, height = 1100, 580
    if not items:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    max_v = max(value for _, value, _ in items) * 1.15 + 1e-8
    x0, y0, bar_w, gap = 80, 470, 120, 34
    elems = []
    for idx, (label, value, reached) in enumerate(items):
        h = value / max_v * 340
        x = x0 + idx * (bar_w + gap)
        y = y0 - h
        color = "#0f766e" if label.startswith("drm") else "#b91c1c"
        opacity = "1.0" if reached else "0.45"
        prefix = "" if reached else ">"
        elems.append(f"<rect x='{x}' y='{y:.1f}' width='{bar_w}' height='{h:.1f}' fill='{color}' opacity='{opacity}'/>")
        if not reached:
            elems.append(f"<line x1='{x}' y1='{y:.1f}' x2='{x + bar_w}' y2='{y:.1f}' stroke='{color}' stroke-width='4'/>")
        elems.append(f"<text x='{x}' y='{y - 8:.1f}' font-size='12'>{prefix}{value:.1f}s</text>")
        elems.append(f"<text x='{x}' y='{y0 + 18}' font-size='11'>{html.escape(label)}</text>")
        status = "reached" if reached else "not reached"
        elems.append(f"<text x='{x}' y='{y0 + 34}' font-size='10' fill='#4b5563'>{status}</text>")
    target_text = f"Target CE: {target_ce:.4f}" if target_ce is not None else "Target CE: n/a"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="60" y="34" font-size="22" font-family="Arial" font-weight="700" fill="#111827">Time To Quality</text>
<text x="60" y="56" font-size="13" font-family="Arial" fill="#4b5563">{html.escape(target_text)}. Faded bars with '&gt;' are censored runs that did not reach target.</text>
<line x1="50" y1="{y0}" x2="{width - 40}" y2="{y0}" stroke="#111827"/>
{''.join(elems)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_dashboard(root: Path, target_ce: float | None, run_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    run_trs = []
    for row in run_rows:
        run_trs.append(
            "<tr>"
            f"<td>{html.escape(str(row['run']))}</td>"
            f"<td>{html.escape(str(row['family']))}</td>"
            f"<td>{row.get('seed')}</td>"
            f"<td>{fmt(row.get('best_val_ce'))}</td>"
            f"<td>{fmt(row.get('tokens_seen'), 0)}</td>"
            f"<td>{fmt(row.get('tokens_per_sec'), 0)}</td>"
            f"<td>{fmt(row.get('seconds_to_target'), 1)}</td>"
            f"<td>{html.escape(str(row.get('target_reached')))}</td>"
            f"<td>{html.escape(str(row.get('plateau_detected')))}</td>"
            "</tr>"
        )
    agg_trs = []
    for row in aggregate_rows:
        agg_trs.append(
            "<tr>"
            f"<td>{html.escape(str(row['family']))}</td>"
            f"<td>{row.get('n')}</td>"
            f"<td>{fmt(row.get('best_val_ce_mean'))}</td>"
            f"<td>{fmt(row.get('tokens_per_sec_mean'), 0)}</td>"
            f"<td>{fmt(row.get('seconds_to_target_median'), 1)}</td>"
            f"<td>{row.get('target_reached_count')}</td>"
            f"<td>{row.get('plateau_detected_count')}</td>"
            "</tr>"
        )
    graph_names = [
        "time_to_quality_by_seed.svg",
        "best_val_ce_by_tokens.svg",
        "val_ce_by_tokens.svg",
        "seconds_to_target_by_seed.svg",
        "tokens_per_sec_by_seed.svg",
    ]
    graphs = []
    for name in graph_names:
        graph = root / name
        if graph.exists():
            graphs.append(f"<section><h2>{html.escape(name)}</h2>{graph.read_text(encoding='utf-8')}</section>")
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Time To Quality</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #f9fafb; }}
    table {{ border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 10px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    section {{ background: white; border: 1px solid #d1d5db; padding: 16px; margin: 20px 0; }}
    svg {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Time To Quality</h1>
  <p>Target CE: {fmt(target_ce)}</p>
  <section>
    <h2>Aggregate</h2>
    <table>
      <thead><tr><th>family</th><th>n</th><th>best CE mean</th><th>tokens/s mean</th><th>seconds to target median</th><th>target reached</th><th>plateau</th></tr></thead>
      <tbody>{''.join(agg_trs)}</tbody>
    </table>
  </section>
  <section>
    <h2>Runs</h2>
    <table>
      <thead><tr><th>run</th><th>family</th><th>seed</th><th>best CE</th><th>tokens</th><th>tokens/s</th><th>seconds to target</th><th>target reached</th><th>plateau</th></tr></thead>
      <tbody>{''.join(run_trs)}</tbody>
    </table>
  </section>
  {''.join(graphs)}
</body>
</html>
"""
    (root / "dashboard.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DRM/GPT-2 time-to-quality runs and generate SVG charts.")
    parser.add_argument("--root", default="runs/time_to_quality_drm_gpt2_37m")
    parser.add_argument("--target-ce", type=float, default=None)
    parser.add_argument("--target-margin-ce", type=float, default=0.01)
    parser.add_argument("--plateau-window", type=int, default=3)
    parser.add_argument("--min-improvement-per-million", type=float, default=0.003)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"missing root: {root}")

    point_rows, first_pass_runs = collect_runs(
        root,
        args.target_ce,
        args.plateau_window,
        args.min_improvement_per_million,
    )
    target_ce = args.target_ce
    if target_ce is None:
        drm_best = [float(row["best_val_ce"]) for row in first_pass_runs if row["family"] == "drm" and row.get("best_val_ce") is not None]
        target_ce = mean(drm_best) + args.target_margin_ce if drm_best else None
    point_rows, run_rows = collect_runs(
        root,
        target_ce,
        args.plateau_window,
        args.min_improvement_per_million,
    )
    aggregate_rows = aggregate_runs(run_rows)

    save_csv(root / "time_to_quality_points.csv", point_rows)
    save_csv(root / "time_to_quality_runs.csv", run_rows)
    save_csv(root / "time_to_quality_aggregate.csv", aggregate_rows)
    save_json(
        root / "time_to_quality_status.json",
        {
            "target_ce": target_ce,
            "target_margin_ce": args.target_margin_ce,
            "plateau_window": args.plateau_window,
            "min_improvement_per_million": args.min_improvement_per_million,
            "runs": run_rows,
            "aggregate": aggregate_rows,
        },
    )
    svg_line(root / "best_val_ce_by_tokens.svg", point_rows, "best_val_ce", "Best Validation CE By Tokens", target_ce)
    svg_line(root / "val_ce_by_tokens.svg", point_rows, "val_ce", "Validation CE By Tokens", target_ce)
    svg_time_to_quality(root / "time_to_quality_by_seed.svg", run_rows, target_ce)
    svg_bar(root / "seconds_to_target_by_seed.svg", run_rows, "seconds_to_target", "Seconds To Target CE")
    svg_bar(root / "tokens_per_sec_by_seed.svg", run_rows, "tokens_per_sec", "Tokens Per Second By Seed", higher_is_better=True)
    write_dashboard(root, target_ce, run_rows, aggregate_rows)
    print(f"target_ce={target_ce}")
    print(f"saved={root / 'dashboard.html'}")


if __name__ == "__main__":
    main()
