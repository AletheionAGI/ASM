from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from plot_asm_scaling_law import COLORS, LABELS, fmt_tokens, svg_line_chart


VARIANT_DIRS = {
    "J_NO_DIRECTION": "variant_j_no_direction_seed_1",
    "J_METRIC_ORTHONORMAL_DIRECTION": "variant_j_metric_orthonormal_direction_seed_1",
    "J": "variant_j_seed_1",
    "J_DIRECT_CONTROL_MATCHED": "variant_j_direct_control_matched_seed_1",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest(history: list[dict], tokens: int) -> dict:
    return min(history, key=lambda row: abs(int(row["tokens_seen"]) - tokens))


def smooth_train(history: list[dict], bucket_tokens: int = 1_000_000) -> list[dict]:
    buckets: dict[int, list[dict]] = {}
    for row in history:
        bucket = max(1, round(int(row["tokens_seen"]) / bucket_tokens))
        buckets.setdefault(bucket, []).append(row)
    return [
        {
            "milestone_tokens": bucket * bucket_tokens,
            "train_ce": sum(float(row["train_ce"]) for row in rows) / len(rows),
        }
        for bucket, rows in sorted(buckets.items())
    ]


def decimate(rows: list[dict], maximum: int = 120) -> list[dict]:
    if len(rows) <= maximum:
        return rows
    stride = math.ceil(len(rows) / maximum)
    selected = rows[::stride]
    if selected[-1] is not rows[-1]:
        selected.append(rows[-1])
    return selected


def bar_chart(path: Path, rows: list[dict], key: str, title: str, subtitle: str, ylabel: str, lower: bool) -> None:
    rows = sorted(rows, key=lambda row: float(row[key]), reverse=not lower)
    width, height = 1100, 650
    left, right, top, bottom = 115, 55, 115, 95
    plot_h = height - top - bottom
    values = [float(row[key]) for row in rows]
    ymin = min(values) - max((max(values) - min(values)) * 0.12, abs(min(values)) * 0.02)
    ymin = max(0.0, ymin)
    ymax = max(values) + max((max(values) - min(values)) * 0.12, abs(max(values)) * 0.02)
    bar_w, gap = 120, 65
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="40" font-family="Arial" font-size="25" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{left}" y="68" font-family="Arial" font-size="14" fill="#4b5563">{subtitle}</text>',
    ]
    for index in range(6):
        value = ymin + (ymax - ymin) * index / 5
        y = top + (ymax - value) / (ymax - ymin) * plot_h
        chunks.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        chunks.append(f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{value:.3f}</text>')
    for index, row in enumerate(rows):
        variant, value = row["variant"], float(row[key])
        x = left + 60 + index * (bar_w + gap)
        y = top + (ymax - value) / (ymax - ymin) * plot_h
        chunks.append(f'<rect x="{x}" y="{y:.2f}" width="{bar_w}" height="{top+plot_h-y:.2f}" rx="5" fill="{COLORS[variant]}"/>')
        chunks.append(f'<text x="{x+bar_w/2}" y="{y-10:.2f}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#111827">{value:.4f}</text>')
        chunks.append(f'<text x="{x+bar_w/2}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">{LABELS[variant]}</text>')
    chunks.append(f'<text x="28" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 28 {top+plot_h/2})" font-family="Arial" font-size="14" fill="#111827">{ylabel}</text>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def scatter_chart(path: Path, rows: list[dict], x_key: str, y_key: str, title: str, x_label: str) -> None:
    width, height = 1050, 680
    left, right, top, bottom = 110, 70, 105, 90
    plot_w, plot_h = width-left-right, height-top-bottom
    xs, ys = [float(row[x_key]) for row in rows], [float(row[y_key]) for row in rows]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xpad, ypad = max((xmax-xmin)*.12, xmax*.03), max((ymax-ymin)*.12, .01)
    xmin, xmax, ymin, ymax = max(0, xmin-xpad), xmax+xpad, ymin-ypad, ymax+ypad
    def xy(x: float, y: float) -> tuple[float,float]:
        return left+(x-xmin)/(xmax-xmin)*plot_w, top+(ymax-y)/(ymax-ymin)*plot_h
    chunks=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="#fff"/>',f'<text x="{left}" y="40" font-family="Arial" font-size="25" font-weight="700">{title}</text>',f'<text x="{left}" y="68" font-family="Arial" font-size="14" fill="#4b5563">Frozen full-validation CE; lower and further left is better.</text>']
    for i in range(6):
        xv=xmin+(xmax-xmin)*i/5; yv=ymin+(ymax-ymin)*i/5
        x,_=xy(xv,ymin); _,y=xy(xmin,yv)
        chunks += [f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#f3f4f6"/>',f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#4b5563">{xv:.2f}</text>',f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>',f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">{yv:.3f}</text>']
    for row in rows:
        x,y=xy(float(row[x_key]),float(row[y_key])); variant=row['variant']
        chunks += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="{COLORS[variant]}"/>',f'<text x="{x+12:.2f}" y="{y-10:.2f}" font-family="Arial" font-size="14" font-weight="700">{LABELS[variant]}</text>']
    chunks += [f'<text x="{left+plot_w/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>',f'<text x="25" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 25 {top+plot_h/2})" font-family="Arial" font-size="14">Validation CE</text>','</svg>']
    path.write_text('\n'.join(chunks)+'\n',encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the complete 100M ASM/Transformer comparison.")
    parser.add_argument("--asm-root", type=Path, default=Path("runs/asm_scaling_law_100m_seed1"))
    parser.add_argument("--asm-summary", type=Path, default=Path("docs/benchmarks/asm_scaling_law_100m_seed1/scaling_law_summary.json"))
    parser.add_argument("--transformer-root", type=Path, default=Path("runs/transformer_asm_r_matched_100m_seed1"))
    parser.add_argument("--output-root", type=Path, default=Path("docs/benchmarks/asm_transformer_100m_seed1/charts"))
    args = parser.parse_args()
    frozen = load(args.asm_summary)
    histories = {variant: load(args.asm_root / directory / "metrics_latest.json")["history"] for variant, directory in VARIANT_DIRS.items()}
    histories["TRANSFORMER"] = load(args.transformer_root / "metrics_latest.json")["history"]
    final_rows=[]
    for variant in VARIANT_DIRS:
        row=next(row for row in frozen["rows"] if row["variant"]==variant and int(row["milestone_tokens"])==100_000_000)
        run=load(args.asm_root / VARIANT_DIRS[variant] / "run_config.json")
        latest=nearest(histories[variant],100_000_000)
        final_rows.append({"variant":variant,"parameters":run["parameter_count"],"validation_ce":row["validation_ce"],"validation_ppl":row["validation_ppl"],"gpu_hours":row["training_gpu_hours"],"tokens_per_sec":latest["tokens_per_sec"]})
    tr_val=load(args.transformer_root/"validation_full.json"); tr_run=load(args.transformer_root/"run_config.json"); tr_latest=histories["TRANSFORMER"][-1]
    final_rows.append({"variant":"TRANSFORMER","parameters":tr_run["parameter_count"],"validation_ce":tr_val["test_ce"],"validation_ppl":tr_val["test_ppl"],"gpu_hours":tr_latest["elapsed_sec"]/3600,"tokens_per_sec":tr_latest["tokens_per_sec"]})
    args.output_root.mkdir(parents=True,exist_ok=True)
    sampled={variant:[{"milestone_tokens":row["tokens_seen"],"val_ce":row["val_ce"]} for row in rows if row.get("val_ce") is not None] for variant,rows in histories.items()}
    smoothed={variant:smooth_train(rows) for variant,rows in histories.items()}
    throughput={variant:decimate([{"milestone_tokens":row["tokens_seen"],"tokens_per_sec":row["tokens_per_sec"]} for row in rows]) for variant,rows in histories.items()}
    elapsed={variant:decimate([{"milestone_tokens":row["tokens_seen"],"gpu_hours":row["elapsed_sec"]/3600} for row in rows]) for variant,rows in histories.items()}
    svg_line_chart(args.output_root/"sampled_validation_ce_by_tokens.svg",sampled,"milestone_tokens","val_ce","Sampled validation CE by training tokens","Same fixed 16-batch validation sample at each evaluation; seed 1; lower is better.","Training tokens (log scale)","Sampled validation CE",fmt_tokens,True)
    svg_line_chart(args.output_root/"smoothed_train_ce_by_tokens.svg",smoothed,"milestone_tokens","train_ce","Smoothed training CE by training tokens","Mean logged training CE in 1M-token buckets; seed 1; lower is better.","Training tokens (log scale)","Training CE",fmt_tokens,True)
    svg_line_chart(args.output_root/"throughput_by_tokens.svg",throughput,"milestone_tokens","tokens_per_sec","Cumulative training throughput","RTX 4090 cumulative tokens per second; seed 1; higher is better.","Training tokens (log scale)","Tokens per second",fmt_tokens,True)
    svg_line_chart(args.output_root/"gpu_hours_by_tokens.svg",elapsed,"milestone_tokens","gpu_hours","Training time by processed tokens","RTX 4090 elapsed GPU hours; seed 1; lower is better.","Training tokens (log scale)","GPU hours",fmt_tokens,True)
    bar_chart(args.output_root/"frozen_validation_ce_at_100m.svg",final_rows,"validation_ce","Frozen validation CE at 100M tokens","Same continuous 4,834,787-token validation stream; seed 1; lower is better.","Cross-entropy",True)
    bar_chart(args.output_root/"frozen_validation_ppl_at_100m.svg",final_rows,"validation_ppl","Frozen validation perplexity at 100M tokens","Same continuous validation stream; seed 1; lower is better.","Perplexity",True)
    bar_chart(args.output_root/"gpu_hours_at_100m.svg",final_rows,"gpu_hours","Training time to 100M tokens","RTX 4090 elapsed hours; lower is better.","GPU hours",True)
    bar_chart(args.output_root/"throughput_at_100m.svg",final_rows,"tokens_per_sec","Training throughput at 100M tokens","Cumulative RTX 4090 tokens per second; higher is better.","Tokens per second",False)
    scatter_chart(args.output_root/"pareto_ce_vs_gpu_hours.svg",final_rows,"gpu_hours","validation_ce","Pareto: frozen CE versus GPU time","GPU hours to 100M tokens")
    scatter_chart(args.output_root/"pareto_ce_vs_parameters_m.svg",[{**row,"parameters_m":row["parameters"]/1e6} for row in final_rows],"parameters_m","validation_ce","Pareto: frozen CE versus parameter count","Parameters (millions)")
    with (args.output_root/"final_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
        fields=["variant","asm_name","parameters","validation_ce","validation_ppl","gpu_hours","tokens_per_sec"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in final_rows: writer.writerow({**row,"asm_name":LABELS[row["variant"]]})
    print(f"saved={args.output_root}")


if __name__ == "__main__":
    main()
