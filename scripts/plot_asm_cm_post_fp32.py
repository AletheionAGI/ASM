from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLORS = ["#06b6d4", "#8b5cf6", "#f59e0b"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def svg_line(path: Path, title: str, ylabel: str, labels: list[str], series: list[tuple[str, list[float]]]) -> None:
    width, height = 900, 520; left, right, top, bottom = 105, 35, 70, 85
    values = [value for _, row in series for value in row]; low, high = min(values), max(values)
    margin = max((high-low)*.15, abs(high)*.02, .001); low -= margin; high += margin
    xs = [left+i*(width-left-right)/(len(labels)-1) for i in range(len(labels))]
    def y(value: float) -> float: return top+(high-value)*(height-top-bottom)/(high-low)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#07111f"/>', f'<text x="{left}" y="38" fill="#f8fafc" font-family="sans-serif" font-size="24" font-weight="700">{title}</text>', f'<text transform="translate(28 {height/2}) rotate(-90)" text-anchor="middle" fill="#94a3b8" font-family="sans-serif" font-size="14">{ylabel}</text>']
    for index in range(5):
        value=low+(high-low)*index/4; yy=y(value)
        parts += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#334155" stroke-width="1"/>', f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" fill="#94a3b8" font-family="sans-serif" font-size="13">{value:.2f}</text>']
    for xx,label in zip(xs,labels): parts.append(f'<text x="{xx:.1f}" y="{height-45}" text-anchor="middle" fill="#cbd5e1" font-family="sans-serif" font-size="14">{label}</text>')
    for color,(name,row) in zip(COLORS,series):
        points=' '.join(f'{xx:.1f},{y(value):.1f}' for xx,value in zip(xs,row)); parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
        for xx,value in zip(xs,row): parts.append(f'<circle cx="{xx:.1f}" cy="{y(value):.1f}" r="6" fill="{color}"/>')
    for index,(color,(name,_)) in enumerate(zip(COLORS,series)):
        xx=left+index*130; parts += [f'<circle cx="{xx}" cy="{height-18}" r="6" fill="{color}"/>', f'<text x="{xx+12}" y="{height-13}" fill="#cbd5e1" font-family="sans-serif" font-size="13">{name}</text>']
    parts.append('</svg>'); path.write_text('\n'.join(parts)+'\n',encoding='utf-8')


def svg_bars(path: Path, rows: list[dict]) -> None:
    names=["ASM-CM","ASM-R","Transformer"]; fields=["candidate_ce","asm_r_ce","transformer_ce"]
    width,height=900,520; left,top,bottom=100,70,85; base=height-bottom; scale=(base-top)/1.5
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<rect width="100%" height="100%" fill="#07111f"/>',f'<text x="{left}" y="38" fill="#f8fafc" font-family="sans-serif" font-size="24" font-weight="700">Qualidade linguística por seed</text>']
    for seed_index,row in enumerate(rows):
        group_x=160+seed_index*250
        for model_index,(name,field,color) in enumerate(zip(names,fields,COLORS)):
            value=float(row[field]); bar_h=value*scale; x=group_x+model_index*55
            parts += [f'<rect x="{x}" y="{base-bar_h:.1f}" width="42" height="{bar_h:.1f}" rx="4" fill="{color}"/>',f'<text x="{x+21}" y="{base-bar_h-8:.1f}" text-anchor="middle" fill="#e2e8f0" font-family="sans-serif" font-size="11">{value:.3f}</text>']
        parts.append(f'<text x="{group_x+76}" y="{base+28}" text-anchor="middle" fill="#cbd5e1" font-family="sans-serif" font-size="14">Seed {row["seed"]}</text>')
    for index,(name,color) in enumerate(zip(names,COLORS)):
        x=180+index*190; parts += [f'<rect x="{x}" y="{height-28}" width="14" height="14" fill="{color}"/>',f'<text x="{x+22}" y="{height-16}" fill="#cbd5e1" font-family="sans-serif" font-size="13">{name}</text>']
    parts.append('</svg>'); path.write_text('\n'.join(parts)+'\n',encoding='utf-8')


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path('runs/asm_cm_post_fp32_validation')); parser.add_argument('--confirmation',type=Path,default=Path('runs/asm_c2_fw_lm_confirmation/decision.json')); parser.add_argument('--output-root',type=Path,default=Path('docs/benchmarks/asm_cm_post_fp32/charts')); args=parser.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    streams=[load(args.root/f'seed_{seed}'/'streaming.json')['streaming'] for seed in (1,2,3)]; labels=['512','4K','32K']
    csv_rows=[]
    for seed,rows in zip((1,2,3),streams):
        for row in rows: csv_rows.append({'seed':seed,**{key:row[key] for key in ('sequence_length','cache_tensor_bytes','cache_mebibytes','cuda_peak_mb','segment_tokens_per_sec')}})
    with (args.output_root/'streaming_metrics.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=csv_rows[0].keys(),lineterminator='\n'); writer.writeheader(); writer.writerows(csv_rows)
    charts={'cache_vs_context':('Estado tensorial retido permanece constante','Cache retido (MiB)','cache_mebibytes'),'vram_vs_context':('Pico de VRAM permanece constante','Pico de VRAM alocada (MiB)','cuda_peak_mb'),'throughput_vs_context':('Throughput não degrada até 32K','Decode (tokens/s)','segment_tokens_per_sec')}
    for filename,(title,ylabel,field) in charts.items(): svg_line(args.output_root/f'{filename}.svg',title,ylabel,labels,[(f'Seed {seed}',[float(row[field]) for row in rows]) for seed,rows in zip((1,2,3),streams)])
    svg_bars(args.output_root/'language_ce_by_seed.svg',load(args.confirmation)['seeds'])


if __name__=='__main__': main()
