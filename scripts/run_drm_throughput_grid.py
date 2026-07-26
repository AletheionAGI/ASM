from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_GRID = [
    {"name": "baseline_g16_a8_n4_b2", "geometry": 16, "aux": 8, "naturalization": 4, "batch": 2, "accum": 1},
    {"name": "g32_a16_n8_b2", "geometry": 32, "aux": 16, "naturalization": 8, "batch": 2, "accum": 1},
    {"name": "g32_a16_n8_b4", "geometry": 32, "aux": 16, "naturalization": 8, "batch": 4, "accum": 1},
    {"name": "g64_a16_n8_b4", "geometry": 64, "aux": 16, "naturalization": 8, "batch": 4, "accum": 1},
    {"name": "g32_a16_n8_b2_compile", "geometry": 32, "aux": 16, "naturalization": 8, "batch": 2, "accum": 1, "compile": True},
    {"name": "g32_a16_n8_b2_no_metric_div", "geometry": 32, "aux": 16, "naturalization": 8, "batch": 2, "accum": 1, "metric_div": 0.0},
]


def load_latest(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics_latest.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8")).get("latest", {})


def run_case(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(args.output_root) / case["name"]
    cmd = [
        sys.executable,
        "scripts/train_drm_memmap.py",
        "--config",
        args.config,
        "--dataset-manifest",
        args.dataset_manifest,
        "--output-root",
        str(run_dir),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(case["batch"]),
        "--grad-accum-steps",
        str(case["accum"]),
        "--seq-len",
        str(args.seq_len),
        "--geometry-update-interval",
        str(case["geometry"]),
        "--aux-loss-interval",
        str(case["aux"]),
        "--naturalization-interval",
        str(case["naturalization"]),
        "--forward-chunk-size",
        str(args.forward_chunk_size),
        "--log-interval",
        str(args.log_interval),
        "--eval-tokens-interval",
        str(args.eval_tokens_interval),
        "--checkpoint-tokens-interval",
        str(args.checkpoint_tokens_interval),
        "--profile-steps",
        str(args.profile_steps),
        "--metrics-format",
        "jsonl",
        "--no-save-final-checkpoint",
    ]
    if args.pin_memory:
        cmd.append("--pin-memory")
    if args.prefetch_batches:
        cmd.extend(["--prefetch-batches", str(args.prefetch_batches)])
    if args.fused_adamw:
        cmd.append("--fused-adamw")
    if case.get("compile"):
        cmd.append("--compile-drm-step")
    if case.get("metric_div") is not None:
        cmd.extend(["--lambda-metric-diversity", str(case["metric_div"])])
    if case.get("metric_rank") is not None:
        cmd.extend(["--metric-rank", str(case["metric_rank"])])

    completed = subprocess.run(cmd, cwd=Path.cwd(), check=False)
    latest = load_latest(run_dir)
    row = {
        "name": case["name"],
        "returncode": completed.returncode,
        "geometry_update_interval": case["geometry"],
        "aux_loss_interval": case["aux"],
        "naturalization_interval": case["naturalization"],
        "batch_size": case["batch"],
        "grad_accum_steps": case["accum"],
        "compile_drm_step": bool(case.get("compile")),
        "lambda_metric_diversity": case.get("metric_div"),
        "metric_rank": case.get("metric_rank"),
        **latest,
    }
    return row


def write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps({"runs": rows}, indent=2), encoding="utf-8")
    keys = sorted({key for row in rows for key in row})
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short DRM throughput grid.")
    parser.add_argument("--config", default="configs/drm_125m_4090_throughput.yaml")
    parser.add_argument("--dataset-manifest", default="data/tokens_5b/manifest.json")
    parser.add_argument("--output-root", default="runs/drm_throughput_grid")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--forward-chunk-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-tokens-interval", type=int, default=1_000_000_000)
    parser.add_argument("--checkpoint-tokens-interval", type=int, default=1_000_000_000)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-batches", type=int, default=1)
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-metric-rank-grid", action="store_true")
    args = parser.parse_args()

    grid = list(DEFAULT_GRID)
    if args.include_metric_rank_grid:
        grid.extend(
            [
                {"name": "g32_a16_n8_b2_rank32", "geometry": 32, "aux": 16, "naturalization": 8, "batch": 2, "accum": 1, "metric_rank": 32},
                {"name": "g32_a16_n8_b2_rank16", "geometry": 32, "aux": 16, "naturalization": 8, "batch": 2, "accum": 1, "metric_rank": 16},
            ]
        )

    rows = []
    for case in grid:
        rows.append(run_case(args, case))
        write_outputs(Path(args.output_root), rows)

    write_outputs(Path(args.output_root), rows)
    print(f"saved={Path(args.output_root) / 'summary.json'}")
    print(f"saved={Path(args.output_root) / 'summary.csv'}")


if __name__ == "__main__":
    main()
