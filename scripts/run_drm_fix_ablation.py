from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.utils import load_yaml_or_json, save_json


def resolve_variant(matrix: dict[str, Any], name: str) -> tuple[DRMConfig, str]:
    variants = matrix["variants"]
    if name not in variants:
        raise ValueError(f"unknown variant {name!r}; choose from {', '.join(variants)}")
    definition = variants[name]
    data = load_yaml_or_json(matrix["base_config"])
    data.update(matrix.get("common", {}))
    for inherited in definition.get("inherit", []):
        data.update(matrix[inherited])
    data.update(definition.get("overrides", {}))
    return DRMConfig.from_dict(data), str(definition["description"])


def parse_variants(raw: str, available: dict[str, Any]) -> list[str]:
    names = list(available) if raw.lower() == "all" else [item.strip().upper() for item in raw.split(",")]
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown variant(s): {', '.join(unknown)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DRM A-I CE ablation matrix.")
    parser.add_argument("--matrix", type=Path, default=Path("configs/drm_fix_ablation_variants.json"))
    parser.add_argument("--variants", default="all", help="Comma-separated A-I names, or 'all'.")
    parser.add_argument("--train-manifest", default="data/benchmark_125m_wikipedia/train/manifest.json")
    parser.add_argument(
        "--validation-manifest",
        default="data/benchmark_125m_wikipedia/validation/manifest.json",
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/drm_fix_ablation"))
    parser.add_argument("--target-tokens", type=int, default=30_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--eval-tokens-interval", type=int, default=1_000_000)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--checkpoint-token-milestones", default="")
    parser.add_argument(
        "--save-best-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-forward", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    names = parse_variants(args.variants, matrix["variants"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"matrix": str(args.matrix), "runs": []}

    for name in names:
        config, description = resolve_variant(matrix, name)
        run_dir = args.output_root / f"variant_{name.lower()}_seed_{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "resolved_config.json"
        save_json(config_path, config.to_dict())
        model = DRMEmitterModel(config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        del model
        command = [
            sys.executable,
            "scripts/train_drm_memmap.py",
            "--config",
            str(config_path),
            "--train-manifest",
            args.train_manifest,
            "--validation-manifest",
            args.validation_manifest,
            "--output-root",
            str(run_dir),
            "--target-tokens",
            str(args.target_tokens),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
            "--seq-len",
            str(args.seq_len),
            "--lr",
            "3e-4",
            "--weight-decay",
            "0.01",
            "--max-grad-norm",
            "1.0",
            "--precision",
            args.precision,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--eval-tokens-interval",
            str(args.eval_tokens_interval),
            "--checkpoint-tokens-interval",
            str(args.target_tokens),
            "--eval-batches",
            str(args.eval_batches),
            "--log-interval",
            "10",
        ]
        if args.save_best_checkpoint:
            command.append("--save-best-checkpoint")
        if args.checkpoint_token_milestones:
            command.extend(
                ["--checkpoint-token-milestones", args.checkpoint_token_milestones]
            )
        if args.dry_run:
            command.append("--dry-run")
        if args.dry_run_forward:
            command.append("--dry-run-forward")
        summary_path = run_dir / "summary.json"
        complete = False
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            complete = int(summary.get("tokens_seen", 0)) >= args.target_tokens
        if not complete and not args.dry_run:
            latest = run_dir / "checkpoint_latest.pt"
            best = run_dir / "checkpoint_best.pt"
            if latest.is_file():
                command.extend(["--resume", "latest"])
            elif best.is_file():
                command.extend(["--resume", str(best)])
        manifest["runs"].append(
            {
                "variant": name,
                "description": description,
                "parameter_count": parameter_count,
                "config": str(config_path),
                "output": str(run_dir),
                "command": command,
            }
        )
        save_json(args.output_root / "ablation_manifest.json", manifest)
        print(
            f"\n=== Variant {name}: {description} ({parameter_count} parameters) ===",
            flush=True,
        )
        if args.plan_only:
            continue
        if complete:
            print(f"Already complete: {run_dir}", flush=True)
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
