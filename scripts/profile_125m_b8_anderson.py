from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.training import count_parameters
from drm_language_emitter.utils import load_yaml_or_json


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device(requested)


def autocast_context(device: torch.device, precision: str):
    enabled = precision == "bf16" and device.type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def apply_drm_overrides(config: DRMConfig, args: argparse.Namespace) -> DRMConfig:
    config.max_seq_len = args.seq_len
    config.vocab_size = 256
    config.sequence_mode = args.sequence_mode
    config.directional_cumsum_block_size = args.drm_block_size
    config.directional_cumsum_step_mode = args.drm_cumsum_step_mode
    config.directional_anderson_iterations = args.drm_anderson_iterations
    config.directional_anderson_history_size = args.drm_anderson_history_size
    config.directional_anderson_transition_mode = args.drm_anderson_transition_mode
    config.directional_anderson_scope = args.drm_anderson_scope
    config.directional_anderson_block_stride = args.drm_anderson_block_stride
    config.directional_local_mixer = args.drm_local_mixer
    config.directional_local_mixer_hidden_size = args.drm_local_mixer_hidden_size
    config.directional_local_mixer_kernel_size = args.drm_local_mixer_kernel_size
    config.directional_local_mixer_layers = args.drm_local_mixer_layers
    config.directional_local_mixer_scale = args.drm_local_mixer_scale
    return config.validated_copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DRM 125M b8/Anderson forward+backward with torch.profiler.")
    parser.add_argument("--config", default="configs/drm_125m_real.yaml")
    parser.add_argument("--dataset-manifest", default="data/tokens_5b/manifest.json")
    parser.add_argument("--output-root", default="runs/profile_125m_b8_anderson")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--active-steps", type=int, default=4)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--sequence-mode",
        choices=["directional_block_cumsum", "directional_superblock_cumsum"],
        default="directional_block_cumsum",
    )
    parser.add_argument("--drm-block-size", type=int, default=8)
    parser.add_argument("--drm-anderson-iterations", type=int, default=2)
    parser.add_argument("--drm-anderson-history-size", type=int, default=4)
    parser.add_argument("--drm-anderson-transition-mode", choices=["candidate", "velocity"], default="candidate")
    parser.add_argument("--drm-anderson-scope", choices=["trajectory", "endpoint"], default="trajectory")
    parser.add_argument("--drm-anderson-block-stride", type=int, default=1)
    parser.add_argument("--drm-cumsum-step-mode", choices=["candidate", "velocity"], default="candidate")
    parser.add_argument("--drm-local-mixer", choices=["none", "causal_conv"], default="none")
    parser.add_argument("--drm-local-mixer-hidden-size", type=int, default=256)
    parser.add_argument("--drm-local-mixer-kernel-size", type=int, default=8)
    parser.add_argument("--drm-local-mixer-layers", type=int, default=1)
    parser.add_argument("--drm-local-mixer-scale", type=float, default=0.1)
    parser.add_argument("--sort-by", default="self_cuda_time_total")
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument("--export-trace", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config = DRMConfig.from_dict(load_yaml_or_json(args.config))
    config = apply_drm_overrides(config, args)
    model = DRMEmitterModel(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    dataset = MemmapTokenDataset(Path(args.dataset_manifest), split="train")
    generator = torch.Generator().manual_seed(args.seed)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    total_steps = args.warmup_steps + args.active_steps
    timings: list[dict[str, float]] = []
    schedule = torch.profiler.schedule(wait=0, warmup=args.warmup_steps, active=args.active_steps, repeat=1)

    def trace_handler(prof: torch.profiler.profile) -> None:
        if args.export_trace:
            prof.export_chrome_trace(str(output_root / "trace.json"))

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            x, y = dataset.make_batch(args.batch_size, args.seq_len, device, generator=generator)
            synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                out = model(x, y, global_step=step, collect_diagnostics=False)
            out["loss"].backward()
            optimizer.step()
            synchronize(device)
            elapsed = time.perf_counter() - start
            if step >= args.warmup_steps:
                timings.append({"step": step, "elapsed_sec": elapsed, "tokens_per_sec": args.batch_size * args.seq_len / elapsed})
            prof.step()

    sort_by = args.sort_by
    if device.type != "cuda" and sort_by.startswith("self_cuda"):
        sort_by = "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=args.row_limit)
    (output_root / "profile_table.txt").write_text(table + "\n", encoding="utf-8")

    summary = {
        "config": config.to_dict(),
        "parameter_count": count_parameters(model),
        "dataset_manifest": args.dataset_manifest,
        "device": str(device),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "warmup_steps": args.warmup_steps,
        "active_steps": args.active_steps,
        "timings": timings,
        "mean_tokens_per_sec": sum(row["tokens_per_sec"] for row in timings) / max(len(timings), 1),
        "mean_elapsed_sec": sum(row["elapsed_sec"] for row in timings) / max(len(timings), 1),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table)
    print(f"saved={output_root / 'profile_table.txt'}")
    print(f"saved={output_root / 'summary.json'}")
    if args.export_trace:
        print(f"saved={output_root / 'trace.json'}")


if __name__ == "__main__":
    main()
