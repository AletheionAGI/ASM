from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.utils import load_yaml_or_json


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def autocast_context(device: torch.device, precision: str):
    enabled = precision == "bf16" and device.type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def resolve_checkpoint(run_dir: Path | None, checkpoint: str) -> Path | None:
    if checkpoint:
        path = Path(checkpoint)
        if path.name == "latest" and run_dir is not None:
            latest = run_dir / "checkpoint_latest.pt"
            return latest if latest.exists() else None
        return path
    if run_dir is None:
        return None
    for name in ("checkpoint_last.pt", "checkpoint_latest.pt", "best_model.pt"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def build_config(args: argparse.Namespace, checkpoint_payload: dict[str, Any] | None) -> DRMConfig:
    if checkpoint_payload is not None and isinstance(checkpoint_payload.get("config"), dict):
        config = DRMConfig.from_dict(checkpoint_payload["config"])
    else:
        config = DRMConfig.from_dict(load_yaml_or_json(args.config))

    config.sequence_mode = args.sequence_mode
    config.max_seq_len = max(config.max_seq_len, args.seq_len)
    config.directional_cumsum_block_size = args.drm_block_size
    config.directional_cumsum_step_mode = args.drm_cumsum_step_mode
    config.directional_candidate_scale = args.drm_candidate_scale
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
    config._validate()
    return config


def prefix_lengths(seq_len: int, requested: list[int]) -> list[int]:
    values = requested or [1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, seq_len - 1]
    return sorted({value for value in values if 1 <= value < seq_len})


@torch.no_grad()
def compare_prefixes(
    model: DRMEmitterModel,
    input_ids: torch.Tensor,
    prefix_len: int,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    changed = input_ids.clone()
    suffix = changed[:, prefix_len:]
    changed[:, prefix_len:] = (suffix + 127) % model.config.vocab_size

    with autocast_context(device, precision):
        base = model(input_ids, return_states=True, collect_diagnostics=False)
        mutated = model(changed, return_states=True, collect_diagnostics=False)

    logits_diff = (base["logits"][:, :prefix_len] - mutated["logits"][:, :prefix_len]).abs().max()
    states_diff = (base["states"][:, :prefix_len] - mutated["states"][:, :prefix_len]).abs().max()
    return {
        "prefix_len": float(prefix_len),
        "max_logit_abs_diff": float(logits_diff.detach().cpu()),
        "max_state_abs_diff": float(states_diff.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check prefix causality for the 125M DRM local mixer path.")
    parser.add_argument("--config", default="configs/drm_125m_real.yaml")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--prefix-len", type=int, action="append", default=[])
    parser.add_argument("--sequence-mode", default="directional_block_cumsum")
    parser.add_argument("--drm-block-size", type=int, default=64)
    parser.add_argument("--drm-cumsum-step-mode", choices=["candidate", "velocity"], default="velocity")
    parser.add_argument("--drm-candidate-scale", type=float, default=0.01)
    parser.add_argument("--drm-anderson-iterations", type=int, default=0)
    parser.add_argument("--drm-anderson-history-size", type=int, default=4)
    parser.add_argument("--drm-anderson-transition-mode", choices=["candidate", "velocity"], default="candidate")
    parser.add_argument("--drm-anderson-scope", choices=["trajectory", "endpoint"], default="trajectory")
    parser.add_argument("--drm-anderson-block-stride", type=int, default=1)
    parser.add_argument("--drm-local-mixer", choices=["none", "causal_conv"], default="causal_conv")
    parser.add_argument("--drm-local-mixer-hidden-size", type=int, default=256)
    parser.add_argument("--drm-local-mixer-kernel-size", type=int, default=8)
    parser.add_argument("--drm-local-mixer-layers", type=int, default=2)
    parser.add_argument("--drm-local-mixer-scale", type=float, default=0.2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    checkpoint_path = resolve_checkpoint(run_dir, args.checkpoint)
    checkpoint_payload = None
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = build_config(args, checkpoint_payload)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    model = DRMEmitterModel(config).to(device)
    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model"])
    model.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device, generator=generator)

    rows = []
    max_logit = 0.0
    max_state = 0.0
    for length in prefix_lengths(args.seq_len, args.prefix_len):
        row = compare_prefixes(model, input_ids, length, device, args.precision)
        rows.append(row)
        max_logit = max(max_logit, row["max_logit_abs_diff"])
        max_state = max(max_state, row["max_state_abs_diff"])

    passed = bool(max_logit <= args.atol + args.rtol * max(1.0, max_logit) and max_state <= args.atol + args.rtol * max(1.0, max_state))
    result = {
        "passed": passed,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "device": str(device),
        "max_logit_abs_diff": max_logit,
        "max_state_abs_diff": max_state,
        "rows": rows,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if not passed or not math.isfinite(max_logit) or not math.isfinite(max_state):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
