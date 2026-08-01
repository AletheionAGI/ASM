#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

checkpoint=""
manifest="data/benchmark_125m_wikipedia/validation/manifest.json"
output_root="runs/asm_r_post_promotion"
device="cuda"
mode="quick"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --manifest) manifest="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --quick) mode="quick"; shift ;;
    --full) mode="full"; shift ;;
    -h|--help)
      printf '%s\n' \
        "Usage: $0 --checkpoint PATH [--manifest PATH] [--output-root PATH]" \
        "          [--device cuda|cpu] [--quick|--full]"
      exit 0
      ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$checkpoint" ]]; then
  printf '%s\n' "--checkpoint is required" >&2
  exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
  printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
  exit 2
fi
if [[ ! -f "$manifest" ]]; then
  printf 'Manifest not found: %s\n' "$manifest" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' ".venv/bin/python not found; create the project virtual environment first" >&2
  exit 2
fi

mkdir -p "$output_root"

if [[ "$mode" == "full" ]]; then
  validation_tokens=4834787
  context_lengths="64,128,256,512,1024,2048"
  context_batches=16
  generation_tokens=128
  mqar_steps=200
  mqar_batch_size=4
  causality_seq_len=512
  decode_prompt_tokens=128
  decode_tokens=64
else
  validation_tokens=262144
  context_lengths="64,128,512,1024"
  context_batches=2
  generation_tokens=32
  mqar_steps=10
  mqar_batch_size=2
  causality_seq_len=128
  decode_prompt_tokens=64
  decode_tokens=16
fi

printf 'ASM-R post-promotion suite: mode=%s device=%s\n' "$mode" "$device"
printf 'checkpoint=%s\n' "$checkpoint"
printf 'output_root=%s\n' "$output_root"

.venv/bin/python -m pytest -q tests/test_generation.py tests/test_checkpoint_security.py

.venv/bin/python scripts/check_125m_local_mixer_causality.py \
  --checkpoint "$checkpoint" \
  --seq-len "$causality_seq_len" \
  --batch-size 1 \
  --device "$device" \
  --precision bf16 \
  --output "$output_root/causality.json"

.venv/bin/python scripts/evaluate_frozen_test.py \
  --family drm \
  --checkpoint "$checkpoint" \
  --manifest "$manifest" \
  --split validation \
  --seq-len 512 \
  --max-tokens "$validation_tokens" \
  --batch-size 4 \
  --device "$device" \
  --output "$output_root/validation.json"

.venv/bin/python scripts/evaluate_asm_r_checkpoint.py \
  --checkpoint "$checkpoint" \
  --manifest "$manifest" \
  --device "$device" \
  --precision bf16 \
  --context-lengths "$context_lengths" \
  --context-batches "$context_batches" \
  --generation-tokens "$generation_tokens" \
  --mqar-steps "$mqar_steps" \
  --mqar-batch-size "$mqar_batch_size" \
  --output "$output_root/checkpoint_evaluation.json"

.venv/bin/python scripts/benchmark_incremental_decode.py \
  --checkpoint "$checkpoint" \
  --prompt-tokens "$decode_prompt_tokens" \
  --decode-tokens "$decode_tokens" \
  --batch-size 1 \
  --device "$device" \
  --precision bf16 \
  --output "$output_root/incremental_decode.json"

.venv/bin/python scripts/summarize_asm_r_post_promotion.py --root "$output_root"

printf 'Suite complete: %s/report.md\n' "$output_root"
