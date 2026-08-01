#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo_root"
checkpoint="${CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt}"
output_root="${OUTPUT_ROOT:-runs/asm_c_streaming_32k}"
mkdir -p "$output_root"

.venv/bin/python -m pytest -q tests/test_generation.py tests/test_asm_c_streaming.py

.venv/bin/python scripts/check_asm_c_parity.py \
  --checkpoint "$checkpoint" --precision bf16 --device cuda \
  --output "$output_root/bf16_parity.json"

.venv/bin/python scripts/benchmark_asm_r_long_streaming.py \
  --checkpoint "$checkpoint" --compact \
  --lengths 512,1024,2048,4096,8192,16384,32768 \
  --prompt-tokens 64 --mqar-steps 5000 \
  --mqar-control-length 40 --mqar-control-threshold 0.8 \
  --mqar-batches 128 --mqar-batch-size 4 \
  --device cuda --output "$output_root/results.json"

.venv/bin/python scripts/compare_asm_r_asm_c_streaming.py \
  --asm-r runs/asm_r_long_streaming_32k/results.json \
  --asm-c "$output_root/results.json" \
  --output "$output_root/comparison.json" \
  --report "$output_root/report.md"

.venv/bin/python scripts/benchmark_asm_transformer_paired.py \
  --asm-checkpoint "$checkpoint" --asm-compact \
  --transformer-checkpoint runs/transformer_asm_r_matched_100m_seed1/checkpoint_milestone_100000000.pt \
  --manifest data/benchmark_125m_wikipedia/validation/manifest.json \
  --context-lengths 64,128,256,512,1024,2048 \
  --context-batches 8 --speed-lengths 64,128,256,512 \
  --speed-repeats 5 --decode-tokens 128 --generation-tokens 128 \
  --device cuda --output "$output_root/asm_c_transformer_paired.json"

printf 'ASM-C validation complete: %s/report.md\n' "$output_root"
