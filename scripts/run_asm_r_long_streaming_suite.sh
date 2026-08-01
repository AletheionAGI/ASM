#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo_root"
checkpoint="${CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt}"
output_root="${OUTPUT_ROOT:-runs/asm_r_long_streaming_32k}"
mkdir -p "$output_root"
.venv/bin/python scripts/benchmark_asm_r_long_streaming.py \
  --checkpoint "$checkpoint" --lengths 512,1024,2048,4096,8192,16384,32768 \
  --prompt-tokens 64 --mqar-steps 5000 --mqar-batches 8 --mqar-batch-size 1 \
  --device cuda --output "$output_root/results.json"
printf 'Suíte streaming concluída: %s/results.json\n' "$output_root"
