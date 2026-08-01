#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

checkpoint="${CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt}"
output="${OUTPUT:-runs/asm_r_mqar_curve_5k/results.json}"
device="${DEVICE:-cuda}"

if [[ ! -f "$checkpoint" ]]; then
  printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' ".venv/bin/python not found" >&2
  exit 2
fi

.venv/bin/python scripts/evaluate_asm_r_mqar_curve.py \
  --checkpoint "$checkpoint" \
  --output "$output" \
  --milestones 200,500,1000,2000,5000 \
  --batch-size 4 \
  --eval-batches 64 \
  --n-pairs 8 \
  --n-queries 8 \
  --device "$device" \
  --precision bf16
