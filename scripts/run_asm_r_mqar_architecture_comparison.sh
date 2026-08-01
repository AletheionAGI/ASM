#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

asm_r_checkpoint="${ASM_R_CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt}"
asm_s_checkpoint="${ASM_S_CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_direct_control_matched_seed_1/checkpoint_milestone_100000000.pt}"
transformer_checkpoint="${TRANSFORMER_CHECKPOINT:-runs/transformer_asm_r_matched_100m_seed1/checkpoint_last.pt}"
output_root="${OUTPUT_ROOT:-runs/asm_r_mqar_architecture_comparison_20k}"
device="${DEVICE:-cuda}"
variants="${VARIANTS:-ASM_R_PRETRAINED,ASM_R_RANDOM,ASM_R_NO_MEMORY,ASM_S_PRETRAINED,TRANSFORMER_PRETRAINED,TRANSFORMER_RANDOM}"

for checkpoint in "$asm_r_checkpoint" "$asm_s_checkpoint" "$transformer_checkpoint"; do
  if [[ ! -f "$checkpoint" ]]; then
    printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
    exit 2
  fi
done
if [[ ! -x .venv/bin/python ]]; then
  printf '%s\n' ".venv/bin/python not found" >&2
  exit 2
fi

.venv/bin/python scripts/run_mqar_architecture_comparison.py \
  --asm-r-checkpoint "$asm_r_checkpoint" \
  --asm-s-checkpoint "$asm_s_checkpoint" \
  --transformer-checkpoint "$transformer_checkpoint" \
  --asm-r-config configs/asm_r_125m.json \
  --variants "$variants" \
  --milestones 200,500,1000,2000,5000,10000,20000 \
  --batch-size 4 \
  --eval-batches 64 \
  --n-pairs 8 \
  --n-queries 8 \
  --device "$device" \
  --precision bf16 \
  --output-root "$output_root"
