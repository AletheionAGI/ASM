#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

asm_c_checkpoint="${ASM_C_CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt}"
asm_s_checkpoint="${ASM_S_CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_direct_control_matched_seed_1/checkpoint_milestone_100000000.pt}"
transformer_checkpoint="${TRANSFORMER_CHECKPOINT:-runs/transformer_asm_r_matched_100m_seed1/checkpoint_milestone_100000000.pt}"
output_root="${OUTPUT_ROOT:-runs/asm_c_mqar_diagnostic_20k}"
device="${DEVICE:-cuda}"
variants="${VARIANTS:-ASM_C_PRETRAINED,ASM_C_MEMORY_2X,ASM_S_PRETRAINED,TRANSFORMER_PRETRAINED}"

for checkpoint in "$asm_c_checkpoint" "$asm_s_checkpoint" "$transformer_checkpoint"; do
  if [[ ! -f "$checkpoint" ]]; then
    printf 'Checkpoint not found: %s\n' "$checkpoint" >&2
    exit 2
  fi
done

.venv/bin/python scripts/run_mqar_architecture_comparison.py \
  --asm-r-checkpoint "$asm_c_checkpoint" \
  --asm-s-checkpoint "$asm_s_checkpoint" \
  --transformer-checkpoint "$transformer_checkpoint" \
  --asm-r-config configs/asm_r_125m.json \
  --variants "$variants" \
  --milestones 5000,10000,20000 \
  --batch-size 4 \
  --eval-batches 128 \
  --n-pairs 8 \
  --n-queries 8 \
  --device "$device" \
  --precision bf16 \
  --output-root "$output_root"

.venv/bin/python scripts/plot_mqar_architecture_comparison.py \
  --results "$output_root/results.json" \
  --output-root "$output_root/charts"

printf 'MQAR diagnostic complete: %s/report.md\n' "$output_root"
