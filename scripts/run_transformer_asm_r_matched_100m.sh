#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-runs/transformer_asm_r_matched_100m_seed1}"
seed="${SEED:-1}"
target_tokens="${TARGET_TOKENS:-100000000}"
milestones="${MILESTONES:-1000000,2000000,5000000,10000000,20000000,30000000,50000000,100000000}"
train_manifest="${TRAIN_MANIFEST:-data/benchmark_125m_wikipedia/train/manifest.json}"
validation_manifest="${VALIDATION_MANIFEST:-data/benchmark_125m_wikipedia/validation/manifest.json}"

if [[ ! -x "$python_bin" ]]; then
  printf 'Python environment not found: %s\n' "$python_bin" >&2
  exit 2
fi
for manifest in "$train_manifest" "$validation_manifest"; do
  if [[ ! -f "$manifest" ]]; then
    printf 'Manifest not found: %s\n' "$manifest" >&2
    exit 2
  fi
done

mkdir -p "$output_root"
resume_args=()
if [[ -f "$output_root/checkpoint_latest.pt" ]]; then
  resume_args=(--resume latest)
fi

"$python_bin" scripts/train_gpt2_memmap.py \
  --model-size gpt2_asm_r_matched \
  --train-manifest "$train_manifest" \
  --validation-manifest "$validation_manifest" \
  --output-root "$output_root" \
  --target-tokens "$target_tokens" \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --seq-len 512 \
  --lr 3e-4 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --dropout 0.0 \
  --precision bf16 \
  --device cuda \
  --seed "$seed" \
  --eval-tokens-interval 1000000 \
  --checkpoint-token-milestones "$milestones" \
  --checkpoint-tokens-interval "$target_tokens" \
  --eval-batches 16 \
  --log-interval 10 \
  "${resume_args[@]}"

final_checkpoint="$output_root/checkpoint_last.pt"
"$python_bin" scripts/evaluate_frozen_test.py \
  --family gpt2 \
  --checkpoint "$final_checkpoint" \
  --manifest "$validation_manifest" \
  --split validation \
  --seq-len 512 \
  --max-tokens 10000000 \
  --batch-size 8 \
  --device cuda \
  --output "$output_root/validation_full.json"

printf 'Transformer pareado concluído: %s\n' "$final_checkpoint"
printf 'Validação congelada: %s\n' "$output_root/validation_full.json"
