#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-runs/asm_cm_e_suite}"
baseline_root="${BASELINE_ROOT:-runs/asm_c2_fw_lm_confirmation}"
device="${DEVICE:-cuda}"
curriculum="${CURRICULUM:-40:1000,80:500,160:500,320:300,512:200,1024:100,4096:25}"
mkdir -p "$output_root"

asm_r_checkpoint() {
  case "$1" in
    1) printf '%s\n' 'runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt' ;;
    2) printf '%s\n' 'runs/asm_confirmation_100m_seed2/variant_j_no_direction_seed_2/checkpoint_milestone_100000000.pt' ;;
    3) printf '%s\n' 'runs/asm_confirmation_100m_seed3/variant_j_no_direction_seed_3/checkpoint_milestone_100000000.pt' ;;
  esac
}

"$python_bin" -m pytest -q \
  tests/test_epistemic_softmax.py \
  tests/test_fast_weight_memory.py \
  tests/test_asm_c2_fw_lm.py

for seed in 1 2 3; do
  asm_r="$(asm_r_checkpoint "$seed")"
  baseline="$baseline_root/seed_${seed}/candidate/checkpoint_final.pt"
  seed_root="$output_root/seed_${seed}"
  candidate="$seed_root/training/checkpoint_final.pt"
  [[ -f "$asm_r" ]] || { printf 'ASM-R checkpoint missing: %s\n' "$asm_r" >&2; exit 2; }
  [[ -f "$baseline" ]] || { printf 'ASM-CM baseline missing: %s\n' "$baseline" >&2; exit 2; }

  if [[ ! -f "$candidate" ]]; then
    "$python_bin" scripts/train_asm_c2_fw_lm.py \
      --checkpoint "$asm_r" \
      --output-root "$seed_root/training" \
      --curriculum "$curriculum" \
      --language-probability "${LANGUAGE_PROBABILITY:-0.8}" \
      --language-seq-len "${LANGUAGE_SEQ_LEN:-128}" \
      --backbone-lr "${BACKBONE_LR:-1e-5}" \
      --memory-lr "${MEMORY_LR:-1e-4}" \
      --distillation-weight "${DISTILLATION_WEIGHT:-0.5}" \
      --batch-size "${BATCH_SIZE:-4}" \
      --eval-batches "${EVAL_BATCHES:-32}" \
      --seed "$seed" \
      --device "$device" \
      --epistemic-memory-gating
  fi

  if [[ ! -f "$seed_root/language.json" ]]; then
    "$python_bin" scripts/evaluate_frozen_test.py \
      --family drm \
      --checkpoint "$candidate" \
      --manifest "${VALIDATION_MANIFEST:-data/benchmark_125m_wikipedia/validation/manifest.json}" \
      --split validation \
      --seq-len 512 \
      --max-tokens "${LANGUAGE_EVAL_TOKENS:-10000000}" \
      --batch-size 4 \
      --device "$device" \
      --output "$seed_root/language.json"
  fi

  if [[ ! -f "$seed_root/long_32k.json" ]]; then
    "$python_bin" scripts/benchmark_asm_r_long_streaming.py \
      --checkpoint "$candidate" \
      --compact \
      --lengths 512,4096,32768 \
      --prompt-tokens 64 \
      --mqar-steps 0 \
      --mqar-control-length 40 \
      --mqar-control-threshold 0.8 \
      --mqar-batches "${LONG_EVAL_BATCHES:-128}" \
      --mqar-batch-size 4 \
      --device "$device" \
      --output "$seed_root/long_32k.json"
  fi
done

"$python_bin" scripts/summarize_asm_cm_e.py \
  --root "$output_root" \
  --baseline-root "$baseline_root"

printf 'ASM-CM-E suite complete: %s/report.md\n' "$output_root"
