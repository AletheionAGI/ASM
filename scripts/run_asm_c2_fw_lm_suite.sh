#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

checkpoint="${ASM_R_CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt}"
output_root="${OUTPUT_ROOT:-runs/asm_c2_fw_lm_suite}"
device="${DEVICE:-cuda}"
curriculum="${CURRICULUM:-40:1000,80:500,160:500,320:300,512:200,1024:100,4096:25}"
language_probability="${LANGUAGE_PROBABILITY:-0.8}"
mkdir -p "$output_root"
[[ -f "$checkpoint" ]] || { printf 'Checkpoint not found: %s\n' "$checkpoint" >&2; exit 2; }

.venv/bin/python -m pytest -q \
  tests/test_fast_weight_memory.py \
  tests/test_asm_c2_streaming.py \
  tests/test_asm_c2_fw_lm.py

seed_results=()
winner_checkpoint=""
passed_count=0
for seed in 1 2 3; do
  seed_root="$output_root/seed_${seed}"
  .venv/bin/python scripts/train_asm_c2_fw_lm.py \
    --checkpoint "$checkpoint" \
    --output-root "$seed_root" \
    --curriculum "$curriculum" \
    --language-probability "$language_probability" \
    --language-seq-len "${LANGUAGE_SEQ_LEN:-128}" \
    --backbone-lr "${BACKBONE_LR:-1e-5}" \
    --memory-lr "${MEMORY_LR:-1e-4}" \
    --distillation-weight "${DISTILLATION_WEIGHT:-0.5}" \
    --distillation-temperature "${DISTILLATION_TEMPERATURE:-2.0}" \
    --batch-size "${BATCH_SIZE:-4}" \
    --eval-batches "${EVAL_BATCHES:-32}" \
    --seed "$seed" \
    --device "$device"
  seed_results+=("$seed_root/results.json")
  passed="$(.venv/bin/python -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["passed"]).lower())' "$seed_root/results.json")"
  if [[ "$passed" == "true" ]]; then
    passed_count=$((passed_count + 1))
    [[ -n "$winner_checkpoint" ]] || winner_checkpoint="$seed_root/checkpoint_final.pt"
  fi
done

if (( passed_count < 2 )); then
  .venv/bin/python scripts/summarize_asm_c2_fw_durable.py \
    --seed-results "${seed_results[@]}" \
    --variant ASM-C2-FW-LM \
    --output "$output_root/decision.json" \
    --report "$output_root/report.md"
  printf '%s\n' 'ASM-C2-FW-LM failed the 2-of-3 MQAR curriculum gate; long evaluation blocked.'
  exit 0
fi

.venv/bin/python scripts/check_asm_c_parity.py \
  --checkpoint "$winner_checkpoint" \
  --precision bf16 \
  --device "$device" \
  --output "$output_root/bf16_parity.json"

long_root="$output_root/long_32k"
mkdir -p "$long_root"
.venv/bin/python scripts/benchmark_asm_r_long_streaming.py \
  --checkpoint "$winner_checkpoint" \
  --compact \
  --lengths 512,4096,32768 \
  --prompt-tokens 64 \
  --mqar-steps 0 \
  --mqar-control-length 40 \
  --mqar-control-threshold 0.8 \
  --mqar-batches "${LONG_EVAL_BATCHES:-128}" \
  --mqar-batch-size 4 \
  --device "$device" \
  --output "$long_root/results.json"

language_root="$output_root/language_regression"
mkdir -p "$language_root"
.venv/bin/python scripts/evaluate_frozen_test.py \
  --family drm \
  --checkpoint "$checkpoint" \
  --manifest data/benchmark_125m_wikipedia/validation/manifest.json \
  --split validation \
  --seq-len 512 \
  --max-tokens "${LANGUAGE_EVAL_TOKENS:-1000000}" \
  --batch-size 4 \
  --device "$device" \
  --output "$language_root/baseline.json"
.venv/bin/python scripts/evaluate_frozen_test.py \
  --family drm \
  --checkpoint "$winner_checkpoint" \
  --manifest data/benchmark_125m_wikipedia/validation/manifest.json \
  --split validation \
  --seq-len 512 \
  --max-tokens "${LANGUAGE_EVAL_TOKENS:-1000000}" \
  --batch-size 4 \
  --device "$device" \
  --output "$language_root/candidate.json"

.venv/bin/python scripts/summarize_asm_c2_fw_durable.py \
  --seed-results "${seed_results[@]}" \
  --long-results "$long_root/results.json" \
  --language-baseline "$language_root/baseline.json" \
  --language-candidate "$language_root/candidate.json" \
  --parity "$output_root/bf16_parity.json" \
  --variant ASM-C2-FW-LM \
  --output "$output_root/decision.json" \
  --report "$output_root/report.md"

.venv/bin/python scripts/plot_asm_c2_results.py \
  --short-results runs/asm_c2_fw_suite/short_control/results.json \
  --long-results "$long_root/results.json" \
  --output-root "$output_root/charts"

printf 'ASM-C2-FW-LM suite complete: %s/report.md\n' "$output_root"
