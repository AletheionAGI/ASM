#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-runs/asm_c2_fw_lm_confirmation}"
device="${DEVICE:-cuda}"
validation_manifest="${VALIDATION_MANIFEST:-data/benchmark_125m_wikipedia/validation/manifest.json}"
language_eval_tokens="${LANGUAGE_EVAL_TOKENS:-10000000}"
curriculum="${CURRICULUM:-40:1000,80:500,160:500,320:300,512:200,1024:100,4096:25}"

[[ -x "$python_bin" ]] || { printf 'Python not found: %s\n' "$python_bin" >&2; exit 2; }
[[ -f "$validation_manifest" ]] || { printf 'Manifest not found: %s\n' "$validation_manifest" >&2; exit 2; }
mkdir -p "$output_root"

asm_checkpoint() {
  case "$1" in
    1) printf '%s\n' 'runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt' ;;
    2) printf '%s\n' 'runs/asm_confirmation_100m_seed2/variant_j_no_direction_seed_2/checkpoint_milestone_100000000.pt' ;;
    3) printf '%s\n' 'runs/asm_confirmation_100m_seed3/variant_j_no_direction_seed_3/checkpoint_milestone_100000000.pt' ;;
  esac
}

transformer_root() {
  case "$1" in
    1) printf '%s\n' 'runs/transformer_asm_r_matched_100m_seed1' ;;
    *) printf '%s/transformer_seed_%s\n' "$output_root" "$1" ;;
  esac
}

"$python_bin" -m pytest -q \
  tests/test_fast_weight_memory.py \
  tests/test_asm_c2_streaming.py \
  tests/test_asm_c2_fw_lm.py \
  tests/test_asm_c2_fw_lm_confirmation.py

for seed in 1 2 3; do
  asm_r="$(asm_checkpoint "$seed")"
  [[ -f "$asm_r" ]] || { printf 'ASM-R seed %s not found: %s\n' "$seed" "$asm_r" >&2; exit 2; }
  tf_root="$(transformer_root "$seed")"
  transformer="$tf_root/checkpoint_last.pt"
  if [[ ! -f "$transformer" ]]; then
    printf 'Training matched Transformer seed=%s to 100M tokens...\n' "$seed"
    SEED="$seed" OUTPUT_ROOT="$tf_root" TARGET_TOKENS=100000000 \
      "$repo_root/scripts/run_transformer_asm_r_matched_100m.sh"
  fi

  seed_root="$output_root/seed_${seed}"
  candidate_root="$seed_root/candidate"
  candidate="$candidate_root/checkpoint_final.pt"
  if [[ ! -f "$candidate" ]]; then
    "$python_bin" scripts/train_asm_c2_fw_lm.py \
      --checkpoint "$asm_r" \
      --output-root "$candidate_root" \
      --curriculum "$curriculum" \
      --language-probability "${LANGUAGE_PROBABILITY:-0.8}" \
      --language-seq-len "${LANGUAGE_SEQ_LEN:-128}" \
      --backbone-lr "${BACKBONE_LR:-1e-5}" \
      --memory-lr "${MEMORY_LR:-1e-4}" \
      --distillation-weight "${DISTILLATION_WEIGHT:-0.5}" \
      --distillation-temperature "${DISTILLATION_TEMPERATURE:-2.0}" \
      --batch-size "${BATCH_SIZE:-4}" \
      --eval-batches "${EVAL_BATCHES:-32}" \
      --seed "$seed" \
      --device "$device"
  fi

  mkdir -p "$seed_root/language"
  for specification in \
    "drm|$candidate|asm_c2_fw_lm" \
    "drm|$asm_r|asm_r" \
    "gpt2|$transformer|transformer"
  do
    IFS='|' read -r family checkpoint label <<< "$specification"
    result="$seed_root/language/$label.json"
    if [[ "$label" == "transformer" && ! -f "$result" && -f "$tf_root/validation_full.json" ]]; then
      cp "$tf_root/validation_full.json" "$result"
    fi
    if [[ ! -f "$result" ]]; then
      "$python_bin" scripts/evaluate_frozen_test.py \
        --family "$family" \
        --checkpoint "$checkpoint" \
        --manifest "$validation_manifest" \
        --split validation \
        --seq-len 512 \
        --max-tokens "$language_eval_tokens" \
        --batch-size 4 \
        --device "$device" \
        --output "$result"
    fi
  done

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
  if [[ ! -f "$seed_root/bf16_parity.json" ]]; then
    "$python_bin" scripts/check_asm_c_parity.py \
      --checkpoint "$candidate" \
      --precision bf16 \
      --device "$device" \
      --output "$seed_root/bf16_parity.json"
  fi

  if [[ ! -f "$seed_root/paired_inference.json" ]]; then
    "$python_bin" scripts/benchmark_asm_transformer_paired.py \
      --asm-checkpoint "$candidate" \
      --asm-compact \
      --transformer-checkpoint "$transformer" \
      --manifest "$validation_manifest" \
      --split validation \
      --context-lengths 64,128,256,512,1024,2048,4096 \
      --context-batches 8 \
      --speed-lengths 64,128,256,512 \
      --speed-repeats 5 \
      --decode-tokens 128 \
      --skip-generation \
      --device "$device" \
      --output "$seed_root/paired_inference.json"
  fi
done

"$python_bin" scripts/summarize_asm_c2_fw_lm_confirmation.py \
  --root "$output_root" \
  --output "$output_root/decision.json" \
  --report "$output_root/report.md"

"$python_bin" scripts/plot_asm_c2_fw_lm_confirmation.py \
  --decision "$output_root/decision.json" \
  --output-root "$output_root/charts"

printf 'Independent confirmation complete: %s/report.md\n' "$output_root"
printf 'Official promotion decision: %s/decision.json\n' "$output_root"
