#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

asm_checkpoint="${ASM_C_CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt}"
asm_s_checkpoint="${ASM_S_CHECKPOINT:-runs/asm_scaling_law_100m_seed1/variant_j_direct_control_matched_seed_1/checkpoint_milestone_100000000.pt}"
transformer_checkpoint="${TRANSFORMER_CHECKPOINT:-runs/transformer_asm_r_matched_100m_seed1/checkpoint_milestone_100000000.pt}"
output_root="${OUTPUT_ROOT:-runs/asm_c2_fw_suite}"
device="${DEVICE:-cuda}"
steps="${STEPS:-500,1000,2000,5000}"
eval_batches="${EVAL_BATCHES:-128}"
mkdir -p "$output_root"

for checkpoint in "$asm_checkpoint" "$asm_s_checkpoint" "$transformer_checkpoint"; do
  [[ -f "$checkpoint" ]] || { printf 'Checkpoint not found: %s\n' "$checkpoint" >&2; exit 2; }
done

.venv/bin/python -m pytest -q tests/test_fast_weight_memory.py tests/test_asm_c2_streaming.py

short_root="$output_root/short_control"
.venv/bin/python scripts/run_mqar_architecture_comparison.py \
  --asm-r-checkpoint "$asm_checkpoint" \
  --asm-s-checkpoint "$asm_s_checkpoint" \
  --transformer-checkpoint "$transformer_checkpoint" \
  --asm-r-config configs/asm_r_125m.json \
  --variants ASM_C_PRETRAINED,ASM_C2_FW,TRANSFORMER_PRETRAINED \
  --milestones "$steps" --batch-size 4 --eval-batches "$eval_batches" \
  --n-pairs 8 --n-queries 8 --device "$device" --precision bf16 \
  --save-final-checkpoints --output-root "$short_root"

.venv/bin/python scripts/compare_asm_c2_controls.py \
  --short-results "$short_root/results.json" \
  --output "$output_root/decision_short.json" \
  --report "$output_root/decision_short.md"

winner="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"] or "")' "$output_root/decision_short.json")"
if [[ "$winner" != "ASM_C2_FW" ]]; then
  printf '%s\n' 'ASM-C2-FW failed the 80% short gate; causal and long evaluations blocked.'
  exit 0
fi

ablation_root="$output_root/ablations"
.venv/bin/python scripts/run_mqar_architecture_comparison.py \
  --asm-r-checkpoint "$asm_checkpoint" \
  --asm-s-checkpoint "$asm_s_checkpoint" \
  --transformer-checkpoint "$transformer_checkpoint" \
  --asm-r-config configs/asm_r_125m.json \
  --variants ASM_C2_FW,ASM_C2_FW_NOREAD,ASM_C2_FW_NOWRITE,ASM_C2_FW_SHUFFLED \
  --milestones "$steps" --batch-size 4 --eval-batches "$eval_batches" \
  --n-pairs 8 --n-queries 8 --device "$device" --precision bf16 \
  --output-root "$ablation_root"

.venv/bin/python scripts/compare_asm_c2_controls.py \
  --short-results "$short_root/results.json" --ablations "$ablation_root/results.json" \
  --output "$output_root/decision_ablations.json" --report "$output_root/decision_ablations.md"

passed="$(.venv/bin/python -c 'import json,sys; print(str(all(json.load(open(sys.argv[1]))["criteria"].values())).lower())' "$output_root/decision_ablations.json")"
if [[ "$passed" != "true" ]]; then
  printf '%s\n' 'ASM-C2-FW failed causal memory ablations; long evaluation blocked.'
  exit 0
fi

winner_checkpoint="$short_root/asm_c2_fw_final.pt"
confirmation_files=()
for confirmation_seed in 1 2 3; do
  confirmation_root="$output_root/confirmation_seed_${confirmation_seed}"
  .venv/bin/python scripts/run_mqar_architecture_comparison.py \
    --asm-r-checkpoint "$asm_checkpoint" \
    --asm-s-checkpoint "$asm_s_checkpoint" \
    --transformer-checkpoint "$transformer_checkpoint" \
    --asm-r-config configs/asm_r_125m.json --variants ASM_C2_FW \
    --milestones "$steps" --batch-size 4 --eval-batches "$eval_batches" \
    --n-pairs 8 --n-queries 8 --seed "$confirmation_seed" \
    --device "$device" --precision bf16 --output-root "$confirmation_root"
  confirmation_files+=("$confirmation_root/results.json")
done

.venv/bin/python scripts/compare_asm_c2_controls.py \
  --short-results "$short_root/results.json" --ablations "$ablation_root/results.json" \
  --confirmations "${confirmation_files[@]}" \
  --output "$output_root/decision_multiseed.json" --report "$output_root/decision_multiseed.md"

multiseed="$(.venv/bin/python -c 'import json,sys; print(str(all(json.load(open(sys.argv[1]))["criteria"].values())).lower())' "$output_root/decision_multiseed.json")"
if [[ "$multiseed" != "true" ]]; then
  printf '%s\n' 'ASM-C2-FW failed the 2-of-3 multiseed gate; long evaluation blocked.'
  exit 0
fi

.venv/bin/python scripts/check_asm_c_parity.py \
  --checkpoint "$winner_checkpoint" --precision bf16 --device "$device" \
  --output "$output_root/bf16_parity.json"

long_root="$output_root/long_32k"
mkdir -p "$long_root"
.venv/bin/python scripts/benchmark_asm_r_long_streaming.py \
  --checkpoint "$winner_checkpoint" --compact \
  --lengths 512,1024,2048,4096,8192,16384,32768 \
  --prompt-tokens 64 --mqar-steps 0 --mqar-control-length 40 \
  --mqar-control-threshold 0.8 --mqar-batches "$eval_batches" \
  --mqar-batch-size 4 --device "$device" --output "$long_root/results.json"

language_root="$output_root/language_regression"
mkdir -p "$language_root"
.venv/bin/python scripts/evaluate_frozen_test.py \
  --family drm --checkpoint "$asm_checkpoint" \
  --manifest data/benchmark_125m_wikipedia/validation/manifest.json \
  --split validation --seq-len 512 --max-tokens 1000000 --batch-size 4 \
  --device "$device" --output "$language_root/asm_c_baseline.json"
.venv/bin/python scripts/evaluate_frozen_test.py \
  --family drm --checkpoint "$winner_checkpoint" \
  --manifest data/benchmark_125m_wikipedia/validation/manifest.json \
  --split validation --seq-len 512 --max-tokens 1000000 --batch-size 4 \
  --device "$device" --output "$language_root/asm_c2_fw_candidate.json"

.venv/bin/python scripts/compare_asm_c2_controls.py \
  --short-results "$short_root/results.json" --ablations "$ablation_root/results.json" \
  --confirmations "${confirmation_files[@]}" --long-results "$long_root/results.json" \
  --language-baseline "$language_root/asm_c_baseline.json" \
  --language-candidate "$language_root/asm_c2_fw_candidate.json" \
  --parity "$output_root/bf16_parity.json" \
  --output "$output_root/decision_final.json" --report "$output_root/report.md"

.venv/bin/python scripts/plot_asm_c2_results.py \
  --short-results "$short_root/results.json" --long-results "$long_root/results.json" \
  --output-root "$output_root/charts"

printf 'ASM-C2-FW suite complete: %s\n' "$output_root"
