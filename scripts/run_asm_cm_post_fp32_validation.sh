#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-runs/asm_cm_post_fp32_validation}"
confirmation_root="${CONFIRMATION_ROOT:-runs/asm_c2_fw_lm_confirmation}"
manifest="${VALIDATION_MANIFEST:-data/benchmark_125m_wikipedia/validation/manifest.json}"
device="${DEVICE:-cuda}"
mkdir -p "$output_root"

"$python_bin" -m pytest -q tests/test_asm_c_streaming.py tests/test_asm_c2_streaming.py tests/test_asm_c2_fw_lm.py
for seed in 1 2 3; do
  checkpoint="$confirmation_root/seed_${seed}/candidate/checkpoint_final.pt"
  seed_root="$output_root/seed_${seed}"
  mkdir -p "$seed_root"
  [[ -f "$checkpoint" ]] || { printf 'Checkpoint not found: %s\n' "$checkpoint" >&2; exit 2; }
  [[ -f "$seed_root/language.json" ]] || "$python_bin" scripts/evaluate_frozen_test.py --family drm --checkpoint "$checkpoint" --manifest "$manifest" --split validation --seq-len 512 --max-tokens 10000000 --batch-size 4 --device "$device" --output "$seed_root/language.json"
  [[ -f "$seed_root/streaming.json" ]] || "$python_bin" scripts/benchmark_asm_r_long_streaming.py --checkpoint "$checkpoint" --compact --lengths 512,4096,32768 --prompt-tokens 64 --skip-mqar --seed "$seed" --device "$device" --output "$seed_root/streaming.json"
  [[ -f "$seed_root/bf16_parity.json" ]] || "$python_bin" scripts/check_asm_c_parity.py --checkpoint "$checkpoint" --precision bf16 --device "$device" --output "$seed_root/bf16_parity.json"
done
"$python_bin" scripts/summarize_asm_c2_fw_lm_post_fp32.py --root "$output_root" --confirmation "$confirmation_root/decision.json"
printf 'Post-FP32 validation complete: %s/report.md\n' "$output_root"
