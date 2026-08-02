#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
root="${CONFIRMATION_ROOT:-runs/asm_c2_fw_lm_confirmation}"
device="${DEVICE:-cuda}"

for seed in 1 2 3; do
  checkpoint="$root/seed_${seed}/candidate/checkpoint_final.pt"
  [[ -f "$checkpoint" ]] || {
    printf 'Candidate checkpoint not found: %s\n' "$checkpoint" >&2
    exit 2
  }
  "$python_bin" scripts/check_asm_c_parity.py \
    --checkpoint "$checkpoint" \
    --precision bf16 \
    --device "$device" \
    --output "$root/seed_${seed}/bf16_parity.json"
done

"$python_bin" scripts/summarize_asm_c2_fw_lm_confirmation.py \
  --root "$root" \
  --output "$root/decision.json" \
  --report "$root/report.md"

"$python_bin" scripts/plot_asm_c2_fw_lm_confirmation.py \
  --decision "$root/decision.json" \
  --output-root "$root/charts"

printf 'Parity correction decision: %s/decision.json\n' "$root"
