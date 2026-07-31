#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-runs/drm_direct_control_ablation_5m}"
target_tokens="${TARGET_TOKENS:-5000000}"
variants="J_NO_DIRECTION,J_DIRECT_CONTROL,J_DIRECT_CONTROL_MATCHED"

"$python_bin" -m pytest -q

"$python_bin" scripts/run_drm_fix_ablation.py \
    --variants "$variants" \
    --device cpu \
    --precision fp32 \
    --batch-size 1 \
    --grad-accum-steps 1 \
    --seq-len 16 \
    --target-tokens 16 \
    --dry-run \
    --dry-run-forward \
    --output-root "${output_root}_smoke"

VARIANTS="$variants" \
OUTPUT_ROOT="$output_root" \
TARGET_TOKENS="$target_tokens" \
PYTHON_BIN="$python_bin" \
    ./scripts/run_drm_fix_paired_5m.sh

for candidate in J_DIRECT_CONTROL J_DIRECT_CONTROL_MATCHED; do
    output="${output_root}/decision_${candidate,,}_vs_j_no_direction.json"
    if "$python_bin" scripts/check_drm_fix_promotion.py \
        --summary "${output_root}/paired_validation_summary.json" \
        --candidate "$candidate" \
        --baseline J_NO_DIRECTION \
        --output "$output"; then
        echo "$candidate venceu J_NO_DIRECTION pelos critérios de promoção."
    else
        echo "$candidate não venceu J_NO_DIRECTION pelos critérios de promoção."
    fi
done

echo "Suíte de controle direto concluída: ${output_root}"
