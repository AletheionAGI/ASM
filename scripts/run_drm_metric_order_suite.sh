#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-runs/drm_metric_order_ablation_5m}"
target_tokens="${TARGET_TOKENS:-5000000}"
variants="J,J_METRIC_SUBSPACE,J_METRIC_ORTHONORMAL_DIRECTION,J_NO_DIRECTION,J_DIRECT_CONTROL_MATCHED"

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

for candidate in J_METRIC_SUBSPACE J_METRIC_ORTHONORMAL_DIRECTION; do
    for baseline in J J_NO_DIRECTION J_DIRECT_CONTROL_MATCHED; do
        output="${output_root}/decision_${candidate,,}_vs_${baseline,,}.json"
        if "$python_bin" scripts/check_drm_fix_promotion.py \
            --summary "${output_root}/paired_validation_summary.json" \
            --candidate "$candidate" \
            --baseline "$baseline" \
            --output "$output"; then
            echo "$candidate venceu $baseline pelos critérios de promoção."
        else
            echo "$candidate não venceu $baseline pelos critérios de promoção."
        fi
    done
done

echo "Suíte de ordem métrica-direção concluída: ${output_root}"
