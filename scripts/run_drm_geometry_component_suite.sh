#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-runs/drm_geometry_component_ablation_5m}"
variants="J,J_NO_METRIC,J_NO_DIRECTION,J_NO_NATURALIZATION"
mqar_steps="${MQAR_STEPS:-1000}"

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

for seed in 1 2 3; do
    "$python_bin" scripts/run_mqar_architecture_probe.py \
        --variants "$variants" \
        --steps "$mqar_steps" \
        --seed "$seed" \
        --device cuda \
        --output "${output_root}_mqar_seed_${seed}.json"
done

VARIANTS="$variants" \
OUTPUT_ROOT="$output_root" \
TARGET_TOKENS=5000000 \
PYTHON_BIN="$python_bin" \
    ./scripts/run_drm_fix_paired_5m.sh

for baseline in J_NO_METRIC J_NO_DIRECTION J_NO_NATURALIZATION; do
    output="${output_root}/decision_j_vs_${baseline,,}.json"
    if "$python_bin" scripts/check_drm_fix_promotion.py \
        --summary "${output_root}/paired_validation_summary.json" \
        --candidate J \
        --baseline "$baseline" \
        --output "$output"; then
        echo "J venceu $baseline pelos critérios de promoção."
    else
        echo "J não venceu $baseline pelos critérios de promoção."
    fi
done

echo "Suíte geométrica concluída: ${output_root}"
