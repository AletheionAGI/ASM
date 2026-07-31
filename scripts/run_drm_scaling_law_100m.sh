#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
variants="${VARIANTS:-J_NO_DIRECTION,J_DIRECT_CONTROL_MATCHED}"
output_root="${OUTPUT_ROOT:-runs/drm_scaling_law_100m_seed1}"
seed="${SEED:-1}"
milestones="${MILESTONES:-1000000,2000000,5000000,10000000,20000000,30000000,50000000,100000000}"

"$python_bin" -m pytest -q

"$python_bin" scripts/run_drm_fix_ablation.py \
    --variants "$variants" \
    --target-tokens 100000000 \
    --seed "$seed" \
    --checkpoint-token-milestones "$milestones" \
    --no-save-best-checkpoint \
    --output-root "$output_root"

"$python_bin" scripts/rescore_drm_scaling_law.py \
    --root "$output_root" \
    --variants "$variants" \
    --milestones "$milestones" \
    --seed "$seed" \
    --device cuda

echo "Scaling law concluída: ${output_root}/scaling_law_summary.json"
