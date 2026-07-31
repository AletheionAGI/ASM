#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-runs/drm_fix_paired_5m}"
target_tokens="${TARGET_TOKENS:-5000000}"
variants="${VARIANTS:-F,I}"
seeds=(1 2 3)

for seed in "${seeds[@]}"; do
    "$python_bin" scripts/run_drm_fix_ablation.py \
        --variants "$variants" \
        --target-tokens "$target_tokens" \
        --seed "$seed" \
        --output-root "$output_root"
done

"$python_bin" scripts/rescore_drm_fix_validation.py \
    --root "$output_root" \
    --variants "${variants,,}" \
    --seeds "1,2,3" \
    --device cuda
