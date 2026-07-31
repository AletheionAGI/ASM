#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"

"$python_bin" scripts/rescore_independent_125m_validation.py \
    --root "${OUTPUT_ROOT:-runs/independent_125m_frozen}" \
    --manifest "${VALIDATION_MANIFEST:-data/benchmark_125m_wikipedia/validation/manifest.json}" \
    --seq-len "${SEQ_LEN:-512}" \
    --max-tokens "${MAX_VALIDATION_TOKENS:-10000000}" \
    --drm-batch-size "${DRM_BATCH_SIZE:-4}" \
    --gpt2-batch-size "${GPT2_BATCH_SIZE:-8}" \
    --device "${DEVICE:-cuda}"
