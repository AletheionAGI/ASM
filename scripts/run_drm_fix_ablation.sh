#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"

exec "$python_bin" scripts/run_drm_fix_ablation.py "$@"
