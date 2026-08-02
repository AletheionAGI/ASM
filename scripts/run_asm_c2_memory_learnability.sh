#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

output_root="${OUTPUT_ROOT:-runs/asm_c2_memory_learnability}"
device="${DEVICE:-cuda}"
mkdir -p "$output_root"

.venv/bin/python -m pytest -q \
  tests/test_addressable_memory.py \
  tests/test_memory_learnability_probe.py

.venv/bin/python scripts/probe_addressable_memory.py \
  --steps "${STEPS:-10000}" \
  --slots "${SLOTS:-16}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --eval-batches "${EVAL_BATCHES:-32}" \
  --device "$device" \
  --output "$output_root/results.json"

if .venv/bin/python -c \
  'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["passed"] else 3)' \
  "$output_root/results.json"
then
  printf '%s\n' 'Fast-weight memory passed the isolated gate. ASM-C2 reintegration is authorized.'
else
  printf '%s\n' 'No architectural memory passed the isolated gate; ASM-C2 reintegration remains blocked.'
fi

printf 'Report: %s\n' "$output_root/report.md"
