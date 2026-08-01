#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$repo_root"
asm_root="${ASM_ROOT:-runs/asm_scaling_law_100m_seed1}"
transformer_root="${TRANSFORMER_ROOT:-runs/transformer_asm_r_matched_100m_seed1}"
output_root="${OUTPUT_ROOT:-runs/asm_transformer_paired_suite}"
manifest="${MANIFEST:-data/benchmark_125m_wikipedia/validation/manifest.json}"
asm_checkpoint="$asm_root/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt"
transformer_checkpoint="$transformer_root/checkpoint_milestone_100000000.pt"
mkdir -p "$output_root"

if [[ ! -f "$output_root/milestone_rescoring.json" ]]; then
  .venv/bin/python scripts/rescore_asm_transformer_100m.py \
    --asm-summary docs/benchmarks/asm_scaling_law_100m_seed1/scaling_law_summary.json \
    --transformer-root "$transformer_root" --manifest "$manifest" --device cuda \
    --max-tokens 10000000 --batch-size 8 --output "$output_root/milestone_rescoring.json"
else
  printf 'Reutilizando rescoring: %s\n' "$output_root/milestone_rescoring.json"
fi

.venv/bin/python scripts/benchmark_asm_transformer_paired.py \
  --asm-checkpoint "$asm_checkpoint" --transformer-checkpoint "$transformer_checkpoint" \
  --manifest "$manifest" --context-lengths 64,128,256,512,1024,2048 \
  --context-batches 8 --speed-lengths 64,128,256,512 --speed-repeats 5 \
  --decode-tokens 128 --generation-tokens 128 --device cuda \
  --output "$output_root/paired_benchmark.json"

if [[ -n "${PG19_MANIFEST:-}" ]]; then
  .venv/bin/python scripts/benchmark_asm_transformer_paired.py \
    --asm-checkpoint "$asm_checkpoint" --transformer-checkpoint "$transformer_checkpoint" \
    --manifest "$PG19_MANIFEST" --split test --context-lengths 64,128,256,512,1024,2048 \
    --context-batches 8 --skip-speed --skip-generation --device cuda \
    --output "$output_root/pg19_context.json"
fi

.venv/bin/python scripts/summarize_asm_transformer_paired.py --root "$output_root"

printf 'Suíte pareada concluída: %s/report.md\n' "$output_root"
