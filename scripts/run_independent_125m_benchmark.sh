#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
train_manifest="${TRAIN_MANIFEST:-data/benchmark_125m_wikipedia/train/manifest.json}"
validation_manifest="${VALIDATION_MANIFEST:-data/benchmark_125m_wikipedia/validation/manifest.json}"
output_root="${OUTPUT_ROOT:-runs/independent_125m_frozen}"
target_tokens="${TARGET_TOKENS:-150000000}"
target_ce="1.3216"
seeds=(1 2 3)

for required in "$python_bin" "$train_manifest" "$validation_manifest"; do
    if [[ ! -e "$required" ]]; then
        echo "Arquivo obrigatório não encontrado: $required" >&2
        exit 1
    fi
done

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA indisponível"; print(f"GPU: {torch.cuda.get_device_name(0)}")'
mkdir -p "$output_root"

is_complete() {
    local summary="$1"
    "$python_bin" - "$summary" "$target_tokens" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
complete = path.exists() and int(json.loads(path.read_text())["tokens_seen"]) >= target
raise SystemExit(0 if complete else 1)
PY
}

resume_args() {
    local run_dir="$1"
    if [[ -f "$run_dir/checkpoint_latest.pt" ]]; then
        printf '%s\n' "--resume" "latest"
    fi
}

for seed in "${seeds[@]}"; do
    run_dir="$output_root/drm_125m_real_local_mixer_seed_$seed"
    if is_complete "$run_dir/summary.json"; then
        echo "DRM seed $seed já concluído; reutilizando."
        continue
    fi
    mapfile -t resume < <(resume_args "$run_dir")
    echo "Treinando DRM seed $seed..."
    "$python_bin" scripts/train_drm_memmap.py \
        --config configs/drm_125m_real.yaml \
        --train-manifest "$train_manifest" \
        --validation-manifest "$validation_manifest" \
        --output-root "$run_dir" \
        --target-tokens "$target_tokens" \
        --batch-size 2 \
        --grad-accum-steps 8 \
        --seq-len 512 \
        --lr 3e-4 \
        --weight-decay 0.01 \
        --max-grad-norm 1.0 \
        --precision bf16 \
        --device cuda \
        --seed "$seed" \
        --eval-tokens-interval 1000000 \
        --checkpoint-tokens-interval 50000000 \
        --eval-batches 4 \
        --log-interval 10 \
        --save-best-checkpoint \
        --sequence-mode directional_block_cumsum \
        --directional-cumsum-block-size 64 \
        --directional-anderson-iterations 0 \
        --directional-cumsum-step-mode velocity \
        --directional-local-mixer causal_conv \
        --directional-local-mixer-hidden-size 256 \
        --directional-local-mixer-kernel-size 8 \
        --directional-local-mixer-layers 2 \
        --directional-local-mixer-scale 0.2 \
        "${resume[@]}"
done

for seed in "${seeds[@]}"; do
    run_dir="$output_root/gpt2_125m_real_seed_$seed"
    if is_complete "$run_dir/summary.json"; then
        echo "GPT-2 seed $seed já concluído; reutilizando."
        continue
    fi
    mapfile -t resume < <(resume_args "$run_dir")
    echo "Treinando GPT-2 seed $seed..."
    "$python_bin" scripts/train_gpt2_memmap.py \
        --model-size gpt2_125m_real \
        --train-manifest "$train_manifest" \
        --validation-manifest "$validation_manifest" \
        --output-root "$run_dir" \
        --target-tokens "$target_tokens" \
        --batch-size 2 \
        --grad-accum-steps 8 \
        --seq-len 512 \
        --lr 3e-4 \
        --weight-decay 0.01 \
        --max-grad-norm 1.0 \
        --dropout 0.0 \
        --precision bf16 \
        --device cuda \
        --seed "$seed" \
        --eval-tokens-interval 1000000 \
        --checkpoint-tokens-interval 50000000 \
        --eval-batches 4 \
        --log-interval 100 \
        --save-best-checkpoint \
        "${resume[@]}"
done

"$python_bin" scripts/analyze_time_to_quality.py \
    --root "$output_root" \
    --target-ce "$target_ce" \
    --plateau-window 3 \
    --min-improvement-per-million 0.003

echo "Treinamentos concluídos. O PG-19 ainda não foi acessado."
echo "Dashboard: $output_root/dashboard.html"
