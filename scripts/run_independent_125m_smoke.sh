#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
train_manifest="${TRAIN_MANIFEST:-data/benchmark_125m_wikipedia/train/manifest.json}"
validation_manifest="${VALIDATION_MANIFEST:-data/benchmark_125m_wikipedia/validation/manifest.json}"
output_root="${OUTPUT_ROOT:-runs/independent_125m_smoke}"

for required in "$python_bin" "$train_manifest" "$validation_manifest"; do
    if [[ ! -e "$required" ]]; then
        echo "Arquivo obrigatório não encontrado: $required" >&2
        exit 1
    fi
done

"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA indisponível"; print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB)")'

echo "Executando dry-run DRM 125M..."
"$python_bin" scripts/train_drm_memmap.py \
    --config configs/drm_125m_real.yaml \
    --train-manifest "$train_manifest" \
    --validation-manifest "$validation_manifest" \
    --output-root "$output_root/drm_seed_1" \
    --target-tokens 8192 \
    --batch-size 2 \
    --grad-accum-steps 8 \
    --seq-len 512 \
    --precision bf16 \
    --device cuda \
    --seed 1 \
    --sequence-mode directional_block_cumsum \
    --directional-cumsum-block-size 64 \
    --directional-anderson-iterations 0 \
    --directional-cumsum-step-mode velocity \
    --directional-local-mixer causal_conv \
    --directional-local-mixer-hidden-size 256 \
    --directional-local-mixer-kernel-size 8 \
    --directional-local-mixer-layers 2 \
    --directional-local-mixer-scale 0.2 \
    --dry-run \
    --dry-run-forward

echo "Executando dry-run GPT-2 125M..."
"$python_bin" scripts/train_gpt2_memmap.py \
    --model-size gpt2_125m_real \
    --train-manifest "$train_manifest" \
    --validation-manifest "$validation_manifest" \
    --output-root "$output_root/gpt2_seed_1" \
    --target-tokens 8192 \
    --batch-size 2 \
    --grad-accum-steps 8 \
    --seq-len 512 \
    --precision bf16 \
    --device cuda \
    --seed 1 \
    --dry-run \
    --dry-run-forward

echo "Smoke test concluído com sucesso."
