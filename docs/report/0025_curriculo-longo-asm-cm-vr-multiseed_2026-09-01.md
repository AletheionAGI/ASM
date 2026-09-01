# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Continuar o trabalho após o término do processo longo ASM-CM-VR: inspecionar a execução, consolidar full-64/fixed-32/adaptive-32 nas seeds 17/29/43, preservar falhas numéricas, validar gráficos/dashboard/testes e publicar a decisão de promoção.

## Summary

Currículo longo concluído nas nove combinações. Fixed-32 não foi promovido: full/fixed passaram em 2/3 seeds e ambos tiveram CE não finita e falha de streaming 32K na seed 29. Adaptive variou rank e recebeu gradientes, mas também passou o threshold 32K em apenas 2/3 seeds. Resumo, PNG/SVG, dashboard e documentação foram atualizados; a classificação finite foi corrigida para falhar fechada.

## Modified files

- [src/aletheion_state_models/benchmarks/cmvr_long_curriculum.py](../../src/aletheion_state_models/benchmarks/cmvr_long_curriculum.py)
- [src/aletheion_state_models/benchmarks/cmvr_long_summary.py](../../src/aletheion_state_models/benchmarks/cmvr_long_summary.py)
- [src/aletheion_state_models/benchmarks/cmvr_long_plots.py](../../src/aletheion_state_models/benchmarks/cmvr_long_plots.py)
- [tests/test_cmvr_long_summary.py](../../tests/test_cmvr_long_summary.py)
- [docs/ARCHITECTURE_ASM_CM_VR.md](../ARCHITECTURE_ASM_CM_VR.md)
- [docs/MODEL_FAMILY.md](../MODEL_FAMILY.md)
- [docs/MODEL_FAMILY_ptbr.md](../MODEL_FAMILY_ptbr.md)
- [docs/MODEL_FAMILY_PURPOSE.md](../MODEL_FAMILY_PURPOSE.md)
- [docs/MODEL_FAMILY_PURPOSE_ptbr.md](../MODEL_FAMILY_PURPOSE_ptbr.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_cm_vr_fixed32_long/README.md](../benchmarks/asm_cm_vr_fixed32_long/README.md)
- [docs/benchmarks/asm_cm_vr_fixed32_long/adaptive_rank.png](../benchmarks/asm_cm_vr_fixed32_long/adaptive_rank.png)
- [docs/benchmarks/asm_cm_vr_fixed32_long/adaptive_rank.svg](../benchmarks/asm_cm_vr_fixed32_long/adaptive_rank.svg)
- [docs/benchmarks/asm_cm_vr_fixed32_long/curriculum_validation.png](../benchmarks/asm_cm_vr_fixed32_long/curriculum_validation.png)
- [docs/benchmarks/asm_cm_vr_fixed32_long/curriculum_validation.svg](../benchmarks/asm_cm_vr_fixed32_long/curriculum_validation.svg)
- [docs/benchmarks/asm_cm_vr_fixed32_long/index.html](../benchmarks/asm_cm_vr_fixed32_long/index.html)
- [docs/benchmarks/asm_cm_vr_fixed32_long/manifest.json](../benchmarks/asm_cm_vr_fixed32_long/manifest.json)
- [docs/benchmarks/asm_cm_vr_fixed32_long/memory_ablations_multiseed.png](../benchmarks/asm_cm_vr_fixed32_long/memory_ablations_multiseed.png)
- [docs/benchmarks/asm_cm_vr_fixed32_long/memory_ablations_multiseed.svg](../benchmarks/asm_cm_vr_fixed32_long/memory_ablations_multiseed.svg)
- [docs/benchmarks/asm_cm_vr_fixed32_long/mqar_long_accuracy.png](../benchmarks/asm_cm_vr_fixed32_long/mqar_long_accuracy.png)
- [docs/benchmarks/asm_cm_vr_fixed32_long/mqar_long_accuracy.svg](../benchmarks/asm_cm_vr_fixed32_long/mqar_long_accuracy.svg)
- [docs/benchmarks/asm_cm_vr_fixed32_long/streaming_state_bytes.png](../benchmarks/asm_cm_vr_fixed32_long/streaming_state_bytes.png)
- [docs/benchmarks/asm_cm_vr_fixed32_long/streaming_state_bytes.svg](../benchmarks/asm_cm_vr_fixed32_long/streaming_state_bytes.svg)
- [docs/benchmarks/asm_cm_vr_fixed32_long/streaming_throughput.png](../benchmarks/asm_cm_vr_fixed32_long/streaming_throughput.png)
- [docs/benchmarks/asm_cm_vr_fixed32_long/streaming_throughput.svg](../benchmarks/asm_cm_vr_fixed32_long/streaming_throughput.svg)
- [docs/benchmarks/asm_cm_vr_fixed32_long/summary.json](../benchmarks/asm_cm_vr_fixed32_long/summary.json)
- [runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_17/curriculum.pt](../../runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_17/curriculum.pt)
- [runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_17/result.json](../../runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_17/result.json)
- [runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_29/curriculum.pt](../../runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_29/curriculum.pt)
- [runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_29/result.json](../../runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_29/result.json)
- [runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_43/curriculum.pt](../../runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_43/curriculum.pt)
- [runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_43/result.json](../../runs/asm_cm_vr_fixed32_long/cm_vr_full64/seed_43/result.json)
- [runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_17/curriculum.pt](../../runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_17/curriculum.pt)
- [runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_17/result.json](../../runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_17/result.json)
- [runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_29/curriculum.pt](../../runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_29/curriculum.pt)
- [runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_29/result.json](../../runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_29/result.json)
- [runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_43/curriculum.pt](../../runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_43/curriculum.pt)
- [runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_43/result.json](../../runs/asm_cm_vr_fixed32_long/cm_vr_fixed32/seed_43/result.json)
- [runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_17/curriculum.pt](../../runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_17/curriculum.pt)
- [runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_17/result.json](../../runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_17/result.json)
- [runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_29/curriculum.pt](../../runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_29/curriculum.pt)
- [runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_29/result.json](../../runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_29/result.json)
- [runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_43/curriculum.pt](../../runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_43/curriculum.pt)
- [runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_43/result.json](../../runs/asm_cm_vr_fixed32_long/cm_vr_adaptive32/seed_43/result.json)
- [docs/report/0025_curriculo-longo-asm-cm-vr-multiseed_2026-09-01.md](0025_curriculo-longo-asm-cm-vr-multiseed_2026-09-01.md)

## Changes

- Consolidadas nove runs full-64, fixed-32 e adaptive-32 nas seeds 17, 29 e 43.
- Corrigida a classificação de finitude para incluir CE held-out e falhas de streaming; médias gerais continuam incluindo runs reprovadas, e médias de sucessos são separadas.
- Registradas explicitamente as falhas full/fixed seed 29 em MQAR e streaming 32K.
- Adicionado teste de regressão fail-closed para resumo e gate longo.
- Regenerados seis gráficos em PNG/SVG e dashboard offline com tabela de falhas.
- Atualizadas arquitetura, taxonomia e guias de propósito EN/PT-BR com a decisão de não promoção.

## Validation

- Processo 1067765 — exit code 0 após 12.776,6 s
- scripts/run_asm_cm_vr_fixed32_long.py — resumo regenerado; status completed_multiseed; passed=false
- .venv/bin/python -m pytest -q — 244 passed; 5 warnings conhecidos
- .venv/bin/python -m compileall -q src scripts tests — passou
- git diff --check — passou
- Validação de 6 PNG, 6 SVG, HTML, links locais, manifest e summary JSON — passou
- Auditoria SOLID Python — 268 compliant, 8 exceções e 4 violações preexistentes; nenhum arquivo novo/modificado excede 300 linhas
