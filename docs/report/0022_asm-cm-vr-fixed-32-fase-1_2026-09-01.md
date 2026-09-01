# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Implementar a proposta aprovada de começar pelo ASM-CM-VR fixed-32, sem usar o controller adaptativo atual, e validar se a memória associativa do ASM-CM pode coexistir com um estado principal de rank lógico 32 sem bypass.

## Summary

Implementado ASM-CM-VR strict fixed-32 com payload fast-weight alinhado ao estado, projeção antes de escrita e após leitura, controller congelado, ordem full/stream unificada e configuração opt-in segura. O smoke seed 17 comparou full-64 e fixed-32 em 1.000 updates MQAR-40: ambos chegaram a 100%, as ablações sem leitura/escrita caíram a próximo do acaso, a paridade máxima foi 2,86e-6 e ambos concluíram streaming 4K. Todos os oito gates da Fase 1 passaram; MQAR longo e multiseed permanecem próximos gates.

## Modified files

- [docs/ARCHITECTURE_ASM_CM_VR.md](../ARCHITECTURE_ASM_CM_VR.md)
- [docs/MODEL_FAMILY.md](../MODEL_FAMILY.md)
- [docs/MODEL_FAMILY_ptbr.md](../MODEL_FAMILY_ptbr.md)
- [docs/MODEL_FAMILY_PURPOSE.md](../MODEL_FAMILY_PURPOSE.md)
- [docs/MODEL_FAMILY_PURPOSE_ptbr.md](../MODEL_FAMILY_PURPOSE_ptbr.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/README.md](../benchmarks/asm_cm_vr_fixed32_phase1/README.md)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/index.html](../benchmarks/asm_cm_vr_fixed32_phase1/index.html)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/manifest.json](../benchmarks/asm_cm_vr_fixed32_phase1/manifest.json)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/memory_causal_ablations.png](../benchmarks/asm_cm_vr_fixed32_phase1/memory_causal_ablations.png)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/memory_causal_ablations.svg](../benchmarks/asm_cm_vr_fixed32_phase1/memory_causal_ablations.svg)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/mqar_learning.png](../benchmarks/asm_cm_vr_fixed32_phase1/mqar_learning.png)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/mqar_learning.svg](../benchmarks/asm_cm_vr_fixed32_phase1/mqar_learning.svg)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/streaming_throughput.png](../benchmarks/asm_cm_vr_fixed32_phase1/streaming_throughput.png)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/streaming_throughput.svg](../benchmarks/asm_cm_vr_fixed32_phase1/streaming_throughput.svg)
- [docs/benchmarks/asm_cm_vr_fixed32_phase1/summary.json](../benchmarks/asm_cm_vr_fixed32_phase1/summary.json)
- [src/drm_language_emitter/config.py](../../src/drm_language_emitter/config.py)
- [src/drm_language_emitter/fast_weight_memory.py](../../src/drm_language_emitter/fast_weight_memory.py)
- [src/drm_language_emitter/rank_aware_memory.py](../../src/drm_language_emitter/rank_aware_memory.py)
- [src/drm_language_emitter/directional_forward.py](../../src/drm_language_emitter/directional_forward.py)
- [src/drm_language_emitter/inference.py](../../src/drm_language_emitter/inference.py)
- [src/aletheion_state_models/variants/compact_variable_rank.py](../../src/aletheion_state_models/variants/compact_variable_rank.py)
- [src/aletheion_state_models/variants/__init__.py](../../src/aletheion_state_models/variants/__init__.py)
- [src/aletheion_state_models/benchmarks/cmvr_phase1.py](../../src/aletheion_state_models/benchmarks/cmvr_phase1.py)
- [src/aletheion_state_models/benchmarks/cmvr_phase1_plots.py](../../src/aletheion_state_models/benchmarks/cmvr_phase1_plots.py)
- [scripts/run_asm_cm_vr_fixed32_phase1.py](../../scripts/run_asm_cm_vr_fixed32_phase1.py)
- [tests/test_asm_cm_vr.py](../../tests/test_asm_cm_vr.py)
- [docs/report/0022_asm-cm-vr-fixed-32-fase-1_2026-09-01.md](0022_asm-cm-vr-fixed-32-fase-1_2026-09-01.md)

## Changes

- Adicionado builder público build_compact_memory_variable_rank(base, fixed_rank=32) e identidade experimental ASM-CM-VR.
- Separados key_dim e value_dim da fast-weight memory; o eixo de valor strict é alinhado a d_state e recebe máscara estrutural com torch.where.
- Projetados estado, matrix, consolidated, candidate, read e saída, fechando bypass de payload e preservando token/key como plano causal de endereçamento.
- Unificada a chamada rank-aware de memória entre full forward e decode, preservando o comportamento legacy quando a policy está desligada.
- Adicionados oito testes dedicados de configuração, canário, Jacobiano, shrink-grow, equivalência full e paridade/caches.
- Executado smoke full-64 versus fixed-32 com ablações causais, streaming, PNG/SVG e dashboard offline.
- Atualizadas taxonomia, guia de propósito e documentação arquitetural bilíngue/referenciada.

## Validation

- ASM-CM-VR Phase 1 seed 17 — 8/8 gates passaram.
- MQAR-40 held-out — full-64 100% CE 0,00747; fixed-32 100% CE 0,01627.
- Ablações fixed-32 — sem leitura 1,56%; sem escrita 1,56%.
- Streaming 4K — full-64 123,14 tok/s; fixed-32 123,70 tok/s; ambos 66.112 bytes e erro full/stream 2,86e-6.
- python -m pytest -q — 217 testes passaram; dois warnings preexistentes.
- Testes direcionados finais ASM-CM-VR/fast-weight/config — 22 passaram.
- python -m compileall -q src scripts tests e git diff --check — passaram.
- Artefatos — 2 JSON estritos, 3 PNG, 3 SVG, dashboard e links passaram; dois gráficos inspecionados visualmente.
- Auditoria modular — todos os arquivos novos conformes; fast_weight_memory.py com 299 linhas. config.py permanece exceção coesa documentada em 495 linhas; quatro violações >500 preexistentes permanecem fora do escopo.
