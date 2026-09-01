# Request Report

- Status: completed
- Date: 2026-08-31

## User request

Terminar a Fase 1 do ASM-VR e determinar/implementar tudo o que era necessário para concluí-la.

## Summary

A Fase 1 foi concluída com uma variante ASM-R opt-in de rank hard por bloco, controller causal input-only por exemplo, frame identidade fixo, projeção antes/depois de cada bloco, cache efetivo tipado sem estado ambiente/prefixo, decode sem fallback e rejeição de todas as rotas auxiliares de bypass. Foram validados logits reais, Jacobiano, causalidade, expansão, cache e streaming.

## Modified files

- [docs/ARCHITECTURE_ASM_VR.md](../ARCHITECTURE_ASM_VR.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_vr_phase1/README.md](../benchmarks/asm_vr_phase1/README.md)
- [docs/benchmarks/asm_vr_phase1/integration_summary.json](../benchmarks/asm_vr_phase1/integration_summary.json)
- [scripts/eval_asm_vr_phase1_integration.py](../../scripts/eval_asm_vr_phase1_integration.py)
- [src/aletheion_state_models/geometry/variable_rank/__init__.py](../../src/aletheion_state_models/geometry/variable_rank/__init__.py)
- [src/aletheion_state_models/geometry/variable_rank/batch_state.py](../../src/aletheion_state_models/geometry/variable_rank/batch_state.py)
- [src/aletheion_state_models/geometry/variable_rank/block_core.py](../../src/aletheion_state_models/geometry/variable_rank/block_core.py)
- [src/aletheion_state_models/geometry/variable_rank/rank_controller.py](../../src/aletheion_state_models/geometry/variable_rank/rank_controller.py)
- [src/aletheion_state_models/variants/__init__.py](../../src/aletheion_state_models/variants/__init__.py)
- [src/aletheion_state_models/variants/variable_rank.py](../../src/aletheion_state_models/variants/variable_rank.py)
- [src/drm_language_emitter/config.py](../../src/drm_language_emitter/config.py)
- [src/drm_language_emitter/directional_forward.py](../../src/drm_language_emitter/directional_forward.py)
- [src/drm_language_emitter/inference.py](../../src/drm_language_emitter/inference.py)
- [src/drm_language_emitter/model.py](../../src/drm_language_emitter/model.py)
- [tests/test_asm_vr_phase1_integration.py](../../tests/test_asm_vr_phase1_integration.py)
- [docs/report/0005_conclusao-fase-1-asm-vr_2026-08-31.md](0005_conclusao-fase-1-asm-vr_2026-08-31.md)

## Changes

- Adicionado VariableRankBatchState padded validado, com posições inativas obrigatoriamente zero.
- Adicionado controller hard por exemplo que observa apenas o primeiro token causal do bloco e não recebe estado recorrente.
- Integrada projeção hard nas fronteiras do block-cumsum do núcleo ASM-R, mantendo o caminho legado default-off intacto.
- Adicionado cache ASM-VR com somente estado efetivo/máscara e bloqueio do fallback de decode para prefixo completo.
- Adicionado builder público build_variable_rank_phase1 e validações que proíbem mixer, residual, memórias, refinamentos e solvers na Fase 1.
- Adicionados testes integrados e artefato reproduzível; documentação marca a Fase 1 como concluída e delimita os claims.

## Validation

- .venv/bin/python -m pytest -q — 174 testes passaram; dois warnings preexistentes.
- .venv/bin/python scripts/eval_asm_vr_phase1_integration.py --seed 2026 --output docs/benchmarks/asm_vr_phase1/integration_summary.json — passou; diferenças pareadas/Jacobiano/cache inativo 0.0 e erro streaming 1.0430813e-07.
- .venv/bin/python -m compileall -q src scripts/eval_asm_vr_phase1_integration.py — passou.
- git diff --check — passou.
- solid_source_modularity — novos módulos conformes; config.py (455), directional_forward.py (381) e model.py (418) permanecem exceções coesas de composição/schema dentro de 301–500 linhas; quatro violações >500 preexistentes e fora do escopo.
