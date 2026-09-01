# Request Report

- Status: partial
- Date: 2026-08-31

## User request

Ler ASM/ e AdamM/, ajudar na Fase 1 do ASM-VR e avaliar se o AdamM pode ser útil ao ASM-VR.

## Summary

A Fase 1 foi delimitada como colapso sem bypass. Foi criado um primeiro gate isolado com frame fixo, máscaras hard, forcing pós-colapso, emissor restrito ao estado efetivo, teste pareado futuro e Jacobiano zero no complemento. A integração com controller hard, ASM-R e streaming permanece pendente. AdamM foi classificado como possível otimizador/ablação futura, não como componente do ASM-VR.

## Modified files

- [docs/ARCHITECTURE_ASM_VR.md](../ARCHITECTURE_ASM_VR.md)
- [docs/ASM_VR_ADAMM_ASSESSMENT.md](../ASM_VR_ADAMM_ASSESSMENT.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_vr_phase1/README.md](../benchmarks/asm_vr_phase1/README.md)
- [docs/benchmarks/asm_vr_phase1/summary.json](../benchmarks/asm_vr_phase1/summary.json)
- [scripts/eval_asm_vr_phase1.py](../../scripts/eval_asm_vr_phase1.py)
- [src/aletheion_state_models/geometry/variable_rank/__init__.py](../../src/aletheion_state_models/geometry/variable_rank/__init__.py)
- [src/aletheion_state_models/geometry/variable_rank/intrinsic_dynamics.py](../../src/aletheion_state_models/geometry/variable_rank/intrinsic_dynamics.py)
- [src/aletheion_state_models/geometry/variable_rank/phase1_experiments.py](../../src/aletheion_state_models/geometry/variable_rank/phase1_experiments.py)
- [tests/test_variable_rank_phase1.py](../../tests/test_variable_rank_phase1.py)
- [docs/report/0004_nucleo-fase-1-asm-vr-e-avaliacao-adamm_2026-08-31.md](0004_nucleo-fase-1-asm-vr-e-avaliacao-adamm_2026-08-31.md)

## Changes

- Criado núcleo de recorrência intrínseca que transporta e colapsa antes de calcular forcing, sem cache ou memória auxiliar.
- Criado experimento reproduzível 8→3→5→8 e testes de indistinguibilidade de estados e saídas futuras.
- Documentado o limite do gate isolado e os próximos marcos de integração com ASM-R.
- Documentada a utilidade indireta, riscos e matriz de comparação futura do AdamM.

## Validation

- .venv/bin/python -m pytest -q — 164 testes passaram; dois warnings preexistentes.
- .venv/bin/python scripts/eval_asm_vr_phase1.py --seed 2026 --output docs/benchmarks/asm_vr_phase1/summary.json — passou; diferenças e Jacobiano no complemento iguais a 0.0.
- .venv/bin/python -m compileall -q src/aletheion_state_models/geometry/variable_rank scripts/eval_asm_vr_phase1.py — passou.
- solid_source_modularity — novos arquivos Python conformes; quatro violações e oito exceções preexistentes fora do escopo.
- git diff --check — passou.
