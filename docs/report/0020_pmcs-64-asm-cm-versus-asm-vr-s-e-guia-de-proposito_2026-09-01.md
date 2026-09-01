# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Realizar um experimento pareado entre ASM-CM e ASM-VR-S, comparar os dois, criar docs/MODEL_FAMILY_PURPOSE.md e docs/MODEL_FAMILY_PURPOSE_ptbr.md, delimitar o propósito prático de cada modelo da família ASM e explicar a utilidade prática de Variable Rank.

## Summary

Concluída a suíte PMCS-64 com três seeds e braços parameter-matched ASM-CM, ASM-VR-S full e fixed-32 em linguagem, especialização MQAR e streaming. Foram produzidos dashboard, nove gráficos PNG/SVG, resultados JSON reproduzíveis e guias bilíngues para toda a família. O experimento separou qualidade de linguagem, memória associativa e estabilidade streaming, registrando sem ocultação as falhas numéricas e gates reprovados.

## Modified files

- [docs/MODEL_FAMILY.md](../MODEL_FAMILY.md)
- [docs/MODEL_FAMILY_ptbr.md](../MODEL_FAMILY_ptbr.md)
- [docs/MODEL_FAMILY_PURPOSE.md](../MODEL_FAMILY_PURPOSE.md)
- [docs/MODEL_FAMILY_PURPOSE_ptbr.md](../MODEL_FAMILY_PURPOSE_ptbr.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/README.md](../benchmarks/asm_cm_vs_vr_s_pmcs64/README.md)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/index.html](../benchmarks/asm_cm_vs_vr_s_pmcs64/index.html)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/language_retention_after_mqar.png](../benchmarks/asm_cm_vs_vr_s_pmcs64/language_retention_after_mqar.png)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/language_retention_after_mqar.svg](../benchmarks/asm_cm_vs_vr_s_pmcs64/language_retention_after_mqar.svg)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/language_test_ce.png](../benchmarks/asm_cm_vs_vr_s_pmcs64/language_test_ce.png)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/language_test_ce.svg](../benchmarks/asm_cm_vs_vr_s_pmcs64/language_test_ce.svg)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/language_validation_ce.png](../benchmarks/asm_cm_vs_vr_s_pmcs64/language_validation_ce.png)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/language_validation_ce.svg](../benchmarks/asm_cm_vs_vr_s_pmcs64/language_validation_ce.svg)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/language_vs_durable_recall.png](../benchmarks/asm_cm_vs_vr_s_pmcs64/language_vs_durable_recall.png)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/language_vs_durable_recall.svg](../benchmarks/asm_cm_vs_vr_s_pmcs64/language_vs_durable_recall.svg)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/manifest.json](../benchmarks/asm_cm_vs_vr_s_pmcs64/manifest.json)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/mqar_accuracy_by_length.png](../benchmarks/asm_cm_vs_vr_s_pmcs64/mqar_accuracy_by_length.png)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/mqar_accuracy_by_length.svg](../benchmarks/asm_cm_vs_vr_s_pmcs64/mqar_accuracy_by_length.svg)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/mqar_ce_by_length.png](../benchmarks/asm_cm_vs_vr_s_pmcs64/mqar_ce_by_length.png)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/mqar_ce_by_length.svg](../benchmarks/asm_cm_vs_vr_s_pmcs64/mqar_ce_by_length.svg)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/observed_training_cost.png](../benchmarks/asm_cm_vs_vr_s_pmcs64/observed_training_cost.png)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/observed_training_cost.svg](../benchmarks/asm_cm_vs_vr_s_pmcs64/observed_training_cost.svg)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/streaming_retained_state.png](../benchmarks/asm_cm_vs_vr_s_pmcs64/streaming_retained_state.png)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/streaming_retained_state.svg](../benchmarks/asm_cm_vs_vr_s_pmcs64/streaming_retained_state.svg)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/streaming_throughput.png](../benchmarks/asm_cm_vs_vr_s_pmcs64/streaming_throughput.png)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/streaming_throughput.svg](../benchmarks/asm_cm_vs_vr_s_pmcs64/streaming_throughput.svg)
- [docs/benchmarks/asm_cm_vs_vr_s_pmcs64/summary.json](../benchmarks/asm_cm_vs_vr_s_pmcs64/summary.json)
- [src/aletheion_state_models/benchmarks/purpose_variants.py](../../src/aletheion_state_models/benchmarks/purpose_variants.py)
- [src/aletheion_state_models/benchmarks/purpose_mqar.py](../../src/aletheion_state_models/benchmarks/purpose_mqar.py)
- [src/aletheion_state_models/benchmarks/purpose_streaming.py](../../src/aletheion_state_models/benchmarks/purpose_streaming.py)
- [src/aletheion_state_models/benchmarks/purpose_summary.py](../../src/aletheion_state_models/benchmarks/purpose_summary.py)
- [src/aletheion_state_models/benchmarks/purpose_plots.py](../../src/aletheion_state_models/benchmarks/purpose_plots.py)
- [scripts/run_asm_cm_vs_vr_s_pmcs64.py](../../scripts/run_asm_cm_vs_vr_s_pmcs64.py)
- [scripts/resume_asm_cm_vs_vr_s_pmcs64.py](../../scripts/resume_asm_cm_vs_vr_s_pmcs64.py)
- [tests/test_purpose_variants.py](../../tests/test_purpose_variants.py)
- [docs/report/0020_pmcs-64-asm-cm-versus-asm-vr-s-e-guia-de-proposito_2026-09-01.md](0020_pmcs-64-asm-cm-versus-asm-vr-s-e-guia-de-proposito_2026-09-01.md)

## Changes

- Adicionado harness modular PMCS-64 com pareamento total de parâmetros em 0,028%, test selado, especialização memory-heavy e probes de estado retido.
- ASM-VR-S full venceu linguagem; fixed-32 permaneceu não inferior dentro de +0,03 nat, concluiu 32K e não mostrou ganho físico por rank.
- ASM-CM aprendeu MQAR curto em 3/3 seeds, mas recall longo robusto passou em apenas 1/3 e o streaming falhou no token 15.200; ASM-VR-S full falhou no token 30.335.
- Criados guias de propósito EN/PT-BR cobrindo 14 variantes, critérios de escolha, status, limites e usos atuais de Variable Rank.
- Gerados nove pares PNG/SVG e dashboard HTML offline; JSON não finito foi normalizado para null com flag de finitude.

## Validation

- PMCS-64 — 9/9 treinos de linguagem, 9/9 especializações memory-heavy e probes streaming concluídos ou registrados como falha de gate.
- python -m pytest -q — 209 testes passaram; dois warnings preexistentes.
- pytest targeted purpose/Variable Rank/fast-weight — 12 testes passaram.
- python -m compileall -q src scripts tests — passou.
- git diff --check — passou.
- Validação de artefatos — 2 JSON estritos, 9 PNG, 9 SVG, dashboard e links Markdown passaram; dois gráficos inspecionados visualmente.
- Auditoria SOLID/modularidade — todos os 8 arquivos-fonte novos conformes e abaixo de 300 linhas; 4 violações >500 linhas permanecem preexistentes e fora do escopo.
