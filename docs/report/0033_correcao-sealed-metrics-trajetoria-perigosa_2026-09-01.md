# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Refazer o gráfico sealed_metrics com a medição correta, após identificar que a AUPRC H8 direta não depende da NLL de próximo estado.

## Summary

## Gráfico corrigido

O gráfico `sealed_metrics` foi refeito para não sugerir que a AUPRC H8 vem da NLL. Ele agora mostra, em cada split:

1. **Direct hazard-head AUPRC H8** — classificação produzida pela head direta, explicitamente rotulada;
2. **Next-state NLL | H8=1** — erro de dinâmica nos passos a partir dos quais ocorrerá entrada unsafe em até oito passos;
3. **Next-state NLL | H8=0** — erro de dinâmica nos demais passos.

## Resultado revelado

O Transformer continua pior em prever o próximo estado mesmo quando a medição é restrita às trajetórias que se aproximam de perigo. Em ID H8-positivo, ASM-X Base/Transformer têm NLL `2,9290/4,1385`; em shift, `6,5802/12,6509`; em OOD, `4,4692/9,1122`. Nos passos H8-negativos, ID é `2,5557/3,5200`.

Portanto, a AUPRC H8 competitiva do Transformer não vem de melhor previsão de trajetória. Ela vem da `HazardHead` direta, que consegue ordenar algumas pistas correlacionadas com perigo mesmo com dinâmica pior. A AUPRC do Transformer continua fraca em termos absolutos: `0,1498` para prevalência `0,1283`.

## Limite metodológico

Os labels H8 são usados somente depois da inferência para separar a NLL em grupos positivos e negativos. Eles não entram no modelo nem viram score de risco. Não foi criada uma “AUPRC baseada em NLL”, porque a NLL realizada depende do próximo estado verdadeiro e usá-la como score antecipatório causaria leakage futuro.

Uma AUPRC realmente derivada da trajetória exige um novo experimento: previsão explícita de estados em múltiplos horizontes e aplicação de um predicado unsafe externo e fixo às trajetórias previstas. Os checkpoints atuais possuem somente uma head de próximo passo, portanto não suportam essa medição sem mudar o protocolo.

## Modified files

- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/README.md](../benchmarks/asm_transformer_transition_risk/p2/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/index.html](../benchmarks/asm_transformer_transition_risk/p2/index.html)
- [docs/benchmarks/asm_transformer_transition_risk/p2/summary.json](../benchmarks/asm_transformer_transition_risk/p2/summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_summary.json](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.png](../benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.svg](../benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/dynamics_vs_anticipation.png](../benchmarks/asm_transformer_transition_risk/p2/dynamics_vs_anticipation.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/dynamics_vs_anticipation.svg](../benchmarks/asm_transformer_transition_risk/p2/dynamics_vs_anticipation.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.png](../benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.svg](../benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_metrics.png](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_metrics.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_metrics.svg](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_metrics.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_horizons.png](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_horizons.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_horizons.svg](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_horizons.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_deltas.png](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_deltas.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_deltas.svg](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_deltas.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_test_id_multiseed.png](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_test_id_multiseed.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_test_id_multiseed.svg](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_test_id_multiseed.svg)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_conditioned_metrics.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_conditioned_metrics.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_summary.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_summary.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_plots.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_plots.py)
- [tests/test_transition_risk_p2_conditioned_metrics.py](../../tests/test_transition_risk_p2_conditioned_metrics.py)
- [docs/report/0033_correcao-sealed-metrics-trajetoria-perigosa_2026-09-01.md](0033_correcao-sealed-metrics-trajetoria-perigosa_2026-09-01.md)

## Changes

- Adicionei agregação de NLL condicionada ao label H8 positivo/negativo.
- Substituí no sealed_metrics os painéis genéricos por AUPRC explicitamente direta e NLL H8-condicionada.
- Mantive os labels H8 apenas como estratificação de avaliação, sem criar score com leakage.
- Documentei que AUPRC derivada de trajetória requer novo protocolo multi-horizonte.

## Validation

- 44 testes transition-risk — passaram; três warnings conhecidos do Transformer
- ruff check nos módulos e teste modificados — passou
- compileall transition_risk — passou
- git diff --check — passou
- sealed_metrics.png aberto e verificado; sealed_metrics.svg parseado
- Dashboard — 23 links válidos
- Auditoria modular — p2_conditioned_metrics 43 linhas, p2_summary 299, p2_plots 179; quatro violações >500 permanecem preexistentes
