# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Colocar a medição de dinâmica condicionada ao hazard e o futuro experimento sem classifier em gráficos separados, sem misturá-los ao sealed_metrics canônico.

## Summary

## Separação dos gráficos

As medições foram separadas para evitar misturar mecanismos diferentes.

- `sealed_metrics` continua sendo o gráfico canônico do P2. Seus painéis agora dizem explicitamente `Direct hazard-head AUPRC H8` e `Direct hazard-head Brier H8`, ao lado da NLL one-step agregada.
- `hazard_conditioned_dynamics` é um novo gráfico independente. Ele mostra NLL de próximo estado nos passos H8=1 e H8=0, deixando visível que o Transformer é pior em dinâmica mesmo nas janelas que antecedem perigo.
- `dynamics_vs_anticipation` continua como scatter descritivo entre NLL e AUPRC direta.

O futuro experimento sem hazard classifier deverá gerar outro artefato separado, proposto como `trajectory_grounded_anticipation`. Esse gráfico só pode ser produzido depois de treinar previsores multi-horizonte e avaliar risco por um predicado unsafe fixo em um test fresco. Nenhum resultado foi inventado ou extrapolado dos checkpoints one-step atuais.

## Resultado atual preservado

Em H8=1, ASM-X Base/Transformer têm NLL `2,9290/4,1385` em ID, `6,5802/12,6509` em shift e `4,4692/9,1122` em OOD. Isso é evidência de dinâmica condicionada ao perigo e está no gráfico próprio; não substitui nem modifica a AUPRC da classifier direta ou os gates P2.

## Modified files

- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/README.md](../benchmarks/asm_transformer_transition_risk/p2/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/index.html](../benchmarks/asm_transformer_transition_risk/p2/index.html)
- [docs/benchmarks/asm_transformer_transition_risk/p2/summary.json](../benchmarks/asm_transformer_transition_risk/p2/summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_summary.json](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/hazard_conditioned_dynamics.png](../benchmarks/asm_transformer_transition_risk/p2/hazard_conditioned_dynamics.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/hazard_conditioned_dynamics.svg](../benchmarks/asm_transformer_transition_risk/p2/hazard_conditioned_dynamics.svg)
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
- [src/aletheion_state_models/benchmarks/transition_risk/p2_plots.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_plots.py)
- [docs/report/0036_separacao-graficos-head-direta-e-dinamica_2026-09-01.md](0036_separacao-graficos-head-direta-e-dinamica_2026-09-01.md)

## Changes

- Restaurei sealed_metrics como gráfico canônico, com rótulos explícitos de head direta.
- Criei hazard_conditioned_dynamics em PNG/SVG como gráfico separado.
- Adicionei o novo gráfico ao dashboard e documentei a reserva de trajectory_grounded_anticipation para o futuro protocolo.

## Validation

- Testes focados de conditioned metrics e summary — 3 passaram
- ruff check p2_plots.py — passou
- git diff --check — passou
- sealed_metrics e hazard_conditioned_dynamics inspecionados visualmente
- Dashboard regenerado com link para hazard_conditioned_dynamics
