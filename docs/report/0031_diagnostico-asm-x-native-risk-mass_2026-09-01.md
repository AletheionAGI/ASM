# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Rodar o mesmo ATTR P2 do ASM-X com risk_mass, comparar ASM-X Base com ASM-X + Native Risk Mass, renomear os modelos e adicionar os resultados aos gráficos P2.

## Summary

## Resultado em linguagem humana

O teste comparou **ASM-X Base** com **ASM-X + Native Risk Mass**. Os dois modelos começaram com os mesmos tensores, tiveram exatamente 226.444 parâmetros incluindo as heads comuns e receberam os mesmos episódios, cinco seeds, objetivo, calibração validation-only e 1.000 updates. A única mudança foi ativar `use_powerlaw_risk`.

O resultado foi equivalência operacional. Em ID, AUPRC H8 foi 0,1504970 no Base e 0,1504975 com Native Risk Mass. O delta hierárquico pareado foi +0,00000001, IC95 [-0,00000560; +0,00000727]. Em shift, o delta foi -0,00000010, IC95 [-0,00000456; +0,00000250]. Em OOD, foi -0,00000124, IC95 [-0,00001633; +0,00000481]. Brier, recall/FPR no threshold de validation e NLL de próximo estado também ficaram praticamente idênticos. Todos os intervalos AUPRC em H1/H4/H8/H16 incluíram zero.

## Interpretação

O Native Risk Mass não ficou simplesmente desligado: o objetivo ATTR produziu gradientes nos parâmetros de risco, e eles se afastaram da inicialização nas cinco seeds. Mesmo assim, sob o peso e objetivo nativos congelados, essa mudança quase não alterou as previsões finais produzidas a partir da trajetória interna.

Portanto, ativar o Native Risk Mass atual **não converteu** a vantagem de dinâmica do ASM-X em melhor antecipação de perigo. Isso não demonstra que a ideia de modelagem de risco seja inútil. Demonstra somente que esta ativação nativa, com a configuração atual, teve influência desprezível neste benchmark. Investigar pesos ou objetivos diferentes seria uma nova hipótese pós-hoc e exigiria novo protocolo e, para confirmação, um test fresco.

## Integridade e limites

Esta variante foi solicitada depois que o P2 original já havia sido aberto. Por isso ela está registrada como extensão exploratória pós-hoc, não como sétimo braço confirmatório. O seal original de seis braços, G0-G5 e `predictive_passed=false` não foram alterados. Cinco checkpoints próprios foram hasheados antes da avaliação da extensão, e os nomes públicos foram padronizados como **ASM-X Base** e **ASM-X + Native Risk Mass**.

## Gráficos

O dashboard P2 agora inclui grades ID/shift/OOD, AUPRC por horizonte, bootstrap pareado e comparação por seed. Os artefatos estão em `docs/benchmarks/asm_transformer_transition_risk/p2/`.

## Modified files

- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md)
- [docs/benchmarks/README.md](../benchmarks/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/README.md](../benchmarks/asm_transformer_transition_risk/p2/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/index.html](../benchmarks/asm_transformer_transition_risk/p2/index.html)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.png](../benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.svg](../benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.png](../benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.svg](../benchmarks/asm_transformer_transition_risk/p2/registered_pair_deltas.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_pretrain_manifest.json](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_pretrain_manifest.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_checkpoint_seal.json](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_checkpoint_seal.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_summary.json](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_metrics.png](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_metrics.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_metrics.svg](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_metrics.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_horizons.png](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_horizons.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_horizons.svg](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_horizons.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_deltas.png](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_deltas.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_deltas.svg](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_deltas.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_test_id_multiseed.png](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_test_id_multiseed.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_test_id_multiseed.svg](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_test_id_multiseed.svg)
- [scripts/run_attr_p2_risk_mass.py](../../scripts/run_attr_p2_risk_mass.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_plots.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_plots.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_models.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_models.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_runner.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_runner.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_summary.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_summary.py)
- [src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_plots.py](../../src/aletheion_state_models/benchmarks/transition_risk/p2_risk_mass_plots.py)
- [tests/test_transition_risk_p2_risk_mass.py](../../tests/test_transition_risk_p2_risk_mass.py)
- [runs/attr_p2](../../runs/attr_p2)
- [docs/report/0031_diagnostico-asm-x-native-risk-mass_2026-09-01.md](0031_diagnostico-asm-x-native-risk-mass_2026-09-01.md)

## Changes

- Criei extensão paralela pós-hoc sem alterar P2_ARMS, preseal, dataset seal ou gates registrados.
- Ativei somente use_powerlaw_risk, mantendo parâmetros e inicialização exatamente pareados.
- Treinei e congelei cinco checkpoints, avaliei validation/ID/shift/OOD e calculei bootstrap pareado por seed→world→episode em H1/H4/H8/H16.
- Adicionei quatro gráficos PNG/SVG e uma seção diagnóstica ao dashboard P2.
- Renomeei os rótulos públicos para ASM-X Base e ASM-X + Native Risk Mass.

## Validation

- 5/5 checkpoints e 5/5 validation JSONL — hashes, 64 episódios e números finitos verificados
- 15/15 test JSONL — 128 episódios cada, JSON válido e sem NaN/Infinity
- Manifesto pré-treino permaneceu idêntico; hash 0760d265dab2af640d74b4c8cf8064b23d1d4c3a6cdae633e818dd7e53d459d3
- Seal da extensão — cinco hashes válidos; hash 88f2b93967b6e5e4ffbe97ab3b6232fa890e6812dbe0c6fa4064cac602182ee0
- python -m pytest -q — 270 testes passaram, cinco warnings conhecidos
- ruff check — passou
- python -m compileall -q src world_model transformer scripts — passou
- git diff --check — passou
- Dashboard — 21 links válidos; sete PNGs e sete SVGs abertos/parseados
- Auditoria modular — todos os quatro módulos risk-mass têm <=300 linhas; quatro violações >500 são preexistentes
