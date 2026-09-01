# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Implementar e executar o experimento ATTR trajectory-grounded sem hazard classifier, usando as mesmas cinco seeds do P2, test fresco e gráfico separado trajectory_grounded_anticipation.

## Summary

# ATTR-TG1 — antecipação fundamentada em trajetória

## Objetivo

Foi executado um novo experimento para testar a hipótese de que ASM-X Base e Transformer podem empatar na classificação direta de hazard, mas usar mecanismos diferentes. A `HazardHead` foi removida por completo. O único caminho permitido foi:

`representação causal → trajetória física prevista → predicado unsafe fixo → risco H8`.

O experimento é novo e não reinterpreta o P2. Foram usados os mesmos cinco optimizer seeds pedidos pelo usuário (`29, 43, 71, 89, 107`), mas mundos, episódios, planos, targets, checkpoints e tests são frescos.

## Protocolo implementado

- braços: `asm_x_base` e `transformer_base`;
- ASM 64→72 por zero-padding sem parâmetros; decoder 72 idêntico nos dois braços;
- parâmetros: ASM `253.874`, Transformer `254.472`, diferença `598` ou `0,235%`;
- 1.000 updates AdamW, batch 4, LR `3e-4`, weight decay `0,01`;
- plano open-loop H8 comprometido causalmente em `t`, sem ler observações futuras;
- previsão categórica autoregressiva de todos os passos H1…H8;
- targets físicos: traps, agente, hazard móvel, velocidade, energia, baixa energia, recuperação, modo e término seguro;
- loss exclusivamente de NLL física; zero BCE de hazard e zero parâmetros de `HazardHead`;
- K=256 trajetórias free-running com common random numbers pareados;
- risco derivado somente de colisão/falha atrasada no predicado fixo;
- bootstrap pareado hierárquico seed→world→episode, 1.000 réplicas.

Train/validation comuns usaram 64×4 e 16×4 episódios. Os tests ID/shift/OOD usaram 32×4 episódios cada e só foram materializados depois do seal dos dez checkpoints e das dez avaliações K256 em validation.

## Integridade

- preseal SHA-256: `576af49425d4e840123eec988a99f39e376135cececa15c908d59993f73fec43`;
- checkpoint seal SHA-256: `e2507f8ed9045ec8ac544d51ca486a80b629080e658a187ac46a3d15e2ff8582`;
- 10/10 checkpoints terminais;
- 10/10 arquivos validation;
- 30/30 arquivos test;
- 40 arquivos no prediction manifest;
- todos os checkpoints finitos e sem `hazard_head`, `hazard_logits`, `unsafe`, `severity` ou `time_to_hazard`;
- código e manifests continuaram idênticos ao preseal após a avaliação.

## Resultado principal H8

| Split | Prevalência | ASM AUPRC | Transformer AUPRC | ΔAUPRC ASM−T | IC95 | ASM Brier | Transformer Brier | ΔBrier (IC95) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ID | 0,5153 | 0,6279 | 0,6236 | +0,0043 | [-0,0115; +0,0206] | 0,2416 | 0,2417 | -0,00010 [-0,00251; +0,00250] |
| shift | 0,6483 | 0,7742 | 0,7732 | +0,0010 | [-0,0135; +0,0143] | 0,2061 | 0,2079 | -0,00188 [-0,00459; +0,00066] |
| OOD | 0,8586 | 0,9354 | 0,9301 | +0,0052 | [-0,00027; +0,01140] | 0,1173 | 0,1197 | -0,00242 [-0,00445; -0,00059] |

O gate registrado exigia ΔAUPRC ID ≥`0,03`, limite inferior do IC95 acima de zero e limite superior do ΔBrier ≤`0,01`. **TG2 falhou.** Os três intervalos de AUPRC H8 incluem zero. O resultado OOD favorece ASM em Brier, mas não confirma vantagem de ranking.

Direção de ΔAUPRC por seed: ID positiva em 3/5 seeds, shift em 3/5 e OOD em 5/5; os ganhos OOD são pequenos. A prevalência sob os planos open-loop é alta. O lift AUPRC/prevalência é `1,218×/1,210×` em ID e `1,089×/1,083×` em OOD para ASM/Transformer, respectivamente.

## Fidelidade da trajetória

| Split | ASM joint NLL H8 | Transformer joint NLL H8 | ΔNLL ASM−T | IC95 |
|---|---:|---:|---:|---:|
| ID | 20,5427 | 20,2058 | +0,3369 | [+0,2251; +0,4543] |
| shift | 19,9059 | 19,6089 | +0,2970 | [+0,1300; +0,4739] |
| OOD | 20,9931 | 20,7392 | +0,2539 | [+0,1144; +0,3998] |

Menor NLL é melhor. O Transformer teve NLL conjunta significativamente menor nos três splits. A vantagem one-step do ASM observada no P2 não se transferiu para esta previsão categórica joint H8 sob planos contrafactuais.

## Lead time validation-calibrated

Em ID, ASM detectou 29,12% dos episódios com evento, com lead médio 5,51 passos; Transformer detectou 32,16%, com lead 5,78. Em shift, ASM/Transformer detectaram 30,32%/33,71%. Em OOD, 89,68%/89,37%. Esses valores são diagnósticos; nenhum gate de intervenção foi avaliado.

## Interpretação

O teste decisivo não revelou a separação mecanística esperada. Quando obrigados a passar por suas próprias trajetórias previstas, ASM e Transformer permaneceram próximos na antecipação H8. Os pequenos deltas pontuais positivos do ASM são menores que o efeito registrado e compatíveis com zero. Ao mesmo tempo, o Transformer modelou melhor a trajetória física pela NLL conjunta.

Isso não prova que os mecanismos internos sejam iguais. Mostra que este protocolo não encontrou evidência de antecipação trajectory-grounded superior do ASM-X Base. A tese forte `estado → trajetória → previsibilidade` não foi confirmada para o ASM neste desenho. `TG4_causal_intervention` permanece não avaliado; não há claim de intervenção, safety ou superioridade geral.

## Limitações

- O risco é específico aos planos open-loop registrados, não a qualquer policy.
- Os targets físicos privilegiados são válidos no simulador, mas não implicam transferência ao mundo real.
- O decoder categórico e a discretização podem dominar parte da dificuldade.
- K=256 introduz erro Monte Carlo, ainda que pareado.
- A prevalência alta eleva a AUPRC absoluta; comparações com P2 devem usar prevalência/lift.
- Um teste de intervenção ainda exigiria clones `do(action)`, redução de unsafe e limite de perda de utilidade, mas não deve ser promovido automaticamente porque TG2 falhou.

## Desvio de reporting

Depois da abertura do test, o leitor de summary procurou nomes planos, enquanto o runner selado havia gravado subdiretórios por split. Nenhum checkpoint, prediction, score ou código selado foi alterado. O summarizer pré-selado e inalterado rodou sobre links temporários read-only para os mesmos 40 arquivos; a verificação do prediction manifest passou. O workaround de I/O está registrado em `reporting_compatibility_manifest.json`. A integridade preditiva foi preservada, mas o desvio permanece como limitação operacional do reporting.

## Artefatos principais

- `docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/index.html`;
- `trajectory_grounded_anticipation.png/.svg`;
- `quality_by_horizon.png/.svg`;
- `paired_deltas.png/.svg`;
- `summary.json` e `README.md`.

## Modified files

- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_checkpoint.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_checkpoint.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_dataset.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_dataset.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_evaluation.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_evaluation.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_head.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_head.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_manifests.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_manifests.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_models.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_models.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_plans.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_plans.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_plots.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_plots.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_predicate.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_predicate.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_protocol_io.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_protocol_io.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_runner.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_runner.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_runtime.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_runtime.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_seal.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_seal.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_statistics.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_statistics.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_summary.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_summary.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_test_runner.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_test_runner.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_train_runner.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_train_runner.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_training.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_training.py)
- [src/aletheion_state_models/benchmarks/transition_risk/trajectory_types.py](../../src/aletheion_state_models/benchmarks/transition_risk/trajectory_types.py)
- [tests/test_transition_risk_trajectory.py](../../tests/test_transition_risk_trajectory.py)
- [tests/test_transition_risk_trajectory_causality.py](../../tests/test_transition_risk_trajectory_causality.py)
- [tests/test_transition_risk_trajectory_dataset.py](../../tests/test_transition_risk_trajectory_dataset.py)
- [tests/test_transition_risk_trajectory_plans.py](../../tests/test_transition_risk_trajectory_plans.py)
- [tests/test_transition_risk_trajectory_predicate.py](../../tests/test_transition_risk_trajectory_predicate.py)
- [tests/test_transition_risk_trajectory_runtime.py](../../tests/test_transition_risk_trajectory_runtime.py)
- [tests/test_transition_risk_trajectory_seal.py](../../tests/test_transition_risk_trajectory_seal.py)
- [tests/test_transition_risk_trajectory_tg1.py](../../tests/test_transition_risk_trajectory_tg1.py)
- [tests/test_transition_risk_trajectory_types.py](../../tests/test_transition_risk_trajectory_types.py)
- [scripts/run_attr_trajectory_grounded.py](../../scripts/run_attr_trajectory_grounded.py)
- [docs/ATTR_TG1_TRAJECTORY_GROUNDED_PROTOCOL.md](../ATTR_TG1_TRAJECTORY_GROUNDED_PROTOCOL.md)
- [docs/ATTR_TG1_TRAJECTORY_GROUNDED_PROTOCOL_ptbr.md](../ATTR_TG1_TRAJECTORY_GROUNDED_PROTOCOL_ptbr.md)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/README.md](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/index.html](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/index.html)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/paired_deltas.png](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/paired_deltas.png)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/paired_deltas.svg](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/paired_deltas.svg)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/quality_by_horizon.png](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/quality_by_horizon.png)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/quality_by_horizon.svg](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/quality_by_horizon.svg)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/reporting_compatibility_manifest.json](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/reporting_compatibility_manifest.json)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/summary.json](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/trajectory_grounded_anticipation.png](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/trajectory_grounded_anticipation.png)
- [docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/trajectory_grounded_anticipation.svg](../benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/trajectory_grounded_anticipation.svg)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_107__asm_x_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_107__asm_x_base.pt)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_107__transformer_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_107__transformer_base.pt)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_29__asm_x_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_29__asm_x_base.pt)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_29__transformer_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_29__transformer_base.pt)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_43__asm_x_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_43__asm_x_base.pt)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_43__transformer_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_43__transformer_base.pt)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_71__asm_x_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_71__asm_x_base.pt)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_71__transformer_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_71__transformer_base.pt)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_89__asm_x_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_89__asm_x_base.pt)
- [runs/attr_trajectory_grounded_tg1/checkpoints/seed_89__transformer_base.pt](../../runs/attr_trajectory_grounded_tg1/checkpoints/seed_89__transformer_base.pt)
- [runs/attr_trajectory_grounded_tg1/data/protocol.manifest.json](../../runs/attr_trajectory_grounded_tg1/data/protocol.manifest.json)
- [runs/attr_trajectory_grounded_tg1/data/train.manifest.json](../../runs/attr_trajectory_grounded_tg1/data/train.manifest.json)
- [runs/attr_trajectory_grounded_tg1/data/validation.manifest.json](../../runs/attr_trajectory_grounded_tg1/data/validation.manifest.json)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_107__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_107__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_107__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_107__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_29__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_29__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_29__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_29__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_43__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_43__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_43__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_43__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_71__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_71__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_71__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_71__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_89__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_89__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_89__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_id/seed_89__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_107__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_107__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_107__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_107__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_29__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_29__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_29__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_29__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_43__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_43__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_43__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_43__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_71__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_71__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_71__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_71__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_89__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_89__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_89__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_ood/seed_89__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_107__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_107__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_107__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_107__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_29__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_29__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_29__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_29__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_43__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_43__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_43__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_43__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_71__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_71__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_71__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_71__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_89__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_89__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_89__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/test_shift/seed_89__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_107__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_107__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_107__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_107__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_29__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_29__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_29__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_29__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_43__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_43__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_43__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_43__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_71__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_71__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_71__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_71__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_89__asm_x_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_89__asm_x_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/predictions/validation/seed_89__transformer_base.jsonl](../../runs/attr_trajectory_grounded_tg1/predictions/validation/seed_89__transformer_base.jsonl)
- [runs/attr_trajectory_grounded_tg1/results/seed_107__asm_x_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_107__asm_x_base.json)
- [runs/attr_trajectory_grounded_tg1/results/seed_107__transformer_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_107__transformer_base.json)
- [runs/attr_trajectory_grounded_tg1/results/seed_29__asm_x_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_29__asm_x_base.json)
- [runs/attr_trajectory_grounded_tg1/results/seed_29__transformer_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_29__transformer_base.json)
- [runs/attr_trajectory_grounded_tg1/results/seed_43__asm_x_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_43__asm_x_base.json)
- [runs/attr_trajectory_grounded_tg1/results/seed_43__transformer_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_43__transformer_base.json)
- [runs/attr_trajectory_grounded_tg1/results/seed_71__asm_x_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_71__asm_x_base.json)
- [runs/attr_trajectory_grounded_tg1/results/seed_71__transformer_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_71__transformer_base.json)
- [runs/attr_trajectory_grounded_tg1/results/seed_89__asm_x_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_89__asm_x_base.json)
- [runs/attr_trajectory_grounded_tg1/results/seed_89__transformer_base.json](../../runs/attr_trajectory_grounded_tg1/results/seed_89__transformer_base.json)
- [runs/attr_trajectory_grounded_tg1/trajectory_checkpoint_seal.json](../../runs/attr_trajectory_grounded_tg1/trajectory_checkpoint_seal.json)
- [runs/attr_trajectory_grounded_tg1/trajectory_checkpoint_seal.json.opened](../../runs/attr_trajectory_grounded_tg1/trajectory_checkpoint_seal.json.opened)
- [runs/attr_trajectory_grounded_tg1/trajectory_prediction_manifest.json](../../runs/attr_trajectory_grounded_tg1/trajectory_prediction_manifest.json)
- [runs/attr_trajectory_grounded_tg1/trajectory_protocol_preseal.json](../../runs/attr_trajectory_grounded_tg1/trajectory_protocol_preseal.json)
- [runs/attr_trajectory_grounded_tg1/trajectory_reporting_compatibility_manifest.json](../../runs/attr_trajectory_grounded_tg1/trajectory_reporting_compatibility_manifest.json)
- [runs/attr_trajectory_grounded_tg1/trajectory_test_open_event.json](../../runs/attr_trajectory_grounded_tg1/trajectory_test_open_event.json)
- [docs/report/0037_attr-tg1-antecipacao-fundamentada-trajetoria_2026-09-01.md](0037_attr-tg1-antecipacao-fundamentada-trajetoria_2026-09-01.md)

## Changes

- Implementei dataset, planos causais, decoder físico autoregressivo, predicado unsafe fixo e loss sem hazard classifier.
- Implementei preseal, checkpoint seal, avaliação K256, bootstrap pareado e dashboard separado.
- Treinei e avaliei ASM-X Base e Transformer em 2×5 seeds e três tests frescos.
- Registrei o mismatch de caminhos do reporting e usei somente links read-only para os mesmos arquivos hashados.

## Validation

- 77 testes transition-risk — passaram; quatro warnings conhecidos do Transformer
- ruff check nos módulos/scripts/testes trajectory-grounded — passou
- compileall transition_risk e CLI — passou
- git diff --check — passou
- Preseal pós-avaliação — verificado sem mudança
- 10 checkpoints, 10 validation JSONL, 30 test JSONL e prediction manifest — verificados
- 3 PNGs abertos, 3 SVGs parseados, summary JSON finito e dashboard com 6 links válidos
- Auditoria modular — todos os novos módulos <=300 linhas; quatro violações >500 permanecem preexistentes
