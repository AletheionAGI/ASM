# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Explicar por que o Transformer pode ter AUPRC H8 competitiva mesmo tendo NLL de próximo estado muito pior, verificar se sealed_metrics tem erro e tornar a relação visível no dashboard.

## Summary

## Resposta direta

O gráfico `sealed_metrics` está lendo os valores corretos, mas AUPRC H8 **não é calculada a partir da NLL** neste benchmark. A `NextStateHead` e a `HazardHead` são projeções separadas da mesma representação. A primeira prevê uma Gaussiana diagonal para o próximo estado de seis dimensões e recebe NLL em todos os passos. A segunda produz diretamente logits para “entrada unsafe dentro de H” e sua AUPRC depende do ranking desses logits. Não existe uma equação no P2 que obrigue NLL menor a gerar AUPRC maior.

Isso permite ao Transformer reconhecer pistas de risco de baixa dimensão — energia, proximidade, tendência recente ou padrões correlacionados — e ordenar alguns eventos de hazard sem reconstruir corretamente toda a distribuição do próximo estado. AUPRC também é invariável a transformações monotônicas do score, enquanto NLL pune erro de média e escala de incerteza em todas as coordenadas, sobretudo nos muitos passos seguros.

## O que os números mostram

Há alguma relação geral: entre os seis braços no ID, a correlação descritiva de Spearman entre NLL e AUPRC é -0,829, ou seja, NLL menor costuma acompanhar AUPRC maior. Porém, o par ASM-X Base/Transformer é uma exceção importante: ASM-X Base tem NLL 2,6036 contra 3,5993 do Transformer, mas AUPRC H8 0,1505 contra 0,1498.

“Competitivo” não significa que o Transformer esteja antecipando bem. Com prevalência H8 de 0,1283, sua AUPRC 0,1498 é somente 1,168× o baseline de prevalência; ASM-X Base chega a 1,173×. Os dois classificadores diretos são fracos. O Transformer apenas empata com um ASM-X que também não conseguiu transformar sua vantagem de dinâmica em ranking de hazard melhor.

## Limitação revelada

A intuição do usuário identifica uma limitação real do desenho: o P2 testa se a representação suporta uma head direta de hazard, não se a melhor previsão de dinâmica **causa** a antecipação. Para testar essa hipótese forte, o risco deve ser derivado de previsões multi-horizonte de estado sob um predicado unsafe externo e fixo, ou deve haver uma análise explícita de mediação entre a head de dinâmica e a head de hazard.

Portanto, não há bug nos dados do gráfico, mas o gráfico anterior não deixava o desacoplamento evidente. Foi adicionado `dynamics_vs_anticipation.png/.svg`, com NLL no eixo x e AUPRC H8 no eixo y para ID, shift e OOD, e a limitação foi documentada no protocolo e no README P2.

## Modified files

- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/README.md](../benchmarks/asm_transformer_transition_risk/p2/README.md)
- [docs/benchmarks/asm_transformer_transition_risk/p2/index.html](../benchmarks/asm_transformer_transition_risk/p2/index.html)
- [docs/benchmarks/asm_transformer_transition_risk/p2/dynamics_vs_anticipation.png](../benchmarks/asm_transformer_transition_risk/p2/dynamics_vs_anticipation.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/dynamics_vs_anticipation.svg](../benchmarks/asm_transformer_transition_risk/p2/dynamics_vs_anticipation.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/summary.json](../benchmarks/asm_transformer_transition_risk/p2/summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_summary.json](../benchmarks/asm_transformer_transition_risk/p2/risk_mass_extension_summary.json)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg](../benchmarks/asm_transformer_transition_risk/p2/sealed_metrics.svg)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.png](../benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.png)
- [docs/benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.svg](../benchmarks/asm_transformer_transition_risk/p2/test_id_multiseed.svg)
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
- [docs/report/0032_explicacao-nll-versus-auprc-h8_2026-09-01.md](0032_explicacao-nll-versus-auprc-h8_2026-09-01.md)

## Changes

- Auditei a extração de sealed_metrics e confirmei que AUPRC H8 e NLL vêm dos campos canônicos corretos.
- Documentei que HazardHead e NextStateHead são heads separadas e que P2 não impõe acoplamento entre as métricas.
- Calculei correlação descritiva NLL×AUPRC nos seis braços e normalizei AUPRC pela prevalência H8.
- Adicionei scatter NLL × AUPRC para ID, shift e OOD ao dashboard.

## Validation

- 42 testes transition-risk — passaram; três warnings conhecidos do Transformer
- ruff check p2_plots.py — passou
- compileall p2_plots.py — passou
- git diff --check — passou
- Dashboard — 23 links válidos
- dynamics_vs_anticipation.png aberto em 2673×855; SVG parseado com sucesso
