# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Explicar em linguagem humana o significado de H8=1 e H8=0 no ATTR P2 e no gráfico sealed_metrics corrigido.

## Summary

## Significado

Em cada instante `t`, H8 pergunta: **“O sistema fará uma nova entrada no estado unsafe em algum momento entre o próximo passo e os próximos oito passos?”**

- **H8=1:** sim, ocorrerá uma entrada unsafe dentro da janela `(t, t+8]`.
- **H8=0:** não ocorrerá entrada unsafe nos próximos oito passos. O perigo ainda pode acontecer depois dessa janela.

H8=1 não significa perigo exatamente no oitavo passo. Pode ocorrer em qualquer um dos passos `t+1` a `t+8`. Também não significa que o sistema já está unsafe no instante atual: o label registra uma futura transição de safe para unsafe. As janelas são cortadas no fim de cada episódio e nunca atravessam para outro episódio.

No `sealed_metrics`, `Next-state NLL | H8=1` é a NLL de próximo estado calculada somente nos instantes que estão dentro dessa janela de alerta anterior a um hazard. `Next-state NLL | H8=0` usa os demais instantes. H8 é usado apenas depois da inferência para separar os resultados; não entra como input do modelo e não é usado para produzir a previsão.

## Modified files

- [docs/report/0034_explicacao-labels-h8_2026-09-01.md](0034_explicacao-labels-h8_2026-09-01.md)

## Changes

- Expliquei a semântica temporal dos labels H8 e sua utilização somente como estratificação de avaliação.

## Validation

- Definição conferida em src/aletheion_state_models/benchmarks/transition_risk/labels.py.
