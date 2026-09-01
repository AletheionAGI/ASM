# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Interpretar e formalizar a proposta de retirar a hazard classifier e obrigar Transformer e ASM a antecipar perigo exclusivamente a partir de suas próprias trajetórias previstas.

## Summary

## Interpretação

A hipótese faz sentido. A AUPRC atual vem de uma `HazardHead` direta e independente da `NextStateHead`. Assim, Transformer e ASM podem obter classificação semelhante usando mecanismos distintos: um pode explorar pistas discriminativas ou proxies no histórico, enquanto o outro pode representar melhor a dinâmica sem conseguir transferi-la para a classifier.

## Experimento decisivo: trajetória fundamentada

O próximo protocolo deve remover completamente a hazard classifier. A partir do histórico causal disponível em `t`, cada modelo deve produzir uma distribuição de trajetórias futuras até H8. O score de risco será calculado exclusivamente por um predicado `unsafe` externo, fixo e não aprendido, aplicado às trajetórias previstas:

`risk_H8 = P(existe entrada unsafe em algum passo t+1...t+8 | histórico até t)`.

Nenhuma head poderá ler diretamente a representação para prever hazard. O único caminho permitido será representação → previsão multi-horizonte de estado → predicado unsafe → probabilidade de risco.

## Requisitos

- previsão explícita dos estados suficientes em H1/H4/H8, incluindo as variáveis necessárias para colisão, energia e falha atrasada;
- política ou plano de ações congelado e causalmente disponível em `t`, igual para os dois modelos;
- rollouts estocásticos pareados com common random numbers;
- mesma head de dinâmica, quantidade de amostras, orçamento e calibração validation-only;
- AUPRC, Brier e lead time calculados do risco derivado das trajetórias;
- NLL/RMSE por horizonte e por proximidade do hazard;
- direct hazard head preservada somente como controle histórico, nunca como caminho do braço principal.

## Decisão que o teste permite

Se o Transformer mantiver AUPRC competitiva também no risco derivado da trajetória, sua representação contém dinâmica útil apesar da NLL agregada pior, ou a NLL está sendo dominada por escala/coordenadas irrelevantes. Se sua AUPRC cair enquanto ASM retiver sinal, a classificação direta do Transformer provavelmente dependia de proxies discriminativos. Se ambos caírem, a head direta estava fazendo a maior parte do trabalho. Se ambos permanecerem semelhantes e fracos, a hipótese de superioridade antecipatória do ASM continua sem suporte.

## Integridade

Os checkpoints P2 atuais possuem somente previsão one-step e o test já foi observado. Não é válido fabricar AUPRC a partir da NLL realizada, pois isso usa o próximo estado verdadeiro. O experimento exige novo protocolo, heads multi-horizonte e um split de test fresco, selado antes do treino. Ele deve ser registrado como ATTR trajectory-grounded, não como reinterpretação retroativa do P2.

## Força e limite da evidência

Se o ASM abrir uma vantagem reproduzível nesse desenho, a evidência será muito mais próxima da tese mecanística **estado → trajetória → previsibilidade** do que a AUPRC de uma head direta. O predicado unsafe fixo elimina o atalho em que uma classifier aprende proxies de perigo sem depender da trajetória prevista. Uma dissociação em que a head direta empata, mas o risco derivado de trajetória favorece o ASM, seria especialmente informativa.

Ainda assim, esse resultado não fecha sozinho o último elo, **intervenção**. Ele mostraria que a representação do ASM produz previsões de trajetória mais úteis para localizar perigo. Para demonstrar intervenção, seria necessário um estágio posterior com simuladores clonados, `do(action)` versus controle, common random numbers, redução de unsafe e limite de degradação de utilidade. Portanto, o teste trajectory-grounded qualificaria a entrada no teste causal; não substituiria o teste causal.

## Modified files

- [docs/report/0035_proposta-attr-trajetoria-sem-hazard-classifier_2026-09-01.md](0035_proposta-attr-trajetoria-sem-hazard-classifier_2026-09-01.md)

## Changes

- Formalizei o contraste entre classificação direta por proxies e antecipação fundamentada em trajetória.
- Defini o caminho obrigatório representação → previsão multi-horizonte → predicado unsafe fixo → risco H8.
- Registrei a necessidade de um novo test fresco e selado.

## Validation

- Compatibilidade conceitual conferida com as limitações da NextStateHead e HazardHead atuais.
