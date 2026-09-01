# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Traduzir o teste ATTR P2 concluído para linguagem humana.

## Summary

Expliquei em linguagem simples que o benchmark mede se os modelos conseguem antecipar perigo antes que ele aconteça, comparando previsão de risco e próximo estado em cenários normais, alterados e fora de distribuição. Interpretei os gates: integridade e previsão de próximo estado passaram, mas ASM-X não demonstrou vantagem confiável de antecipação sobre o Transformer; robustez falhou e safety/causalidade não foram testadas.

## Em linguagem humana

O teste perguntou:

> **“Ao observar um sistema evoluindo, o modelo consegue perceber que algo perigoso está se formando antes que o problema aconteça?”**

É parecido com mostrar ao modelo os instrumentos de um veículo e perguntar:

- haverá perigo em 1, 4, 8 ou 16 passos?
- como estará o veículo no próximo instante?
- o alerta pode ser dado cedo e com poucos alarmes falsos?

### Como o teste foi feito

Seis modelos receberam:

- os mesmos cenários;
- o mesmo histórico disponível;
- o mesmo objetivo;
- o mesmo orçamento de treino.

Cada modelo foi treinado cinco vezes, com seeds diferentes. Isso verifica se o resultado é repetível, em vez de depender de sorte.

Depois, os modelos fizeram três provas:

1. **ID:** situações semelhantes ao treino;
2. **shift:** sensores e dinâmica um pouco alterados;
3. **OOD:** situações mais diferentes e difíceis.

As provas ficaram lacradas até os 30 modelos terminarem o treino. Portanto, ninguém pôde “estudar as respostas”.

### O que aconteceu

#### ASM-X versus Transformer

Na previsão antecipada de perigo, os dois ficaram praticamente empatados:

- ASM-X: AUPRC `0,1505`;
- Transformer: AUPRC `0,1498`.

A diferença foi somente `+0,0007`. A margem de incerteza vai de `-0,0340` a `+0,0214`.

Em termos humanos:

> **Não podemos afirmar que ASM-X seja melhor. A pequena diferença pode ser apenas variação experimental.**

Para passar, ASM-X precisava ganhar pelo menos `0,03`, com evidência estatística de que a vantagem era realmente positiva. Isso não aconteceu.

#### Onde ASM-X foi melhor

ASM-X previu muito melhor o próximo estado do sistema:

- ASM-X NLL: `2,60`;
- Transformer NLL: `3,60`;
- menor é melhor.

Ou seja:

> **ASM-X compreendeu melhor como o sistema provavelmente evoluiria no passo seguinte, mas isso não se transformou em uma vantagem comprovada para antecipar perigo.**

### Significado dos gates

- **G0 passou:** a prova foi executada de forma íntegra e selada.
- **G1 passou:** ASM-X previu adequadamente o próximo estado.
- **G2 falhou:** ASM-X não demonstrou vantagem confiável de alerta antecipado.
- **G3 não foi testado:** ainda não sabemos se seus alertas oferecem tempo útil para agir.
- **G4 não foi testado:** ainda não sabemos se agir com base nesses alertas realmente evita acidentes.
- **G5 falhou:** a direção do resultado não foi consistente o suficiente entre as cinco repetições e não havia um limite crítico OOD completo registrado.

### Sobre os outros modelos

ASM-VR-S fixed-32 e ASM-R tiveram as maiores AUPRCs agregadas no teste normal, próximas de `0,175`.

Mas essas eram comparações suplementares. Elas não substituem o confronto principal ASM-X versus Transformer. Além disso, as diferenças entre fixed-32 e full-64 ainda incluem zero na margem de incerteza.

Portanto:

> **Não há base para promover fixed-32 nem declarar um vencedor geral.**

### Conclusão simples

O resultado foi:

> **ASM-X demonstrou modelar melhor o próximo estado do sistema, mas essa vantagem de dinâmica ainda não se traduziu em uma representação que permita ao detector comum antecipar perigo significativamente melhor que no Transformer.**

Também não testamos ainda se seus alertas salvam o sistema na prática. Portanto, não podemos falar em segurança ou entendimento causal.

## Modified files

- [docs/report/0030_traducao-humana-teste-attr-p2_2026-09-01.md](0030_traducao-humana-teste-attr-p2_2026-09-01.md)

## Changes

- Tradução conceitual dos objetivos, execução e resultados do ATTR P2 sem alterar código ou dados.

## Validation

- Valores e gates conferidos em docs/benchmarks/asm_transformer_transition_risk/p2/summary.json.
