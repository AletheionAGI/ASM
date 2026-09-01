# Resultado da ablação geométrica e controle direto

Data: 31 de julho de 2026

## 1. Resultado observado

As variantes J, J sem métrica, J sem campo direcional e J sem
naturalização foram treinadas com 5 milhões de tokens, três seeds pareadas e
rescoring sobre a mesma sequência contínua de 4.834.787 tokens de validação.

| Variante | CE médio | Desvio-padrão | CE mínimo | CE máximo |
|---|---:|---:|---:|---:|
| J | 1,760581 | 0,003057 | 1,756632 | 1,764080 |
| J_NO_METRIC | 1,762347 | 0,004216 | 1,758393 | 1,768189 |
| J_NO_DIRECTION | **1,751030** | **0,000477** | 1,750542 | 1,751677 |
| J_NO_NATURALIZATION | 1,762405 | 0,004284 | 1,758302 | 1,768318 |

`J_NO_DIRECTION` melhorou o CE médio em 0,009551 contra J, venceu nas três
seeds pareadas e apresentou a menor dispersão. Pelo gate atual — ganho mínimo
de 0,005, pelo menos duas vitórias em três seeds e desvio-padrão máximo de
0,03 — ela supera J.

J melhorou apenas 0,001766 contra `J_NO_METRIC` e 0,001824 contra
`J_NO_NATURALIZATION`. Esses ganhos ficam abaixo do limiar de promoção. Além
disso, J perdeu para `J_NO_METRIC` na seed 1.

## 2. Interpretação

O resultado não invalida toda a arquitetura DRM, mas rejeita a promoção da J
completa no protocolo atual. A decomposição do fluxo em direções aprendidas
prejudicou a eficiência amostral em 5 milhões de tokens. Uma transição neural
causal direta produziu CE menor e mais estável.

Não há evidência suficiente de contribuição positiva da métrica ou da
naturalização. Como as perdas geométricas auxiliares estão zeradas, a métrica
afeta o CE somente pela naturalização. A proximidade entre `J_NO_METRIC` e
`J_NO_NATURALIZATION` é, portanto, esperada.

A vantagem anterior de J contra `SSM_CONTROL` continua mostrando que algum
componente além da memória seletiva estreita é útil. Ela não demonstra que o
campo direcional seja esse componente.

## 3. Experimento decisivo seguinte

Foram acrescentados dois controles:

| Variante | Composição | Parâmetros esperados |
|---|---|---:|
| J_NO_DIRECTION | transição direta + métrica/naturalização + memória | 83.206.400 |
| J_DIRECT_CONTROL | transição direta + memória, sem métrica/naturalização | 29.777.664 |
| J_DIRECT_CONTROL_MATCHED | controle direto sem geometria, com orçamento redistribuído | 83.206.700 |

`J_DIRECT_CONTROL` é a ablação estrutural pura: remove a métrica sem mudar os
módulos restantes. Se empatar ou vencer mesmo com muito menos parâmetros, a
métrica não justifica seu custo.

`J_DIRECT_CONTROL_MATCHED` redistribui o orçamento liberado para a memória
seletiva. Sua diferença para `J_NO_DIRECTION` é de apenas 300 parâmetros. Ele
responde qual arquitetura utiliza melhor aproximadamente o mesmo orçamento,
embora a redistribuição impeça atribuir o resultado exclusivamente à remoção
da métrica.

## 4. Critérios de interpretação

- `J_NO_DIRECTION` vence os dois controles: existe evidência de contribuição
  da métrica/naturalização sobre a transição direta.
- `J_DIRECT_CONTROL` empata ou vence: a métrica é dispensável mesmo sem
  redistribuir seus parâmetros.
- Somente `J_DIRECT_CONTROL_MATCHED` vence: o orçamento é mais útil na memória
  seletiva, mas ainda é necessário separar efeito arquitetural de capacidade.
- Os controles diretos vencem `SSM_CONTROL`: a transição causal direta agrega
  capacidade além de mixer, residual, memória seletiva e emitter.

O resultado deve continuar sendo descrito como evidência para CE nessa escala,
e não como prova geral a favor ou contra a hipótese geométrica.

## 5. Execução completa

O launcher executa:

1. suíte de testes;
2. forward smoke das três variantes;
3. treinamento de 5 milhões de tokens nas seeds 1, 2 e 3;
4. rescoring contínuo idêntico da validação;
5. agregação de média e desvio-padrão;
6. gates dos dois controles contra `J_NO_DIRECTION`.

Comando:

```bash
chmod +x scripts/run_drm_direct_control_suite.sh
./scripts/run_drm_direct_control_suite.sh
```

Os resultados serão gravados em:

```text
runs/drm_direct_control_ablation_5m/
```

O arquivo agregado será:

```text
runs/drm_direct_control_ablation_5m/paired_validation_summary.json
```

As decisões automáticas serão gravadas em:

```text
runs/drm_direct_control_ablation_5m/decision_j_direct_control_vs_j_no_direction.json
runs/drm_direct_control_ablation_5m/decision_j_direct_control_matched_vs_j_no_direction.json
```

## 6. Estimativa de duração

Na RTX 4090 medida, `J_NO_DIRECTION` treinou cada seed em aproximadamente
5,2 minutos. O rescoring levou aproximadamente 65 segundos por seed. Os
controles sem métrica devem ser mais rápidos, mas a versão com parâmetros
redistribuídos pode recuperar parte desse custo.

A estimativa operacional para a suíte completa é de **35 a 50 minutos**:

- testes e smokes: 1 a 3 minutos;
- nove treinamentos: aproximadamente 25 a 37 minutos;
- nove rescoring completos: aproximadamente 5 a 8 minutos;
- agregação e gates: menos de 1 minuto.

Essa faixa pressupõe GPU livre, dados já preparados e ausência de throttling.
O primeiro log com `tokens_per_sec` permite recalibrar a previsão.
