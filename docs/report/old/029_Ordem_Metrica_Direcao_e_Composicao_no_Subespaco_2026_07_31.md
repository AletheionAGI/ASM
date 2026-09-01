# Ordem métrica-direção e composição no subespaço

Data: 31 de julho de 2026

## 1. Motivação

A variante `J_NO_DIRECTION` superou J completa em 5 milhões de tokens. O
primeiro diagnóstico foi que o campo direcional explícito poderia representar
um gargalo. Uma segunda hipótese surgiu da própria filosofia DRM: talvez
direção e métrica estejam corretas isoladamente, mas compostas na ordem errada.

A sequência operacional original é:

```text
entrada → estado → direção → movimento bruto → naturalização métrica → saída
```

Em notação matricial:

$$
v_{\mathrm{raw}}=Vc,
\qquad
v=G^{-1}Vc.
$$

Em geral:

$$
G^{-1}Vc
\notin
\mathrm{span}(V).
$$

Portanto, a métrica pode retirar o movimento do espaço que o campo direcional
acabou de declarar como disponível. Isso enfraquece a interpretação de
direções como possibilidades efetivamente permitidas.

## 2. Por que uma simples troca não basta

Como $G^{-1}$ é linear:

$$
G^{-1}\sum_i c_iV_i
=
\sum_i c_iG^{-1}V_i.
$$

Aplicar a métrica antes ou depois da soma não produz uma nova arquitetura se
gates, coeficientes e direções permanecerem iguais. Para mudar a ordem de forma
substantiva, a métrica precisa participar da definição ou combinação das
direções.

## 3. Variante `J_METRIC_SUBSPACE`

As direções formam a matriz $V$ e a métrica induzida no espaço dos coeficientes
é:

$$
C=V^\top GV.
$$

Com damping $\lambda$, os coeficientes metricamente condicionados são:

$$
\hat c=(V^\top GV+\lambda I)^{-1}c.
$$

O movimento é:

$$
v=V\hat c.
$$

Por construção:

$$
v\in\mathrm{span}(V).
$$

A variante preserva gates, campo, fluxo, métrica, memória seletiva, mixer e
emitter de J. Apenas a regra de composição entre movimento e métrica muda.

## 4. Variante `J_METRIC_ORTHONORMAL_DIRECTION`

A segunda variante calcula a matriz de Gram regularizada:

$$
C_\lambda=V^\top GV+\lambda I.
$$

Se:

$$
C_\lambda=LL^\top,
$$

as direções são transformadas por:

$$
Q=VL^{-\top}.
$$

Com damping pequeno:

$$
Q^\top GQ\approx I.
$$

O movimento passa a ser formado sobre direções aproximadamente ortonormais na
geometria relacional:

$$
v=Qc.
$$

Essa variante testa se o problema era a redundância ou o condicionamento das
direções na métrica.

## 5. Implementação e estabilidade

Os cálculos de Gram, Cholesky e solve são realizados em FP32 quando o treino usa
BF16. O damping já configurado para a métrica regulariza sistemas quase
singulares.

A matriz de Gram é calculada uma vez por estado inicial de bloco, e não uma vez
por token. Isso evita duplicar o custo geométrico nos 64 tokens do bloco.

A intensidade das duas composições respeita o mesmo strength e warmup usados
pela naturalização original. Ambas preservam o subespaço direcional, inclusive
durante a interpolação com o movimento bruto.

## 6. Matriz experimental

| Variante | Hipótese |
|---|---|
| J | direção primeiro e naturalização posterior |
| J_METRIC_SUBSPACE | métrica atua nos coeficientes dentro do subespaço |
| J_METRIC_ORTHONORMAL_DIRECTION | métrica normaliza a própria base direcional |
| J_NO_DIRECTION | transição causal direta ainda com métrica |
| J_DIRECT_CONTROL_MATCHED | transição direta sem geometria, parâmetros pareados |

O protocolo usa 5 milhões de tokens, seeds 1, 2 e 3, mesmos manifests,
hiperparâmetros pareados e rescoring sobre a mesma validação contínua.

## 7. Critérios de decisão

- Se uma variante métrica-primeiro vencer J, a ordem antiga estava
  prejudicando o campo.
- Se também vencer `J_NO_DIRECTION`, direções geometricamente condicionadas
  recuperam utilidade prática.
- Se vencer `J_DIRECT_CONTROL_MATCHED`, haverá evidência de que a geometria
  oferece mais do que capacidade redistribuída.
- Se ambas perderem para os controles diretos, a fatoração direcional explícita
  continuará sem justificativa para CE nessa escala.
- Ganhos devem atingir pelo menos 0,005 CE, ocorrer em duas das três seeds e
  manter desvio-padrão máximo de 0,03.

## 8. Comando completo

```bash
chmod +x scripts/run_drm_metric_order_suite.sh
./scripts/run_drm_metric_order_suite.sh
```

O resultado agregado será gravado em:

```text
runs/drm_metric_order_ablation_5m/paired_validation_summary.json
```

O runner também grava decisões automáticas de cada nova variante contra J,
`J_NO_DIRECTION` e `J_DIRECT_CONTROL_MATCHED`.

## 9. Estimativa de duração

São quinze treinamentos de 5 milhões de tokens, seguidos de quinze rescoring
completos. As novas variantes acrescentam solves de matrizes pequenas por
bloco, em FP32.

Na RTX 4090, a estimativa inicial é de **2 a 3 horas**. O tempo poderá cair se o
custo dos solves ficar oculto pelos demais módulos ou subir se as fatorações
reduzirem significativamente o throughput. Os primeiros 100 passos de
`J_METRIC_SUBSPACE` permitirão recalibrar a estimativa.

## 10. Interpretação científica

O experimento não tenta salvar o campo direcional a qualquer custo. Ele testa
uma crítica interna legítima: uma geometria relacional deveria organizar as
possibilidades antes do movimento, e não apenas corrigir um vetor depois que ele
foi decidido.

Se a hipótese falhar, o resultado fortalecerá a simplificação arquitetural. Se
funcionar, indicará que o problema estava na composição, não necessariamente na
ideia de direções relacionais.
