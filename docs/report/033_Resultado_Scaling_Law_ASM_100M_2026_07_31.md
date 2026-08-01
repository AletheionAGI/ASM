# Resultado da scaling law ASM até 100M tokens

**Data:** 31 de julho de 2026
**Execução:** `runs/asm_scaling_law_100m_seed1`
**Artefatos versionados:** `docs/benchmarks/asm_scaling_law_100m_seed1/`
**Commit avaliado:** `2c56203ea973c68feca38d54f60bbc7ed49717bb`

## Resumo executivo

A scaling law comparou quatro arquiteturas sobre marcos contínuos entre 1M e 100M
tokens. Todos os checkpoints foram avaliados sobre a mesma sequência congelada de
4.834.787 tokens de validação.

O resultado separa dois objetivos:

- **ASM-R (`J_NO_DIRECTION`) venceu por tokens**, com CE 1,344849 em 100M;
- **ASM-S (`J_DIRECT_CONTROL_MATCHED`) venceu por tempo**, concluindo 100M em
  aproximadamente 48 minutos e mais de duas vezes o throughput de ASM-R.

A variante direcional reformulada ASM-F terminou apenas 0,001197 CE atrás de ASM-R,
mas utilizou aproximadamente 43 milhões de parâmetros adicionais. O DRM explícito
original ASM-X também permaneceu competitivo, porém foi dominado por ASM-F em
qualidade e velocidade no marco final.

A conclusão provisória é:

> A métrica relacional sobreviveu à seleção arquitetural, mas o campo direcional
> explícito ainda não demonstrou benefício líquido para entropia cruzada nesta
> escala.

## Protocolo

| Item | Valor |
|---|---|
| Dataset de treino | Wikipedia, memmap byte-token |
| Manifesto de validação | `data/benchmark_125m_wikipedia/validation/manifest.json` |
| SHA-256 do manifesto | `4adabd5a6a64c30bda37ec23fd2db0341421995f6cdf516f46757e49c1948c07` |
| Seed | 1 |
| Comprimento de sequência | 512 |
| Batch físico | 2 |
| Acumulação de gradiente | 8 |
| Precisão | BF16 |
| Otimizador | AdamW, LR 3e-4, weight decay 0,01 |
| Hardware | NVIDIA RTX 4090 |
| Marcos | 1M, 2M, 5M, 10M, 20M, 30M, 50M e 100M |
| Tokens congelados de validação | 4.834.787 por checkpoint |

As quatro variantes foram treinadas sequencialmente e todos os marcos foram
rescoreados sobre o mesmo fluxo de validação. Isso elimina o viés do
`best_val_ce`, que seleciona mínimos sobre amostras diferentes.

## Arquiteturas

### ASM-R — `J_NO_DIRECTION`

Remove o catálogo explícito de direções e o fluxo restrito. Preserva transição
contextual direta, métrica relacional, naturalização, memória seletiva, mixer e
emitter.

Parâmetros: **83.206.400**.

### ASM-X — `J`

Implementação DRM explícita com campo direcional, gates, fluxo, métrica relacional,
naturalização, memória seletiva e mixer.

Parâmetros: **126.080.896**.

### ASM-F — `J_METRIC_ORTHONORMAL_DIRECTION`

Reformula a composição direcional: a base é ortonormalizada na métrica relacional
antes da composição do movimento.

Parâmetros: **126.080.896**.

### ASM-S — `J_DIRECT_CONTROL_MATCHED`

Controle direto sem direção explícita, métrica ou naturalização. Redistribui a
capacidade para a transição e memória seletiva.

Parâmetros: **83.206.700**.

## CE congelado por marco

| Tokens | ASM-R | ASM-X | ASM-F | ASM-S |
|---:|---:|---:|---:|---:|
| 1M | 2,231452 | 2,294840 | 2,297790 | **2,125045** |
| 2M | 2,010348 | 2,049713 | 2,055631 | **1,931247** |
| 5M | 1,750542 | 1,760782 | 1,761830 | **1,730209** |
| 10M | **1,612410** | 1,613735 | 1,620511 | 1,615035 |
| 20M | **1,513506** | 1,514428 | 1,518115 | 1,521634 |
| 30M | 1,465237 | **1,464950** | 1,468502 | 1,477225 |
| 50M | **1,413747** | 1,414180 | 1,416346 | 1,428362 |
| 100M | **1,344849** | 1,347103 | 1,346046 | 1,358291 |

## Resultado final

| Posição por CE | Variante | CE | PPL | Horas de treino | Tokens/s aproximados |
|---:|---|---:|---:|---:|---:|
| 1 | **ASM-R** | **1,344849** | **3,8376** | 1,6762 | 16.572 |
| 2 | ASM-F | 1,346046 | 3,8422 | 1,6145 | 17.205 |
| 3 | ASM-X | 1,347103 | 3,8463 | 1,9080 | 14.559 |
| 4 | ASM-S | 1,358291 | 3,8895 | **0,7965** | **34.873** |

Diferença de CE em relação a ASM-R:

- ASM-F: +0,001197;
- ASM-X: +0,002255;
- ASM-S: +0,013442.

## Interpretação por tokens

ASM-S liderou de 1M a 5M tokens, demonstrando a melhor velocidade inicial de
otimização. ASM-R assumiu a liderança aproximadamente em 10M. A partir desse ponto,
ASM-R e ASM-X permaneceram quase empatadas, com uma inversão mínima em 30M, e ASM-R
terminou na frente em 50M e 100M.

As duas inversões registradas entre ASM-R e ASM-X nos intervalos 20–30M e 30–50M
não devem ser interpretadas como dois crossovers estruturais. As diferenças são da
ordem de décimos de milésimo e são compatíveis com oscilações ao redor de um quase
empate.

O crossover robusto é entre ASM-S e as variantes relacionais: a arquitetura direta
aprende mais depressa no início, mas apresenta melhora mais lenta por token após
aproximadamente 10M–15M tokens.

## Interpretação por tempo de GPU

ASM-S processou 100M tokens em 0,7965 hora. No mesmo orçamento aproximado, ASM-R e
ASM-F alcançaram apenas cerca de 50M tokens:

- ASM-S, 100M: CE 1,358291;
- ASM-R, 50M: CE 1,413747;
- ASM-F, 50M: CE 1,416346.

Portanto, ASM-S é claramente superior quando o orçamento é medido por tempo de uma
RTX 4090. O experimento produz uma fronteira de Pareto, e não um vencedor universal:

- ASM-R: melhor qualidade por token;
- ASM-S: melhor qualidade por hora;
- ASM-F: variante explícita geométrica mais competitiva.

## Ajustes de power law

O ajuste utilizou:

```text
L(N) = L_inf + A N^(-alpha)
```

| Variante | L_inf ajustado | A | alpha | Erro quadrático |
|---|---:|---:|---:|---:|
| ASM-R | 1,133454 | 150,3712 | 0,355714 | 0,0003777 |
| ASM-X | 1,165193 | 264,6799 | 0,394486 | 0,0006339 |
| ASM-F | 1,156140 | 239,2889 | 0,386414 | 0,0007324 |
| ASM-S | 1,113412 | 66,7105 | 0,303245 | **0,0000974** |

ASM-S possui o melhor ajuste e a menor assíntota estimada, mas também o menor
expoente. Extrapolar essas curvas sugeriria uma possível recuperação de ASM-S apenas
muito além do intervalo observado. Essa extrapolação não é evidência: `L_inf`, `A`
e `alpha` são correlacionados, há somente oito marcos e uma única seed.

Os parâmetros ajustados servem para descrever os dados entre 1M e 100M; não devem
ser tratados como previsão confirmada para bilhões de tokens.

## Consequência para o DRM

O resultado não demonstra que toda geometria relacional é inútil. ASM-R, a melhor
variante por tokens, ainda contém métrica e naturalização. O que não se justificou
foi a fatoração explícita:

```text
estado → catálogo de direções → gates → coeficientes → movimento
```

ASM-F mostra que combinar direção e métrica de maneira coerente recupera quase toda
a diferença. Mesmo assim, ela utiliza mais parâmetros e não supera ASM-R.

Assim, a hipótese mais sustentada pelos dados atuais é:

> Uma transição contextual implícita, condicionada por geometria relacional, é mais
> eficiente por parâmetro do que um campo direcional explicitamente fatorado.

## Decisão arquitetural provisória

1. Promover provisoriamente **ASM-R — Relational State Model** como arquitetura
   principal orientada a qualidade por token.
2. Manter **ASM-S — Selective State Model** como variante de alta eficiência.
3. Manter **ASM-F — Relational Frame Model** como linha geométrica experimental.
4. Despromover ASM-X como arquitetura principal; ASM-F oferece uma formulação
   direcional mais eficiente e ligeiramente melhor no marco final.

A promoção é provisória porque a curva até 100M usa apenas seed 1. Entretanto, sua
direção é consistente com as ablações pareadas anteriores.

## Próximas ações

1. Preservar os artefatos leves, hashes e configurações desta execução.
2. Não usar `best_val_ce` amostrado para selecionar arquitetura.
3. Executar o roadmap de correção descrito no relatório 032.
4. Corrigir paridade entre treino e geração antes de publicar checkpoints para uso.
5. Após as correções, confirmar que o rescoring de checkpoints históricos não
   apresenta regressão.
6. Planejar confirmação multiseed seletiva de ASM-R versus ASM-F e ASM-S somente
   quando houver orçamento experimental.

## Limitações

- Uma única seed foi executada até 100M.
- O estudo usa um único dataset, tokenizer byte-level e comprimento 512.
- CE não mede sozinho contexto longo, associative recall, controlabilidade ou
  qualidade de geração.
- Os tempos são específicos da implementação e da RTX 4090 usada.
- O caminho atual de geração possui divergências arquiteturais documentadas na
  auditoria 031 e ainda precisa ser corrigido.

## Proveniência

Os dados completos, hashes, ambiente, configurações e comandos estão em:

- `docs/benchmarks/asm_scaling_law_100m_seed1/scaling_law_summary.json`;
- `docs/benchmarks/asm_scaling_law_100m_seed1/ablation_manifest.json`;
- `docs/benchmarks/asm_scaling_law_100m_seed1/configs/`;
- `docs/benchmarks/asm_scaling_law_100m_seed1/training_summaries/`.

Os checkpoints grandes permanecem no diretório local ignorado `runs/`.
