# FAQ técnico e metodologia de benchmark

Este documento descreve a arquitetura atualmente implementada do DRM Language
Emitter e suas evidências experimentais. Ele distingue comportamento
verificado, resultados preliminares e questões de pesquisa em aberto.

Versão original em inglês: [TECHNICAL_QA.md](TECHNICAL_QA.md).

## O que está implementado?

O DRM é um emissor causal de linguagem baseado em dinâmica recorrente de
estado latente, não em attention Transformer.

O caminho recorrente básico é:

```text
input_ids
  → TokenEmbedding
  → estado inicial z_0
  → DirectionField(z_t) → direções, gates
  → RelationalMetric(z_t) → diag + U Uᵀ
  → DRMFlow(z_t, e_t, direções, gates) → dz_raw
  → metric.naturalize(dz_raw, diag, U)
  → StateUpdater(z_t, dz) → z_{t+1}
  → LanguageEmitter(z_{t+1}) → logits
```

O caminho experimental 125M mais forte usa blocos causais, soma cumulativa de
deltas direcionais e um mixer convolucional causal local. A implementação foi
modularizada em:

- `model.py`: montagem do modelo e forward recorrente principal;
- `model_components.py`: inicializador e local mixer;
- `directional_forward.py`: forward cumsum, losses e diagnósticos;
- `directional_blocks.py`: blocos e superblocos;
- `directional_solvers.py`: fixed point, Anderson e transições auxiliares;
- `geometric_steps.py`: bounding, candidatos e refinamento geodésico.

O modelo é autoregressivo e a loss principal é cross-entropy do próximo token.

O caminho 125M líder usa cumsum de velocidade causal em blocos de 64 tokens e
um mixer convolucional causal de duas camadas. Testes de causalidade garantem
que alterações em tokens futuros não mudam estados ou logits do prefixo.

## Como o DRM difere de um Transformer?

Um Transformer contextualiza tokens principalmente com projeções de attention
e relações token-token. O DRM atualiza um estado latente usando direções,
gates, uma métrica relacional e um fluxo.

```text
Transformer: embedding → attention/QKV → MLP → logits
DRM:         embedding → estado → direções → métrica → fluxo → emissor
```

Não é apenas uma diferença de nomes: o núcleo DRM não instancia blocos
Transformer ou camadas de attention.

## O núcleo usa attention ou QKV?

Não. O núcleo não usa self-attention, projeções Q/K/V,
`nn.MultiheadAttention` ou KV cache.

`tests/test_no_transformer.py` verifica que `DRMEmitterModel` não contém
`nn.MultiheadAttention` e que o pacote não define nomes usuais de projeções
QKV, como `q_proj`, `k_proj` e `v_proj`.

Transformers e modelos Hugging Face existem apenas como baselines.

## Qual é a equação operacional principal?

```text
z_{t+1} = z_t + dt * dz_t

dz_raw_t = Σ_i gates_i(z_t) * c_i(z_t, e_t) * direction_i(z_t)
dz_t = naturalize_G(dz_raw_t)

logits_t = Emitter(z_{t+1})
p(x_{t+1} | z_{t+1}) = softmax(logits_t)
```

Onde:

- `z_t` é o estado latente;
- `e_t` é o embedding do token atual;
- `direction_i(z_t)` é produzido por `DirectionField`;
- `gates_i(z_t)` controla direções ativas;
- `G(z_t)` é a métrica aprendida;
- `naturalize_G` aplica precondicionamento métrico;
- `Emitter` projeta o estado para logits do vocabulário.

## O que é a métrica relacional?

A métrica implementada é:

```text
G(z) = diag(softplus(d(z)) + eps) + U(z)U(z)ᵀ
```

A diagonal é estritamente positiva e `UUᵀ` é positiva semidefinida. Assim, a
métrica neural atual é positiva definida (SPD) sem materializar uma matriz
densa completa `d_state × d_state`.

O código também calcula:

```text
low_rank_scale = sum(U²)
upper = max(diag) + low_rank_scale
lower = max(min(diag), 1e-8)
condition_proxy = upper / lower
```

Isso é um proxy numérico, não um cálculo exato do espectro.

## A métrica atual implementa o rank formal do paper?

Não. Como `softplus(d) + eps > 0`, a métrica tem rank matemático completo e:

```text
Ker(G) = {0}
rank(G) = d_state
```

O diagnóstico atual `dimD` é a soma dos gates direcionais. Ele mede atividade
direcional, não:

```text
d_DRM(p) = rank(g_p)
```

definido em `docs/paper/drm_v6.tex`.

Um rank numérico por tolerância pode aproximar dimensionalidade espectral, mas
deve ser rotulado como aproximação. Um kernel formal exige uma parametrização
PSD realmente degenerável.

## O que significa “direção”?

É um vetor aprendido no espaço latente. `DirectionField(z)` produz direções e
gates sigmoid; o fluxo se move no span das direções ativas.

Direção não significa ética, intenção ou alignment. É uma variável operacional
da atualização neural.

## Quais diagnósticos existem?

- `action_mean`: energia métrica média do movimento;
- `dimD_mean`: soma média dos gates;
- `soft_active_fraction`: `dimD_mean / n_directions`;
- frações ativas em diferentes thresholds;
- `condition_proxy`;
- `metric_U_norm_mean`;
- `recurrence_proxy`;
- `stability_proxy`;
- diagnósticos do risk field, quando habilitado.

Esses valores não são medições psicológicas ou semânticas. Sua utilidade exige
análises de estabilidade, correlação, intervenção e ablação.

## Qual é o benchmark histórico 36M/37M?

```text
docs/benchmarks/bench_36M/
```

Ele compara famílias com aproximadamente 37 milhões de parâmetros:

| Label | Família | Parâmetros | Seeds | Tokens por seed |
|---|---|---:|---|---:|
| `drm_36M` | DRM | 37.253.702 | 1, 2, 3 | 2.048.000 |
| `gpt2_36M` | GPT-2 | 36.915.984 | 1, 2, 3 | 2.048.000 |
| `opt_36M` | OPT | 36.916.992 | 1, 2, 3 | 2.048.000 |

Labels internos legados ainda contêm `125m`, mas não representam o número real
de parâmetros desse experimento. Ele não é mais o benchmark grande mais
recente.

## Qual dataset e tokenizer foram usados no benchmark 36M?

| Item | Valor |
|---|---|
| Dataset | `wikimedia/wikipedia` |
| Config | `20231101.en` |
| Split original | `train` |
| Carregamento | streaming |
| Amostra escrita | 50.000.002 caracteres |
| Documentos | 2.272 |
| Tamanho mínimo | 200 caracteres |
| Arquivo | `data/wikipedia_en_20231101_sample.txt` |
| Tokenizer | byte-level |
| Vocabulário efetivo | 256 |

O split desse benchmark antigo foi feito sobre a sequência tokenizada:
aproximadamente 90% para treino e a parte final para validação.

## Por que tokenização byte-level?

Ela evita depender de um vocabulário aprendido e cobre texto arbitrário com
256 símbolos. Também permite que DRM, GPT-2 e OPT usem o mesmo vocabulário
efetivo.

Como desvantagem, bytes normalmente produzem sequências mais longas que
subwords, alterando custo, contexto efetivo e qualidade linguística. É uma
escolha controlada, não uma alegação de que bytes são ideais em escala.

## Quantos tokens cada modelo 36M processou?

```text
steps * grad_accum_steps * batch_size * seq_len
= 1000 * 1 * 4 * 512
= 2.048.000 tokens
```

Cada família processou 6.144.000 tokens somando as três seeds.

## Qual foi o protocolo 36M?

| Item | Valor |
|---|---:|
| Steps | 1.000 |
| Batch size | 4 |
| Gradient accumulation | 1 |
| Sequence length | 512 |
| Learning rate | 3e-4 |
| Otimizador | AdamW |
| Gradient clipping | 1,0 |
| Intervalo de avaliação | 100 |
| Batches de avaliação | 1 |
| Seeds | 1, 2, 3 |
| Precisão | padrão PyTorch, sem AMP |
| Hardware | não registrado |

O experimento foi parameter-matched e protocol-matched, mas não rigorosamente
time-matched ou compute-matched.

## Quais foram os resultados 36M?

Cross-entropy de validação:

| Modelo | Melhor CE média | Desvio | CE final média | Desvio |
|---|---:|---:|---:|---:|
| DRM | 2,3063 | 0,0391 | 2,3225 | 0,0468 |
| GPT-2 | 2,8914 | 0,0211 | 2,9252 | 0,0166 |
| OPT | 2,8927 | 0,0248 | 2,9729 | 0,0536 |

Throughput e memória:

| Modelo | Tokens/s | Memória máxima média |
|---|---:|---:|
| DRM | 1.806,5 | 1.131,8 MB |
| GPT-2 | 61.382,2 | 2.139,2 MB |
| OPT | 67.926,2 | 1.568,8 MB |

O DRM obteve menor CE nessa configuração preliminar, mas treinou muito mais
lentamente. Isso não demonstra superioridade geral.

## Quais são as limitações do benchmark 36M?

- três seeds;
- orçamento pequeno para LMs modernos;
- um batch de validação por ponto;
- ausência de metadados de hardware;
- não compute-matched ou time-matched;
- ausência de Mamba/SSM moderno;
- ausência de avaliação humana em escala;
- sem alegação formal de significância;
- sensibilidade possível ao split e a duplicações.

## Quais são os parâmetros DRM do benchmark 36M?

O arquivo interno é `configs/drm_125m.yaml`, embora o modelo produzido tenha
37.253.702 parâmetros:

| Parâmetro | Valor |
|---|---:|
| `vocab_size` | 256 |
| `d_token` | 768 |
| `d_state` | 768 |
| `n_directions` | 32 |
| `metric_rank` | 32 |
| `hidden_size` | 2048 |
| `n_flow_steps` | 1 |
| `dt` | 0,08 |
| `max_seq_len` | 512 |
| `dropout` | 0,0 |
| `bounded_state` | true |
| `state_clip_norm` | 8,0 |
| `direction_norm` | true |
| `direction_basis_size` | 128 |
| `metric_u_basis_size` | 128 |
| `geometry_update_interval` | 4 |
| `bptt_truncate_interval` | 64 |
| `emitter_layers` | 1 |
| `emitter_swiglu` | false |
| `emitter_residual` | false |
| `tie_embeddings` | false |

Regularização:

| Setting | Valor |
|---|---:|
| `lambda_action` | 0,01 |
| `lambda_dim_sparsity` | 0,001 |
| `lambda_dim_entropy` | 0,001 |
| `lambda_dim_variance` | 0,01 |
| `target_dim_std` | 0,15 |
| `lambda_metric_reg` | 0,001 |
| `lambda_metric_diversity` | 0,001 |
| `lambda_active_fraction` | 0,01 |
| `target_active_fraction` | 0,65 |
| `lambda_condition` | 0,001 |
| `target_condition` | 100,0 |
| `lambda_metric_u_floor` | 0,001 |
| `metric_u_min_norm` | 0,05 |
| `lambda_metric_u_target` | 0,001 |
| `metric_u_target_norm` | 1,0 |
| `lambda_recurrence` | 0,0 |
| `lambda_stability` | 0,0 |
| `lambda_blindspot` | 0,0 |

## O experimento usa weight tying?

Não. `tie_embeddings: false`. A entrada usa `TokenEmbedding.embedding` e a
saída usa uma projeção separada no `LanguageEmitter`.

Além disso, `tie_embeddings` é atualmente uma flag inerte quando ativada: a
implementação ainda não conecta os pesos. Experimentos futuros devem
implementar essa opção ou rejeitá-la explicitamente.

## De onde vêm os 37M parâmetros?

`count_parameters(model)` soma todos os parâmetros treináveis. As maiores
fontes são:

- embedding byte-level `256 × 768`;
- trunk, basis e heads do `DirectionField`;
- trunk, diagonal, basis e head low-rank da métrica;
- rede de coeficientes do `DRMFlow`;
- RMSNorm e projeções do `LanguageEmitter`;
- estado inicial `z0`.

## Como DRM se relaciona com RNN, SSM, Mamba e Neural ODE?

DRM é relacionado a métodos recorrentes e state-space porque mantém um estado.
A combinação proposta é:

- campo direcional aprendido;
- gates de direção;
- métrica relacional;
- fluxo precondicionado pela métrica;
- emissor causal.

Mamba e outros SSMs modernos são baselines relevantes ainda pendentes.

O modelo também lembra Neural ODEs, mas usa update discreto:

```text
z_next = z + dt * dz
```

Não há solver ODE adaptativo no núcleo.

## DRM resolve alignment ou segurança?

Não. Expor estado, direções, métrica e fluxo pode ajudar pesquisas de
observabilidade e controle, mas não resolve alignment ou segurança.

Continuam necessários avaliação externa, testes adversariais, controles,
sandboxing, monitoramento e restrições de implantação.

## Variáveis internas causam o resultado?

Inspeção não prova causalidade. São necessárias intervenções:

- zerar gates;
- substituir a métrica pela identidade;
- congelar o campo direcional;
- randomizar componentes;
- remover naturalização;
- alterar `metric_rank`;
- medir efeitos em loss, estabilidade, trajetória e geração.

Se intervenções não alterarem comportamento, interpretações fortes devem ser
reduzidas.

## Qual é o benchmark 125M já publicado?

O benchmark versionado mais forte usa três seeds por modelo e 150M tokens:

| Modelo | Parâmetros | Melhor CE média | Desvio | Tokens/s |
|---|---:|---:|---:|---:|
| DRM block64 + mixer causal | 127,27M | 1,3116 | 0,0019 | 10.678,7 |
| GPT-2 real | 126,08M | 1,7305 | 0,0259 | 41.224,4 |

```text
docs/benchmarks/competition_125m_local_mixer_h256_l2_s02_150m/
```

Esse resultado antigo não preservou os melhores checkpoints DRM exatos e usou
split por tokens. Ele é evidência preliminar, não validação externa.

## O que mudou no benchmark independente 125M?

O protocolo atual adiciona:

- split documental da Wikipédia;
- deduplicação antes da tokenização;
- PG-19 oficial como teste externo congelado;
- auditoria de contaminação com zero sobreposição;
- hardware e hashes registrados;
- melhores checkpoints;
- três seeds por família;
- uma única consulta externa após seleção por validação.

Em 2026-07-30, os treinamentos estão em andamento e o PG-19 ainda não foi
usado para seleção de modelo nem possui resultados de teste publicados.

```text
configs/independent_125m_protocol.json
scripts/run_independent_125m_benchmark.sh
scripts/evaluate_frozen_test.py
```

Avaliações intermediárias de quatro batches ainda são ruidosas. Checkpoints
candidatos devem ser comparados posteriormente sobre tokens determinísticos e
idênticos de validação antes de acessar o PG-19. O desvio do protocolo deve ser
documentado.

## A geração reproduz o forward local-mixer treinado?

Ainda não. `generation.py` avança o estado pela transição recorrente básica e
não reproduz o caminho block-cumsum/local-mixer usado pelo modelo 125M líder.

Isso não invalida treino ou avaliação CE, que chamam
`DRMEmitterModel.forward`. Porém, chat e amostras desses checkpoints ainda não
devem ser apresentados como inferência fiel do sequence engine treinado. É
necessária uma API comum de prefill/decode com testes de paridade.

## Como reproduzir o benchmark 36M?

```powershell
.\scripts\run_wiki_en_125m_matched.ps1
```

Ele usa:

```text
--models drm_125m gpt2_125m opt_125m
--dataset wikipedia-en
--steps 1000
--seeds 1 2 3
--batch-size 4
--grad-accum-steps 1
--seq-len 512
--lr 3e-4
--eval-interval 100
--eval-batches 1
--hf-vocab-size 256
```

Dependências Hugging Face:

```bash
pip install -e ".[hf]"
```

## Como reproduzir o benchmark independente?

Smoke test:

```bash
./scripts/run_independent_125m_smoke.sh
```

Treinar ou retomar os seis runs:

```bash
./scripts/run_independent_125m_benchmark.sh
```

Não executar `evaluate_frozen_test.py` sobre PG-19 antes de selecionar os
checkpoints.

## O que deve melhorar a seguir?

Prioridades:

- concluir e publicar a validação independente;
- aumentar e tornar determinística a seleção por validação;
- unificar geração com o forward block-cumsum/local-mixer;
- implementar ou rejeitar flags inertes;
- adicionar Mamba e SSMs modernos;
- executar comparações compute-matched e time-matched;
- realizar ablações de direções, métrica, gates e naturalização;
- medir inferência separadamente;
- testar contexto e retenção;
- avançar para rank/kernel/estratos do roadmap formal.

## Qual é a licença?

O projeto público usa `AGPL-3.0-only`, com texto em `LICENSE`. O repositório
também contém `LICENCE-COMMERCIAL.md`.

Em resumo: o código público é AGPL-3.0-only; uso proprietário ou comercial
deve seguir a via de licença comercial.
