# DRM Blockwise Cumsum: Comparativo Com GPT-2 E Diagnostico Do Bloco 16

Data do relatorio: 2026-07-26  
Branch alvo: `drm-deer-drm`  
Projeto auditado: `drm-language-emitter`  

## 1. Resumo executivo

Este relatorio registra a primeira rodada de implementacao e testes da linha `directional_cumsum` / `directional_block_cumsum`, criada para atacar diretamente o `for t` temporal do DRM.

A conclusao principal e:

```text
O cumsum total remove quase todo o loop temporal e acelera muito, mas degrada CE.
O blockwise cumsum recupera qualidade ao reintroduzir realimentacao geometrica entre blocos.
Em escala ~37M parametros, blocos 2, 4 e 8 vencem GPT-2 em CE no budget de 384k tokens.
O bloco 16 fica perto em throughput, mas ja perde CE para GPT-2.
Em 2M tokens com 3 seeds, a vantagem do bloco 8 nao se sustenta contra GPT-2.
```

Os melhores candidatos desta rodada sao:

```text
1. directional_block_cumsum bloco 8:
   melhor candidato DRM blockwise, mas ainda abaixo de GPT-2 no teste 2M/3 seeds.

2. directional_block_cumsum bloco 16:
   candidato de throughput, mas a perda de CE e estruturalmente maior.
```

## 2. Contexto tecnico

O DRM original executa a sequencia de forma recorrente:

```text
for t in seq_len:
    z_{t+1} = F(z_t, token_t)
```

Essa forma preserva a dependencia causal do estado, mas e lenta porque a GPU nao consegue paralelizar o eixo temporal como em um Transformer.

A nova linha experimental explora uma aproximacao:

```text
F(z_start, token_t) para varios tokens em paralelo
delta_t = F(z_start, token_t) - z_start
z_t ~= z_start + cumsum(delta_t)
```

No modo `directional_cumsum`, o bloco e a sequencia inteira:

```text
block_size = seq_len
```

No modo `directional_block_cumsum`, a sequencia e dividida em blocos:

```text
for bloco in sequencia:
    calcula cumsum paralelo dentro do bloco
    atualiza z_start para o ultimo estado do bloco
```

Isso reduz o numero de passos sequenciais de:

```text
seq_len
```

para:

```text
ceil(seq_len / block_size)
```

Com `seq_len=64`:

| block size | passos sequenciais aproximados |
|---:|---:|
| 1 | 64 |
| 2 | 32 |
| 4 | 16 |
| 8 | 8 |
| 16 | 4 |
| 64/full | 1 |

## 3. Implementacao realizada

Foram adicionados modos experimentais em `DRMConfig.sequence_mode`:

```text
local_step
geodesic_step
directional_candidates
directional_cumsum
directional_block_cumsum
```

O modo relevante deste relatorio e:

```text
directional_block_cumsum
```

Parametros novos:

```text
directional_candidate_scale
directional_candidate_temperature
directional_cumsum_block_size
```

O caminho padrao do modelo continua sendo:

```text
sequence_mode = local_step
```

Portanto, os modos novos sao opt-in e experimentais.

## 4. Setup experimental

Todos os runs principais desta secao usaram:

```text
dataset: data\tokens\manifest.json
target_tokens: 384000
seq_len: 64
batch_size: 16
grad_accum_steps: 1
precision: bf16
device: cuda
lr: 0.0003
eval_tokens_interval: 128000
eval_batches: 8
checkpoint_tokens_interval: 384000
```

Comparacao parametrica:

| Modelo | Parametros |
|---|---:|
| DRM `configs/drm_125m.yaml` | 37.253.702 |
| GPT-2 `gpt2_125m` | 36.819.216 |

Apesar do nome historico `drm_125m`, a configuracao usada nesta branch/benchmark e a escala de aproximadamente 37M parametros, conforme `parameter_count` reportado no treino.

## 5. Resultados principais

### 5.1 Comparativo 37M vs GPT-2

| Modelo / modo | Bloco | Best val CE | Tokens/s final | Tempo |
|---|---:|---:|---:|---:|
| DRM `block_cumsum` | 2 | **2.5587** | 3.575 | 107,4s |
| DRM `block_cumsum` | 4 | 2.7891 | 7.042 | 54,5s |
| DRM `block_cumsum` | 8 | 2.9726 | 13.420 | 28,6s |
| GPT-2 `gpt2_125m` | n/a | 3.0425 | **31.869** | 12,0s |
| DRM `block_cumsum` | 16 | 3.0757 | 22.773 | 16,9s |
| DRM `cumsum full` | 64 | 3.1790 | 23.980 | 16,0s |

### 5.2 Leitura direta

O DRM 37M com `block_cumsum` bate GPT-2 em CE ate bloco 8:

```text
block 2: 2.5587
block 4: 2.7891
block 8: 2.9726
GPT-2:   3.0425
```

O bloco 16 ja perde para GPT-2:

```text
block 16: 3.0757
GPT-2:    3.0425
```

O cumsum full perde ainda mais:

```text
full cumsum: 3.1790
```

Isso mostra uma curva consistente: quanto maior o bloco, maior o paralelismo temporal e maior a perda de realimentacao geometrica.

### 5.3 Validacao 2M tokens com 3 seeds

A recomendacao metodologica seguinte foi testar primeiro mais tokens e multiplas seeds antes de implementar outro modo experimental, especialmente `directional_block_anderson`.

Setup:

```text
dataset: data\tokens\manifest.json
target_tokens: 2.000.000
tokens_seen final: 2.000.896
seq_len: 64
batch_size: 16
grad_accum_steps: 1
precision: bf16
device: cuda
lr: 0.0003
eval_tokens_interval: 500000
eval_batches: 8
seeds: 1, 2, 3
```

Resultados por seed:

| Modelo / modo | Seed | Parametros | Best val CE | Tokens/s final | Tempo |
|---|---:|---:|---:|---:|---:|
| DRM `block_cumsum` b8 | 1 | 37.253.702 | 2.7477 | 13.971 | 143,2s |
| DRM `block_cumsum` b8 | 2 | 37.253.702 | 2.7450 | 13.568 | 147,5s |
| DRM `block_cumsum` b8 | 3 | 37.253.702 | 2.7471 | 13.437 | 148,9s |
| DRM `block_cumsum` b16 | 1 | 37.253.702 | 2.9122 | 23.226 | 86,1s |
| DRM `block_cumsum` b16 | 2 | 37.253.702 | 2.9083 | 23.099 | 86,6s |
| DRM `block_cumsum` b16 | 3 | 37.253.702 | 2.9032 | 23.197 | 86,3s |
| GPT-2 `gpt2_125m` | 1 | 36.819.216 | 2.6790 | 32.140 | 62,3s |
| GPT-2 `gpt2_125m` | 2 | 36.819.216 | 2.6861 | 32.091 | 62,4s |
| GPT-2 `gpt2_125m` | 3 | 36.819.216 | 2.6629 | 32.008 | 62,5s |

Agregado:

| Modelo / modo | N | Media best val CE | Desvio padrao | Media tokens/s | Media tempo |
|---|---:|---:|---:|---:|---:|
| GPT-2 `gpt2_125m` | 3 | **2.6760** | 0.0119 | **32.080** | **62,4s** |
| DRM `block_cumsum` b8 | 3 | 2.7466 | 0.0014 | 13.659 | 146,5s |
| DRM `block_cumsum` b16 | 3 | 2.9079 | 0.0046 | 23.174 | 86,3s |

Leitura:

```text
GPT-2 vence b8 em media por ~0.0706 CE.
GPT-2 vence b16 em media por ~0.2319 CE.
b8 e muito estavel entre seeds, mas estabiliza pior que GPT-2.
b16 preserva mais throughput que b8, mas o gap de CE cresce demais.
```

Esse resultado muda a interpretacao do teste de 384k: a vitoria inicial do bloco 8 era promissora, mas nao validada. Com mais tokens e seeds, GPT-2 aprende melhor no mesmo dataset/protocolo.

## 6. Melhores candidatos

### 6.1 Melhor candidato DRM blockwise: bloco 8

O bloco 8 continua sendo o melhor candidato dentro da familia DRM blockwise:

```text
384k best_val_ce: 2.9726
2M media best_val_ce: 2.7466
384k tokens/s: 13.420
2M media tokens/s: 13.659
```

No budget curto de 384k ele venceu GPT-2:

```text
2.9726 vs 3.0425
```

Mas no teste mais forte, com 2M tokens e 3 seeds, ele perdeu:

```text
DRM b8 medio: 2.7466
GPT-2 medio:  2.6760
gap:          +0.0706 CE
```

Portanto, bloco 8 nao deve ser tratado como vencedor contra GPT-2 ainda. Ele deve ser tratado como o melhor ponto de partida DRM para diagnosticar a perda de qualidade do blockwise.

Em termos estruturais, bloco 8 reduz a profundidade temporal sequencial de 64 passos para 8 blocos.

Esse e o candidato mais promissor para:

```text
1. diagnosticar o gap contra GPT-2;
2. testar mais escala de tokens;
3. ajustar parametrizacao do delta/cumsum;
4. tentar otimizacoes de kernel/compile somente depois de recuperar CE.
```

### 6.2 Candidato de throughput: bloco 16

O bloco 16 e importante porque se aproxima muito mais do regime paralelo:

```text
seq_len=64 -> 4 blocos sequenciais
```

Resultado:

```text
best_val_ce: 3.0757
tokens/s: 22.773
```

Ele ainda e mais lento que GPT-2:

```text
22.773 vs 31.869 tokens/s
```

mas ja esta na mesma ordem de grandeza. O problema e que perde CE:

```text
3.0757 vs 3.0425
```

Portanto, bloco 16 e o candidato certo se o objetivo for tentar recuperar qualidade mantendo alto paralelismo.

No teste 2M/3 seeds, a situacao ficou mais clara:

```text
DRM b16 medio: 2.9079
GPT-2 medio:   2.6760
gap:           +0.2319 CE
```

Isso sugere que o bloco 16 nao esta apenas subtreinado no budget curto. A aproximacao de geometria congelada por 16 posicoes provavelmente remove informacao temporal demais para o modelo atual compensar.

## 7. Diagnostico: por que o CE piora quando o bloco cresce

### 7.1 Perda de realimentacao de estado

No loop local:

```text
z_1 = F(z_0, x_1)
z_2 = F(z_1, x_2)
z_3 = F(z_2, x_3)
```

Cada token usa a geometria do estado atualizado.

No cumsum dentro de bloco:

```text
delta_i = F(z_start, x_i) - z_start
z_i ~= z_start + cumsum(delta_i)
```

Todos os tokens do bloco usam a mesma origem geometrica `z_start`. Isso cria erro crescente dentro do bloco:

```text
erro pequeno no inicio do bloco
erro maior perto do fim do bloco
```

Quanto maior o bloco, maior a distancia entre a geometria usada para calcular `delta_i` e o estado aproximado real daquele token.

### 7.2 Metrica congelada dentro do bloco

O DRM depende de:

```text
direction_field(z)
metric(z)
risk(z)
flow(z, token)
```

No block cumsum, esses componentes sao calculados a partir de `z_start` para todos os tokens do bloco. A metrica relacional nao acompanha a trajetoria interna do bloco.

Isso provavelmente explica a curva:

```text
b2 bom
b4 bom
b8 ainda bom
b16 comeca a perder
full perde bastante
```

### 7.3 Erro acumulado por soma

O cumsum assume que deltas locais podem ser somados em uma geometria aproximadamente constante. Isso e valido apenas se a dinamica for quase aditiva dentro do bloco.

Quando o bloco cresce, essa hipotese fica pior:

```text
delta_1 calculado em z_start
delta_16 tambem calculado em z_start
mas o estado real no token 16 ja deveria ter mudado varias vezes
```

### 7.4 Falta de correcao no endpoint do bloco

O modo atual usa uma aproximacao direta. Ele nao faz uma etapa de correcao no final do bloco para alinhar o endpoint com uma aplicacao recorrente mais fiel.

Isso torna o bloco 16 muito rapido, mas menos preciso.

## 8. Como tentar melhorar o CE do bloco 16

### 8.1 Sub-bloco corretivo interno

Uma opcao e manter `block_size=16`, mas calcular uma geometria intermediaria:

```text
tokens 1-8: cumsum a partir de z_start
tokens 9-16: recalcular geometria a partir de z_8 aproximado
```

Isso equivale a um bloco 16 com duas fases internas. Pode recuperar CE com menos overhead que bloco 8 completo, dependendo da implementacao.

Nome possivel:

```text
directional_block_cumsum_refine
```

### 8.2 Endpoint correction

Depois do cumsum do bloco, aplicar uma correcao barata no endpoint:

```text
z_end_approx = ultimo estado do cumsum
z_end_corrected = F(z_end_approx, token_final_ou_resumo)
```

Ou uma correcao residual aprendida:

```text
z_end = z_end_approx + residual_block(z_start, z_end_approx, resumo_tokens)
```

Objetivo: corrigir drift acumulado sem voltar ao loop token-a-token.

### 8.3 Anderson curto por bloco

Usar o cumsum como warmstart e aplicar poucas iteracoes Anderson dentro do bloco:

```text
block_size=16
warmstart=cumsum
anderson_iterations=2 ou 4
```

Os probes de Anderson mostraram que warmstart cumulativo reduz muito o erro inicial. Para bloco 16, duas a quatro iteracoes podem corrigir parte da trajetoria sem pagar o custo de 16 passos recorrentes.

Essa e talvez a opcao mais alinhada ao plano DEER.

### 8.4 Escala adaptativa por posicao

O erro cresce com a posicao dentro do bloco. Podemos reduzir agressividade dos deltas no fim do bloco:

```text
delta_i *= decay(i)
```

Exemplos:

```text
linear decay
sqrt decay
learned positional gate
```

Isso pode reduzir drift, mas tambem pode enfraquecer a memoria do bloco.

### 8.5 Geometria de baixa frequencia

Em vez de recalcular geometria a cada token, calcular em pontos internos:

```text
z_start
z_mid aproximado
z_end aproximado
```

e interpolar direcoes/metrica dentro do bloco. Isso preserva parte da curvatura sem voltar ao custo total do loop local.

### 8.6 Treinar mais tokens

O budget de 384k tokens e curto. O bloco 16 pode precisar de mais tokens para aprender a compensar sua propria aproximacao. Antes de rejeitar o modo, vale testar:

```text
2M tokens
mesmo protocolo
block 8 vs block 16 vs GPT-2
```

Se o gap do bloco 16 diminuir com mais treino, a aproximacao e treinavel. Se aumentar ou estabilizar pior, o problema e estrutural.

## 9. Proximos experimentos recomendados

### 9.1 Repetir com 2M tokens

Status: realizado.

Runs:

```text
GPT-2 37M
DRM block_cumsum b8
DRM block_cumsum b16
```

Mesmo dataset, `seq_len=64`, `batch=16`, seeds 1/2/3.

Resultado:

```text
GPT-2 medio:   2.6760 CE
DRM b8 medio:  2.7466 CE
DRM b16 medio: 2.9079 CE
```

Conclusao: o bloco 8 nao confirmou vantagem contra GPT-2 quando o teste ficou menos fragil. O bloco 16 ficou ainda mais distante.

### 9.2 Testar Anderson por bloco

Status: adiar.

Adicionar modo:

```text
directional_block_anderson
```

Parametros:

```text
block_size=16
anderson_iterations=2
history_size=4
warmstart=cumsum
```

Critério:

```text
CE melhor que b16
throughput ainda acima de b8
```

Depois do teste 2M/3 seeds, Anderson nao deve ser a proxima implementacao automatica. O motivo e que o bloco 8, que era a base empirica mais forte, perdeu para GPT-2. Antes de criar `directional_block_anderson`, e melhor diagnosticar se o gap do b8 vem de:

```text
1. parametrizacao fraca do delta;
2. escala/normalizacao do cumsum;
3. ausencia de informacao posicional dentro do bloco;
4. objetivo auxiliar insuficiente para estados aproximados;
5. vantagem arquitetural real do GPT-2 nesse regime.
```

### 9.3 Testar bloco 12

O ponto de virada parece entre 8 e 16. Um bloco intermediario pode ser melhor:

```text
block_size=12
```

Como `seq_len=64`, isso cria blocos irregulares, mas pode revelar a curva real.

### 9.4 Comparar seeds

Status: realizado para o protocolo 2M.

Seeds:

```text
seed 1
seed 2
seed 3
```

Resultado: o DRM b8 teve baixa variancia, mas media pior que GPT-2. Portanto, o problema nao parece ser uma seed ruim; parece ser gap sistematico neste protocolo.

### 9.5 Proximos testes depois da validacao 2M

Antes de implementar novo solver, os proximos testes mais informativos sao:

```text
1. DRM b8 com 5M ou 10M tokens para ver se o gap fecha com escala.
2. DRM b8 com escalas de delta: 0.003, 0.01, 0.03.
3. DRM b8 com temperature: 0.7, 1.0, 1.3.
4. DRM b8 com loss auxiliar no estado aproximado, se ja existir caminho limpo no treino.
5. Comparar b6, b8, b10, b12 para localizar melhor a fronteira qualidade/paralelismo.
```

O criterio de continuidade deve ser simples:

```text
Se b8 nao reduzir o gap contra GPT-2 com mais tokens ou ajuste leve,
nao vale investir engenharia pesada em Anderson ainda.
```

## 10. Conclusao

O experimento confirmou que a ideia blockwise e substancialmente melhor que cumsum full. O cumsum total prova throughput, mas perde geometria. O blockwise recupera a realimentacao do DRM em frequencia controlada.

Resultado mais importante apos a validacao 2M/3 seeds:

```text
DRM 37M block_cumsum b8:
  melhor candidato DRM blockwise,
  mas ainda pior que GPT-2 36.8M no protocolo 2M.
```

O bloco 16 tem throughput mais competitivo, mas perde CE de forma grande demais. O diagnostico mais provavel e erro acumulado por geometria congelada dentro do bloco. As melhores formas de atacar isso sao:

```text
1. primeiro diagnosticar e tentar fechar o gap do b8;
2. testar escala/temperatura do delta;
3. testar blocos intermediarios;
4. so entao considerar Anderson curto por bloco.
```

O proximo passo recomendado nao e implementar `directional_block_anderson` agora. A decisao mais rigorosa e rodar uma segunda bateria focada em b8, porque b8 e o unico candidato DRM blockwise perto o bastante para justificar engenharia adicional.
