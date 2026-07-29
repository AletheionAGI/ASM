# DRM 125M b8 Throughput Bottleneck

Data do relatorio: 2026-07-29  
Escopo: diagnostico de throughput do DRM 125M em `directional_block_cumsum` com blocos pequenos (`b8`) e Anderson causal.  
Status: analise de engenharia; estimativas de melhoria precisam de benchmark confirmatorio.

## 1. Resumo executivo

O problema de throughput do DRM 125M nao parece ser causado principalmente por diagnosticos, `torch.linalg.solve` do Anderson ou candidate scoring isolado. Os probes recentes indicam um gargalo estrutural: o forward em `seq_len=512` com `block_size=8` executa 64 blocos sequenciais por microbatch, e cada bloco reaplica varias redes pequenas (`direction_field`, `metric`, `flow`, `risk`/candidate ou velocity) em batches efetivos pequenos.

Esse padrao e ruim para GPU:

```text
seq_len=512
block_size=8
blocks_per_microbatch = 512 / 8 = 64
batch_size = 2
effective positions per block = 2 * 8 = 16
grad_accum_steps = 8
```

Mesmo com `bf16`, o modelo dispara muitas chamadas pequenas e sequenciais. Isso reduz ocupacao de GPU e aumenta overhead de launch/autograd. A arquitetura local `b8` pode ser boa como escala geometrica, mas e ruim como unidade de execucao.

## 2. Evidencias medidas

As medicoes abaixo foram feitas em probes locais no regime 125M. Algumas linhas sao medidas curtas ate step 10/1M tokens, portanto devem ser tratadas como indicativas, nao conclusivas.

| Configuracao | Medida observada | Interpretacao |
|---|---:|---|
| `seq512 b8 iter2 candidate/trajectory` | ~625-736 tok/s no inicio | Baseline b8 iter2 lento |
| `seq512 b8 iter2 velocity/trajectory` | ~870-954 tok/s no inicio | Remover candidate do Anderson ajuda pouco |
| `seq512 b8 iter2 velocity/trajectory stride4` | ~1,476 tok/s no step 10 | Reduzir frequencia do Anderson ajuda, mas insuficiente |
| `seq512 b8 iter2 velocity/endpoint` | ~870 tok/s no step 10 | Endpoint simples nao resolve; chamadas pequenas continuam caras |
| `seq512 b32 iter1 candidate/trajectory` | ~3,762 tok/s em 1M-token sweep | Blocos maiores melhoram muito throughput |
| `seq512 b32 iter2 candidate/trajectory` | ~2,532 tok/s em 1M-token sweep | Mais Anderson reduz, mas ainda melhor que b8 |
| `seq512 b32 iter1` em 150M tokens | ~2,657 tok/s medio agregado | Throughput melhora, mas qualidade perdeu para GPT-2 |
| GPT-2 125M real em 150M tokens | ~42,451 tok/s medio agregado | Baseline Transformer segue muito mais eficiente |

Conclusao das evidencias: trocar detalhes dentro do Anderson nao basta. O custo dominante e a granularidade de execucao do caminho b8.

## 3. Caminho do codigo fonte

O forward do modelo fica em:

```text
src/drm_language_emitter/model.py
```

Quando `sequence_mode` esta em `directional_block_cumsum`, o modelo entra em `_forward_directional_cumsum`. O ponto critico e o loop por blocos:

```text
_forward_directional_cumsum(...)
  for block_start in range(0, seq_len, block_size):
      _directional_cumsum_block(...)
```

Com `seq_len=512` e `block_size=8`, esse loop roda 64 vezes por microbatch.

Cada bloco chama `_directional_cumsum_block_base`, que faz:

```text
direction_field(z_start)
metric(z_start)
risk(z_start)
flow(flat_z, flat_tokens, directions, gates)
metric.naturalize(...)
updater(...) ou _directional_candidate_step(...)
torch.cumsum(local_delta)
_apply_block_anderson(...)
```

O Anderson causal esta em:

```text
src/drm_language_emitter/deer.py
causal_anderson_solve(...)
```

Ele executa um loop por iteracao Anderson:

```text
for _ in range(iterations):
    image = trajectory_fixed_point(...)
    residual = image - trajectory
    ...
    gram_prefix = torch.cumsum(...)
    coeffs = torch.linalg.solve(...)
```

O `trajectory_fixed_point` reaplica a transicao na trajetoria do bloco. No regime `b8 iter2`, cada bloco faz aproximadamente:

```text
1 warmstart
2 Anderson fixed-point images
= 3 avaliacoes de campo por bloco
```

Por microbatch:

```text
64 blocos * 3 avaliacoes = 192 avaliacoes pequenas
```

Com `grad_accum_steps=8`:

```text
192 * 8 = 1536 avaliacoes pequenas por optimizer step
```

Isso explica por que o throughput fica abaixo de 1k tok/s mesmo depois de remover candidate scoring em alguns caminhos.

## 4. Alternativa 1: Superblocos com checkpoints b8

### Ideia

Manter a semantica local b8 como escala geometrica, mas mudar a unidade de execucao para superblocos maiores, por exemplo 32, 64 ou 128 tokens.

Em vez de o Python/autograd executar 64 blocos independentes de tamanho 8, o forward executaria 8 superblocos de 64 tokens. Dentro de cada superbloco, ainda existiriam checkpoints ou subpassos b8, mas computados de forma mais agrupada.

### Como poderia funcionar

Exemplo com `superblock_size=64` e `local_block_size=8`:

```text
seq512 -> 8 superblocos
cada superbloco -> 8 subblocos b8 internos
Anderson/consistencia -> aplicado nos endpoints ou em representacao compacta
logits -> emitidos para todos os tokens
```

O ganho vem de:

- menos iteracoes Python no nivel externo;
- batches efetivos maiores por chamada;
- maior chance de usar GEMMs grandes;
- menos overhead de autograd por bloco;
- possibilidade de aplicar Anderson em poucos checkpoints.

### Estimativa de throughput

| Variante | Throughput esperado | Ganho vs b8 atual |
|---|---:|---:|
| Superbloco 32, Anderson em checkpoints | ~2k-4k tok/s | ~2x-4x |
| Superbloco 64, Anderson em checkpoints | ~3k-6k tok/s | ~3x-6x |
| Superbloco 128, bem vetorizado | ~4k-8k tok/s | ~4x-8x |

Estimativa conservadora inicial: **3k-5k tok/s** se implementado em PyTorch puro com cuidado.

### Risco

O risco principal e qualidade: b32 puro ja mostrou que throughput melhora, mas qualidade caiu em 125M/150M. A solucao precisa preservar o mecanismo local b8, nao simplesmente trocar tudo para b32/b64.

### Prioridade

Alta. Esta parece a melhor relacao entre impacto e complexidade.

## 5. Alternativa 2: Anderson em endpoints/checkpoints, nao em toda trajetoria

### Ideia

Aplicar Anderson apenas em estados selecionados: endpoints de bloco, endpoints de superbloco ou checkpoints intermediarios. Os estados internos ficam com warmstart/cumsum.

O teste simples `directional_anderson_scope=endpoint` nao resolveu, mas ele ainda operava no regime b8 com 64 chamadas pequenas. A versao promissora e endpoint/checkpoint dentro de superblocos, nao endpoint b8 isolado.

### Como poderia funcionar

Em um superbloco de 64 tokens:

```text
tokens 0..63
checkpoints: 7, 15, 23, 31, 39, 47, 55, 63
warmstart local para todos os tokens
Anderson apenas nos checkpoints
interpolacao/cumsum causal para internos
```

Isso reduziria o custo do Anderson por posicao e evitaria solve completo para cada token.

### Estimativa de throughput

| Variante | Throughput esperado | Ganho vs b8 atual |
|---|---:|---:|
| Endpoint b8 simples | ~0.8k-1.0k tok/s medido | sem ganho |
| Checkpoints em superbloco 32 | ~2k-4k tok/s | ~2x-4x |
| Checkpoints em superbloco 64 | ~3k-6k tok/s | ~3x-6x |
| Checkpoints esparsos + stride Anderson | ~4k-7k tok/s | ~4x-7x |

Estimativa realista: **3k-6k tok/s** se combinado com superblocos.

### Risco

Pode perder a qualidade do b8 iter2 se os logits internos dependerem muito da correcao Anderson em cada posicao. Precisa medir CE em 10M antes de escalar para 150M.

### Prioridade

Alta, mas deve ser implementada junto com superblocos. Isoladamente ja mostrou pouco ganho.

## 6. Alternativa 3: Kernel/scan especializado

### Ideia

Preservar a semantica b8 atual, mas remover overhead de Python/autograd/launch usando compilacao ou kernel especializado.

Possiveis niveis:

1. `torch.compile` em partes do bloco.
2. Reescrever loops internos para `torch.func.scan`/vmap quando disponivel.
3. Triton/CUDA custom para scan de estados.
4. Kernel fused para `flow + naturalize + updater` em blocos.

### Por que isso pode funcionar

O problema atual tem muitas chamadas pequenas. Kernel fusion pode transformar varias operacoes pequenas em poucas operacoes grandes. Essa e a unica alternativa que tenta preservar a semantica b8 mais fielmente.

### Estimativa de throughput

| Nivel | Throughput esperado | Ganho vs b8 atual |
|---|---:|---:|
| `torch.compile` parcial | ~1.2k-2k tok/s | ~1.3x-2x |
| Reorganizacao PyTorch/vmap | ~2k-4k tok/s | ~2x-4x |
| Triton/fused kernels | ~5k-10k tok/s | ~5x-10x |
| CUDA altamente especializado | ~8k-15k tok/s | ~8x-15x |

Estimativa realista de curto prazo: **2k-4k tok/s** sem escrever kernel custom.

### Risco

Complexidade alta. Tambem pode ser fragil com bf16, autograd e Windows. Triton no Windows pode adicionar atrito operacional.

### Prioridade

Media. Muito importante se o projeto precisar manter b8 completo, mas nao e o primeiro caminho mais pragmatico.

## 7. Alternativa 4: Treino hibrido

### Ideia

Nao usar Anderson completo em todos os optimizer steps. O treino alternaria entre passos baratos e passos geometricamente ricos.

Exemplos:

```text
75% steps: b8 cumsum rapido sem Anderson
25% steps: b8 Anderson iter2
```

ou:

```text
fase 1: treino rapido sem Anderson ate N tokens
fase 2: fine-tune com Anderson
```

ou:

```text
Anderson a cada 4 optimizer steps
```

### Como isso melhora throughput

Se Anderson completo custa ~750-950 tok/s e o modo rapido conseguir ~2k-4k tok/s, o throughput medio melhora proporcionalmente ao mix.

Exemplo aproximado:

| Agenda | Throughput medio esperado |
|---|---:|
| Anderson todo step | ~0.8k-1.0k tok/s |
| Anderson 1/2 steps | ~1.3k-2k tok/s |
| Anderson 1/4 steps | ~1.8k-3k tok/s |
| Anderson 1/8 steps | ~2.2k-3.5k tok/s |
| Anderson so fine-tune final | depende da fase, possivelmente ~3k+ medio |

### Risco

Pode perder justamente o efeito que deu vantagem no 37M. Mas e barato de testar e pode revelar se Anderson e necessario durante todo o treino ou apenas para estabilizar/refinar.

### Prioridade

Alta para experimentacao rapida. Baixa para solucao final se a qualidade depender de Anderson denso.

## 8. Recomendacao de proximo passo

Eu nao continuaria tentando microflags no b8 atual. Os probes ja deram a resposta principal: `b8` como unidade de execucao e pequeno demais para 125M.

Sequencia recomendada:

1. Implementar `directional_superblock_cumsum` experimental.
2. Usar `superblock_size=64`, `local_block_size=8`.
3. Emitir logits para todos os tokens.
4. Aplicar Anderson apenas nos checkpoints b8 ou no endpoint do superbloco.
5. Rodar probes de 1M e 10M tokens contra:
   - `b8 iter2` atual;
   - `b32 iter1`;
   - GPT-2 125M;
   - superbloco novo.

Meta minima para valer continuar:

```text
throughput >= 3,000 tok/s em 125M seq512
CE em 10M claramente melhor que b32 iter1
```

Meta competitiva:

```text
throughput >= 6,000 tok/s
qualidade proxima do b8 iter2 37M-style
```

Meta de produto:

```text
throughput >= 10,000 tok/s em uma GPU local
```

## 9. Conclusao

O resultado 37M continua tecnicamente importante, mas a escala 125M revelou que o mecanismo vencedor (`b8 iter2`) esta acoplado a uma unidade de execucao pequena demais. O desafio agora nao e apenas "otimizar Anderson"; e redesenhar o caminho de execucao para preservar a geometria local b8 enquanto a GPU trabalha em blocos grandes.

A direcao mais promissora e:

```text
geometria local b8
+ execucao por superblocos
+ Anderson em checkpoints/endpoints
```

Essa e a ponte mais plausivel entre a qualidade observada em 37M e throughput aceitavel em 125M.
