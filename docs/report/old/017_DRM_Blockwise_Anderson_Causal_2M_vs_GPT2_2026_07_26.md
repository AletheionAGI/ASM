# DRM Blockwise - Anderson Causal 2M vs GPT-2

Data do relatorio: 2026-07-26  
Branch alvo: `drm-deer-drm`  
Objetivo: validar a correcao causal do Anderson blockwise e comparar contra GPT-2 em 2M tokens.

## 1. Contexto

A auditoria identificou que o `anderson_solve` global anterior calculava coeficientes a partir do residual da trajetoria inteira. Isso permitia que estados de prefixo dependessem de tokens futuros, quebrando a propriedade autoregressiva esperada para language modeling.

O protocolo de comparacao foi posteriormente ampliado para "time-to-quality", seguindo o precedente metodologico de DAWNBench/MLPerf Training: medir tempo de relogio ate uma qualidade alvo, em vez de avaliar CE e throughput como numeros isolados. Este repositorio nao reivindica conformidade com DAWNBench ou MLPerf; a referencia e apenas ao criterio experimental.

Referencias:

```text
DAWNBench: https://dawn.cs.stanford.edu/dawnbench
MLPerf Training: https://mlperf.pw/benchmarks/training/index.html
```

A correcao implementada separou dois solvers:

```text
anderson_solve
  solver global nao causal, mantido para experimentos

causal_anderson_solve
  solver causal, usado no forward autoregressivo
```

O caminho autoregressivo do `DRMEmitterModel` agora usa `causal_anderson_solve` dentro de `_apply_block_anderson`.

## 2. Correcao Tecnica

O solver causal nao itera em Python sobre cada posicao do prefixo. Ele vetoriza os produtos internos que alimentam Anderson:

```text
residual_history: [batch, seq_len, d_state, history]
gram_step:        [batch, seq_len, history, history]
gram_prefix:      cumsum(gram_step, dim=1)
solve batched:    [batch * seq_len, history, history]
```

Assim, cada posicao `t` recebe coeficientes calculados apenas com residuos de `0..t`, mas o calculo continua paralelo e adequado para GPU. A causalidade entra pela soma cumulativa prefixal, nao por um loop sequencial em `t`.

Tambem foi mantida a resolucao do sistema pequeno em fp32 com autocast desabilitado, para preservar estabilidade numerica em CUDA bf16.

## 3. Testes De Causalidade

Foram adicionadas verificacoes para:

```text
causal_anderson_solve nao acoplar prefixo a inputs futuros
DRM directional_transition + causal Anderson preservar prefixo
directional_cumsum puro preservar prefixo antes do Anderson
```

Resultado da suite:

```text
53 passed, 1 warning
```

O aviso restante e conhecido:

```text
torch.nn.modules.linear: Initializing zero-element tensors is a no-op
```

Ele aparece no teste de `metric_rank=0` e nao indica falha funcional.

## 4. Setup Experimental 2M

Protocolo usado nos novos runs:

```text
dataset: data\tokens\manifest.json
target_tokens: 2,000,000
tokens_seen final: 2,000,896
seq_len: 64
batch_size: 16
grad_accum_steps: 1
lr: 0.0003
weight_decay: 0.01
precision: bf16
device: cuda
eval_tokens_interval: 500,000
eval_batches: 8
seeds: 1, 2, 3
```

DRM:

```text
config: configs\drm_125m.yaml
parameter_count: 37,253,702
sequence_mode: directional_block_cumsum
directional_cumsum_block_size: 8
directional_candidate_scale: 0.01
directional_candidate_temperature: 1.0
directional_anderson_iterations: 2
directional_anderson_history_size: 4
directional_anderson_ridge: 0.0001
directional_anderson_relaxation: 1.0
```

GPT-2:

```text
model_size: gpt2_125m
parameter_count: 36,819,216
```

Observacao: apesar do nome historico `125m`, esta configuracao tem cerca de 36.8M parametros.

## 5. Runs Gerados

```text
runs\compare_2m_drm_37m_causal_anderson_b8_seed_1
runs\compare_2m_drm_37m_causal_anderson_b8_seed_2
runs\compare_2m_drm_37m_causal_anderson_b8_seed_3

runs\compare_2m_gpt2_36m_rerun_seed_1
runs\compare_2m_gpt2_36m_rerun_seed_2
runs\compare_2m_gpt2_36m_rerun_seed_3
```

## 6. Resultados Por Seed

| Modelo | Seed | Params | Best val CE | Tokens/s | Tempo |
|---|---:|---:|---:|---:|---:|
| DRM causal Anderson b8 | 1 | 37,253,702 | 2.1999 | 5,528 | 362.0s |
| DRM causal Anderson b8 | 2 | 37,253,702 | 2.2157 | 5,518 | 362.6s |
| DRM causal Anderson b8 | 3 | 37,253,702 | 2.1775 | 5,426 | 368.8s |
| GPT-2 36M | 1 | 36,819,216 | 2.6790 | 32,208 | 62.1s |
| GPT-2 36M | 2 | 36,819,216 | 2.6861 | 31,986 | 62.6s |
| GPT-2 36M | 3 | 36,819,216 | 2.6629 | 31,753 | 63.0s |

## 7. Agregado

| Modelo | Params | Best val CE medio | CE std | Tokens/s medio | Tempo medio |
|---|---:|---:|---:|---:|---:|
| DRM causal Anderson b8 | 37,253,702 | **2.1977** | 0.0157 | 5,490 | 364.5s |
| GPT-2 36M | 36,819,216 | 2.6760 | 0.0097 | **31,982** | 62.6s |

Diferenca principal:

```text
DRM melhora best_val_ce medio em 0.4783 contra GPT-2.
GPT-2 e aproximadamente 5.8x mais rapido em tokens/s.
```

## 8. Leitura

O resultado muda a leitura do Passo 4. O Anderson curto continua caro, mas a versao causal vetorizada preserva a propriedade autoregressiva e mantem a vantagem de CE em 2M tokens.

O ganho de qualidade e forte:

```text
GPT-2 36M media:                 2.6760
DRM causal Anderson b8 media:    2.1977
```

O custo continua sendo o gargalo principal:

```text
GPT-2 36M media:                 31,982 tokens/s
DRM causal Anderson b8 media:     5,490 tokens/s
```

Portanto, a linha Anderson causal e tecnicamente valida como modo autoregressivo experimental, mas ainda nao resolve o problema de throughput.

## 9. Limitacoes Antes De Conclusao Fechada

A magnitude do ganho e grande: cerca de 0.48 CE contra GPT-2 no agregado de 2M tokens. Isso nao implica bug por si so, especialmente depois dos testes de causalidade, mas exige explicacao mecanica antes de tratar o resultado como conclusivo.

Verificacoes pendentes:

```text
1. medir residual inicial e residual final do Anderson por bloco
2. registrar quantas iteracoes realmente convergem e em quais blocos
3. comparar iter1 vs iter2 para separar ganho de refinamento de custo extra
4. medir se o history_size=4 cria uma mistura causal de curto alcance dentro do bloco
5. comparar contra mecanismos causais simples de mistura local com custo parecido
```

Uma interpretacao plausivel e que o Anderson causal nao esteja apenas "melhorando a geometria", mas tambem introduzindo uma forma de mistura causal local dentro do bloco. Isso nao diminui o valor do resultado, mas muda a narrativa: o ganho pode vir de um mecanismo causal de refinamento/mistura de curto alcance, nao apenas de geometria emergente no sentido mais estrito.

O throughput tambem deve ser tratado como problema central, nao detalhe operacional:

```text
DRM causal Anderson b8:  5,490 tokens/s
GPT-2 36M:              31,982 tokens/s
gap:                    GPT-2 ~5.8x mais rapido
```

Nesse estado, o resultado e cientificamente forte, mas ainda nao competitivo em custo pratico. A pergunta correta passa a ser:

```text
O DRM tem vantagem real de qualidade neste regime, mas consegue reduzir custo
suficiente para ser competitivo em escala?
```

Por fim, este ainda e um unico ponto de escala:

```text
37M parametros
2M tokens
seq_len 64
3 seeds
```

Como a propria investigacao ja mostrou mudanca de leitura entre 384k e 2M tokens, a conclusao deve permanecer limitada a este protocolo. Antes de declarar vantagem definitiva, e necessario testar pelo menos uma escala maior, idealmente 10M-20M tokens.

## 10. Decisao

Manter `directional_block_cumsum + causal_anderson_solve` como candidato principal de qualidade para a linha DEER/blockwise.

Proximos testes recomendados:

```text
1. b16 com Anderson causal iter2 em 2M tokens
2. b8 e b16 com Anderson causal iter1
3. history_size 2 vs 4
4. ridge 1e-3 vs 1e-4
5. perfil CUDA do solve batched para descobrir custo dominante
6. logar residual/convergencia do Anderson por bloco
7. rodar validacao em 10M-20M tokens se houver orcamento
```

O criterio de decisao agora deve ser preservar a maior parte do ganho de CE enquanto aproxima o throughput de b16/b32 puro.
