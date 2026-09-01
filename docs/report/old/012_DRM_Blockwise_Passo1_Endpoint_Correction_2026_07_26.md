# DRM Blockwise - Passo 1: Endpoint Correction

Data do relatorio: 2026-07-26  
Branch alvo: `drm-deer-drm`  
Objetivo: testar se corrigir o fim do bloco reduz o erro causado por geometria congelada em `z_start`.

## 1. Hipotese

No `directional_block_cumsum`, todos os deltas internos do bloco sao calculados a partir de `z_start`. O erro deve crescer ao longo do bloco. Uma correcao barata e recalcular apenas o endpoint usando o estado aproximado anterior ao ultimo token:

```text
z_last_corrected = F(z_{last-1 approx}, x_last)
correction = z_last_corrected - z_last_approx
states_i = states_i + weight_i * correction
```

O peso cresce por posicao dentro do bloco. Assim, o inicio do bloco quase nao muda e o final recebe a maior correcao.

## 2. Implementacao

Parametros adicionados:

```text
directional_endpoint_correction_weight
directional_endpoint_correction_power
```

Padrao:

```text
weight = 0.0
power = 1.0
```

Ou seja, desligado por padrao.

## 3. Setup experimental

```text
dataset: data\tokens\manifest.json
target_tokens: 384000
seq_len: 64
batch_size: 16
precision: bf16
device: cuda
lr: 0.0003
seed: 1
eval_tokens_interval: 128000
eval_batches: 8
```

## 4. Resultados

| Run | Best val CE | Tokens/s | Tempo |
|---|---:|---:|---:|
| baseline b8 | 2.9726 | 13.420 | 28,6s |
| endpoint b8 `weight=0.5` | **2.9478** | 7.897 | 48,6s |
| baseline b16 | 3.0757 | 22.773 | 16,9s |
| endpoint b16 `weight=0.5` | 3.0739 | 14.692 | 26,1s |

## 5. Leitura

Endpoint correction ajudou claramente o b8:

```text
2.9726 -> 2.9478
ganho: -0.0249 CE
```

No b16, o ganho foi quase nulo:

```text
3.0757 -> 3.0739
ganho: -0.0018 CE
```

Isso sugere que, para blocos muito longos, corrigir apenas o endpoint e pouco. O erro ja contaminou a trajetoria interna antes do final do bloco.

## 6. Decisao

Endpoint correction e uma boa melhoria barata para b8, mas nao resolve b16. Vale manter como candidato combinado com outros metodos, especialmente Anderson ou sub-blocos.

