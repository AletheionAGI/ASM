# DRM Blockwise - Passo 2: Sub-blocos Internos

Data do relatorio: 2026-07-26  
Branch alvo: `drm-deer-drm`  
Objetivo: testar se atualizar a geometria no meio de um bloco 16 recupera CE.

## 1. Hipotese

Se o problema do b16 e geometria congelada por tempo demais, entao dividir um bloco 16 em duas metades internas deve melhorar a qualidade:

```text
bloco externo 16
  sub-bloco 8 usando z_start
  sub-bloco 8 usando z_mid
```

Isso nao remove totalmente o custo sequencial, mas testa diretamente o mecanismo de erro.

## 2. Implementacao

Parametro adicionado:

```text
directional_cumsum_inner_block_size
```

Quando `0`, o comportamento antigo e preservado. Quando menor que `directional_cumsum_block_size`, o bloco e resolvido em partes internas.

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
block_size: 16
inner_block_size: 8
```

## 4. Resultados

| Run | Best val CE | Tokens/s | Tempo |
|---|---:|---:|---:|
| baseline b16 | 3.0757 | 22.773 | 16,9s |
| b16 com inner 8 | **2.9788** | 13.099 | 29,3s |
| baseline b8 | 2.9726 | 13.420 | 28,6s |

## 5. Leitura

O ganho contra b16 puro foi grande:

```text
3.0757 -> 2.9788
ganho: -0.0969 CE
```

Isso confirma o diagnostico mecanistico: atualizar o estado/geometria no meio do bloco reduz bastante o erro.

Mas o resultado ficou praticamente no regime de custo do b8:

```text
b16 inner8: 13.099 tokens/s
b8 puro:    13.420 tokens/s
```

Como esperado, dividir 16 em 8+8 se comporta quase como b8.

## 6. Decisao

Sub-blocos internos sao otimos como diagnostico e podem ser usados como baseline de recuperacao de qualidade. Para vencer o tradeoff, precisamos de um metodo que recupere parte do ganho de CE sem pagar quase todo o custo de b8.

