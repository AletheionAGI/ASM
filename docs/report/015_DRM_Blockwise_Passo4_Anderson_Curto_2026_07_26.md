# DRM Blockwise - Passo 4: Anderson Curto Por Bloco

Data do relatorio: 2026-07-26  
Branch alvo: `drm-deer-drm`  
Objetivo: testar se poucas iteracoes de Anderson sobre o cumsum recuperam a trajetoria recorrente dentro do bloco.

## 1. Hipotese

O blockwise cumsum e um chute inicial barato para a trajetoria:

```text
Z0 = cumsum(F(z_start, x_t) - z_start)
```

Anderson aplica iteracoes de ponto fixo sobre a trajetoria inteira do bloco:

```text
Z_{k+1} ~= Anderson(Phi(Z_k))
```

Isso ataca diretamente o erro de geometria congelada, porque cada iteracao reavalia `F` usando estados aproximados anteriores, nao apenas `z_start`.

## 2. Implementacao

Parametros adicionados:

```text
directional_anderson_iterations
directional_anderson_history_size
directional_anderson_ridge
directional_anderson_relaxation
```

Tambem foi necessario corrigir o solver para CUDA bf16: o sistema linear pequeno de Anderson agora e resolvido em fp32, com autocast desabilitado nesse trecho.

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
anderson_iterations: 2
history_size: 4
ridge: 0.001
relaxation: 1.0
```

## 4. Resultados

| Run | Best val CE | Tokens/s | Tempo |
|---|---:|---:|---:|
| baseline b8 | 2.9726 | 13.420 | 28,6s |
| baseline b16 | 3.0757 | 22.773 | 16,9s |
| Anderson b16 iter2 | 2.6171 | 9.937 | 38,6s |
| Anderson b8 iter2 | **2.5094** | 5.519 | 69,6s |
| GPT-2 baseline | 3.0425 | 31.869 | 12,0s |

## 5. Leitura

Anderson foi o primeiro metodo a melhorar CE de forma grande:

```text
b16: 3.0757 -> 2.6171
b8:  2.9726 -> 2.5094
```

Isso confirma fortemente o diagnostico da geometria congelada. Quando a trajetoria dentro do bloco e refinada por ponto fixo, o CE melhora muito.

O custo, porem, tambem e alto:

```text
b16 puro:      22.773 tokens/s
Anderson b16:   9.937 tokens/s
b8 puro:       13.420 tokens/s
Anderson b8:    5.519 tokens/s
```

Mesmo assim, Anderson b16 fica mais rapido que Anderson b8 e muito melhor em CE que b8 puro.

## 6. Decisao

Anderson curto e o melhor candidato tecnico ate agora para recuperar qualidade. O proximo teste correto nao e descartar pelo custo, mas procurar o ponto minimo de iteracoes:

```text
1 iteracao vs 2 iteracoes
b16 vs b32
relaxation 0.5 vs 1.0
history_size 2 vs 4
```

Se 1 iteracao mantiver grande parte do ganho, b16 Anderson pode virar o candidato principal.

