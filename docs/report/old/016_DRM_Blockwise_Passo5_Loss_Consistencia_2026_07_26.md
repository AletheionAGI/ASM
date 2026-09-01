# DRM Blockwise - Passo 5: Loss Auxiliar De Consistencia

Data do relatorio: 2026-07-26  
Branch alvo: `drm-deer-drm`  
Objetivo: testar se uma loss auxiliar ensina o cumsum a imitar a trajetoria sequencial local.

## 1. Hipotese

Durante treino, podemos comparar a trajetoria aproximada blockwise com um rollout sequencial target-free:

```text
L_consistency = mean(||Z_blockwise - stopgrad(Z_sequential)||^2)
```

A CE continua sendo a loss principal. A consistencia tenta regularizar os estados aproximados para ficarem mais proximos da dinamica recorrente real.

## 2. Implementacao

Parametros adicionados:

```text
lambda_block_consistency
block_consistency_weight
```

O alvo sequencial e calculado com `detach()`, para evitar transformar a loss em treino duplo do caminho sequencial.

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
lambda_block_consistency: 0.1
```

## 4. Resultados

| Run | Best val CE | Tokens/s | Tempo |
|---|---:|---:|---:|
| baseline b8 | 2.9726 | 13.420 | 28,6s |
| consistency b8 lambda 0.1 | 2.9825 | 4.377 | 87,7s |
| baseline b16 | 3.0757 | 22.773 | 16,9s |
| consistency b16 lambda 0.1 | 3.0784 | 5.079 | 75,6s |

## 5. Leitura

Com `lambda=0.1`, a consistencia piorou CE e foi cara:

```text
b8:  2.9726 -> 2.9825
b16: 3.0757 -> 3.0784
```

O custo e alto porque o alvo sequencial precisa ser calculado durante o forward. Isso elimina parte do ganho de paralelismo e nao trouxe melhora nesta configuracao.

## 6. Decisao

Nao vale usar `lambda_block_consistency=0.1` como proximo caminho principal. Ainda pode haver uma versao util com:

```text
lambda menor: 0.001 ou 0.01
aplicacao intermitente
sequencias menores
alvo sequencial apenas em poucos blocos
```

Mas, comparado com Anderson, a evidencia inicial e fraca.

## 7. Ranking apos os cinco passos

| Candidato | Best val CE | Tokens/s | Leitura |
|---|---:|---:|---|
| Anderson b8 iter2 | **2.5094** | 5.519 | melhor CE, caro |
| Anderson b16 iter2 | 2.6171 | 9.937 | melhor compromisso novo |
| endpoint b8 | 2.9478 | 7.897 | melhor ajuste barato |
| b16 inner8 | 2.9788 | 13.099 | confirma diagnostico, parecido com b8 |
| melhor sweep b8 | 2.9756 | 13.322 | ajuste parametrico insuficiente |
| consistency b8 0.1 | 2.9825 | 4.377 | piorou CE e throughput |
| baseline b8 | 2.9726 | 13.420 | baseline blockwise |
| baseline b16 | 3.0757 | 22.773 | throughput bom, CE ruim |

Conclusao: o caminho mais promissor agora e otimizar Anderson curto, principalmente b16 com 1-2 iteracoes. Endpoint b8 tambem merece teste de escala por ser simples e melhorar CE sem solver completo.

