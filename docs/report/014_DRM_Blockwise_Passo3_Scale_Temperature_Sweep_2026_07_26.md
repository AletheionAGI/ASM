# DRM Blockwise - Passo 3: Sweep De Scale E Temperature

Data do relatorio: 2026-07-26  
Branch alvo: `drm-deer-drm`  
Objetivo: testar se o gap do b8 vem de parametrizacao simples do delta direcional.

## 1. Hipotese

O cumsum pode estar acumulando deltas com escala inadequada. Se a escala for alta, o bloco deriva; se for baixa, o estado anda pouco. A temperatura tambem altera a mistura dos candidatos direcionais.

Parametros testados:

```text
directional_candidate_scale: 0.003, 0.01, 0.03
directional_candidate_temperature: 0.7, 1.0, 1.3
```

## 2. Setup experimental

```text
dataset: data\tokens\manifest.json
target_tokens: 384000
seq_len: 64
batch_size: 16
precision: bf16
device: cuda
lr: 0.0003
seed: 1
block_size: 8
```

## 3. Resultados

| Scale | Temp | Best val CE | Tokens/s | Tempo |
|---:|---:|---:|---:|---:|
| 0.003 | 0.7 | 2.9831 | 13.364 | 28,7s |
| 0.003 | 1.0 | 2.9812 | 13.652 | 28,1s |
| 0.003 | 1.3 | 2.9815 | 13.400 | 28,7s |
| 0.010 | 0.7 | 2.9792 | 13.167 | 29,2s |
| 0.010 | 1.0 | 2.9788 | 13.353 | 28,8s |
| 0.010 | 1.3 | 2.9784 | 13.775 | 27,9s |
| 0.030 | 0.7 | **2.9756** | 13.322 | 28,8s |
| 0.030 | 1.0 | 2.9799 | 13.142 | 29,2s |
| 0.030 | 1.3 | 2.9758 | 13.405 | 28,6s |

Comparativos relevantes:

```text
baseline b8 anterior: 2.9726
endpoint b8:          2.9478
melhor sweep puro:    2.9756
```

## 4. Leitura

O sweep nao melhorou o baseline b8 anterior. A diferenca entre configuracoes foi pequena, e todas ficaram piores que endpoint correction.

Isso indica que o problema principal nao e apenas escala global ou temperatura dos candidatos. A perda vem mais provavelmente do mecanismo temporal: a geometria congelada dentro do bloco.

## 5. Decisao

Nao vale investir primeiro em sweep amplo de scale/temperature. Esses parametros podem ser revisitados depois de um metodo estrutural melhor, mas nao resolvem o CE sozinhos.

