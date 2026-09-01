# DRM TTA Multiseed 37M vs GPT-2

Data do relatorio: 2026-07-26  
Branch alvo: `drm-tta-multiseed`  
Artefatos versionados: `docs/benchmarks/tta/`

## 1. Objetivo

Confirmar se o resultado time-to-quality observado em seed unica se mantem em 3 seeds.

O criterio segue a ideia metodologica de time-to-accuracy usada por DAWNBench/MLPerf Training: medir tempo ate uma qualidade alvo, e nao avaliar CE e throughput como numeros desconectados. Este repositorio nao reivindica conformidade com DAWNBench ou MLPerf.

## 2. Protocolo

```text
dataset: data\tokens\manifest.json
seq_len: 64
batch_size: 16
grad_accum_steps: 1
precision: bf16
device: cuda
eval_tokens_interval: 1,000,000
eval_batches: 8
seeds: 1, 2, 3
```

DRM:

```text
params: 37,253,702
sequence_mode: directional_block_cumsum
directional_cumsum_block_size: 8
directional_anderson_iterations: 2
directional_anderson_history_size: 4
directional_anderson_ridge: 0.0001
```

GPT-2:

```text
params: 36,819,216
model_size: gpt2_125m
```

O alvo global foi:

```text
Target CE = 1.84488347133001
```

Esse alvo corresponde ao melhor CE DRM agregado acrescido de margem de 0.01 no protocolo de analise.

## 3. Resultado Agregado

| Modelo | Seeds | Best val CE medio | Std | Target reached | Tokens ate alvo mediano | Segundos ate alvo mediano | Tokens/s medio |
|---|---:|---:|---:|---:|---:|---:|---:|
| DRM causal Anderson b8 | 3 | 1.8349 | 0.0039 | 3/3 | 17,000,448 | 2,947.9s | 5,660.8 |
| GPT-2 36M | 3 | 2.0664 | 0.0142 | 0/3 | n/a | censurado | 29,470.6 |

## 4. Resultado Por Seed

| Modelo | Seed | Tokens vistos | Best val CE | Atingiu alvo | Tempo ate alvo |
|---|---:|---:|---:|---:|---:|
| DRM causal Anderson b8 | 1 | 20,004,864 | 1.8295 | sim | 2,947.9s |
| DRM causal Anderson b8 | 2 | 20,004,864 | 1.8368 | sim | 5,273.9s |
| DRM causal Anderson b8 | 3 | 20,004,864 | 1.8383 | sim | 2,896.1s |
| GPT-2 36M | 1 | 22,005,760 | 2.0715 | nao | censurado |
| GPT-2 36M | 2 | 20,004,864 | 2.0807 | nao | censurado |
| GPT-2 36M | 3 | 24,006,656 | 2.0471 | nao | censurado |

## 5. Interpretacao

O resultado confirma que o sinal de seed unica nao parece ser apenas variancia de inicializacao. Em 3/3 seeds, DRM atingiu o alvo de CE. Em 0/3 seeds, GPT-2 atingiu o mesmo alvo antes de plateau, mesmo treinando ate pelo menos o piso de tokens do DRM.

O resultado correto a comunicar e estreito:

```text
Em um benchmark time-to-quality de 3 seeds, pareado em parametros na escala ~37M,
DRM atingiu o alvo de validacao em todas as seeds, enquanto GPT-2 nao atingiu
o mesmo alvo em nenhuma seed no mesmo criterio de parada.
```

O resultado incorreto a comunicar seria:

```text
DRM e melhor que GPT-2/Transformers em geral.
```

## 6. Limitacoes

- Escala pequena: ~37M parametros.
- Orcamento pequeno para language modeling moderno: ~20M-24M tokens.
- Dataset/tokenizador especificos deste repositorio.
- Throughput DRM ainda e cerca de 5x menor que GPT-2 neste setup.
- Nao ha avaliacao conversacional, alignment, safety ou downstream tasks.
- O proximo risco cientifico e inversao de tendencia em escala maior.

## 7. Proximos Passos

1. Rodar continuidade maior, por exemplo 50M tokens.
2. Testar b16/block size maior depois da confirmacao b8.
3. Medir perfis de kernel/memoria para reduzir o custo do causal Anderson.
4. Repetir em escala maior de parametros.
5. Criar pacote de data room com protocolo, graficos, limitacoes e scripts reproduziveis.
