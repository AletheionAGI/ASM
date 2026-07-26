# Time-To-Quality Benchmark

Data: 2026-07-26  
Origem dos artefatos: `runs\compare_10m_seed1`  
Escopo: 3 seeds, DRM causal Anderson b8 vs GPT-2 36M.

## Precedente Metodologico

Este benchmark usa a mesma ideia geral de "time-to-accuracy" popularizada por DAWNBench e depois consolidada em benchmarks de treinamento como MLPerf Training: medir tempo de relogio ate uma qualidade alvo, em vez de comparar apenas acuracia/CE ou apenas throughput.

Referencias:

```text
DAWNBench: https://dawn.cs.stanford.edu/dawnbench
MLPerf Training: https://mlperf.pw/benchmarks/training/index.html
```

Este nao e um resultado DAWNBench ou MLPerf, nem segue as regras oficiais dessas suites. O uso aqui e metodologico: adaptar a pergunta "tempo ate qualidade alvo" para comparar DRM e GPT-2 neste repositorio.

## Protocolo

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
directional_candidate_scale: 0.01
```

GPT-2:

```text
params: 36,819,216
model_size: gpt2_125m
```

## Resultado

O alvo time-to-quality foi definido como:

```text
TargetCe = best_val_ce_DRM + 0.01 = 1.84488347133001
```

Resumo:

| Modelo | Seeds | Best val CE medio | Std | Atingiu alvo | Tokens ate alvo mediano | Tempo ate alvo mediano |
|---|---:|---:|---:|---:|---:|---:|
| DRM causal Anderson b8 | 3 | 1.8349 | 0.0039 | 3/3 | 17,000,448 | 2,947.9s |
| GPT-2 36M | 3 | 2.0664 | 0.0142 | 0/3 | n/a | >807.9s medio censurado |

GPT-2 rodou ate pelo menos o piso de tokens do DRM antes de parar por plateau, conforme o criterio do benchmark. Em seed 3, rodou alem do piso ate 24,006,656 tokens. Nenhuma seed GPT-2 atingiu o alvo `1.84488347133001`.

Detalhe por seed:

| Modelo | Seed | Tokens vistos | Best val CE | Atingiu alvo | Tempo ate alvo |
|---|---:|---:|---:|---:|---:|
| DRM causal Anderson b8 | 1 | 20,004,864 | 1.8295 | sim | 2,947.9s |
| DRM causal Anderson b8 | 2 | 20,004,864 | 1.8368 | sim | 5,273.9s |
| DRM causal Anderson b8 | 3 | 20,004,864 | 1.8383 | sim | 2,896.1s |
| GPT-2 36M | 1 | 22,005,760 | 2.0715 | nao | censurado |
| GPT-2 36M | 2 | 20,004,864 | 2.0807 | nao | censurado |
| GPT-2 36M | 3 | 24,006,656 | 2.0471 | nao | censurado |

Interpretacao curta: este e um sinal multi-seed de qualidade ate alvo em escala pequena. Ele nao demonstra superioridade geral sobre Transformers; demonstra que, neste protocolo controlado, DRM atingiu o alvo em 3/3 seeds e GPT-2 em 0/3.

## Artefatos

```text
dashboard.html
time_to_quality_status.json
time_to_quality_runs.csv
time_to_quality_aggregate.csv
time_to_quality_points.csv
time_to_quality_by_seed.svg
best_val_ce_by_tokens.svg
val_ce_by_tokens.svg
tokens_per_sec_by_seed.svg
seconds_to_target_by_seed.svg
```

Observacao: este diretorio contem apenas artefatos leves de analise. Checkpoints e pesos permanecem em `runs\compare_10m_seed1` e nao foram copiados.
