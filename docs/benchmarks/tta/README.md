# Time-To-Quality Benchmark

Data: 2026-07-26  
Origem dos artefatos: `runs\compare_10m_seed1`  
Escopo: seed 1, DRM causal Anderson b8 vs GPT-2 36M.

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
seed: 1
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
TargetCe = best_val_ce_DRM + 0.01 = 1.83948632538319
```

Resumo:

| Modelo | Tokens vistos | Best val CE | Atingiu alvo | Tempo ate alvo |
|---|---:|---:|---:|---:|
| DRM causal Anderson b8 | 20,004,864 | 1.8295 | sim | 2,947.9s |
| GPT-2 36M | 22,005,760 | 2.0715 | nao | >701.1s |

GPT-2 rodou alem do piso de tokens do DRM antes de parar por plateau, conforme o criterio do benchmark.

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

