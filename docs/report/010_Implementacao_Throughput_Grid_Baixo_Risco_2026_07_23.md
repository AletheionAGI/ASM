# Implementacao Throughput Grid Baixo Risco

Data: 2026-07-23  
Escopo: implementacao dos itens 1 a 6 da ordem recomendada apos o report 009.

## 1. Itens implementados

Foram implementados os seis itens solicitados:

1. Grid mais agressivo de amortizacao.
2. Aumento de batch real por GPU via grid automatizado.
3. Profiling real com auto-grid.
4. Override para desligar ou espacar `lambda_metric_diversity`.
5. Validacao controlada de `compile_drm_step` via grid.
6. Prefetch simples do dataset, logging JSONL para runs longos e override de `metric_rank` para fases.

O objetivo foi manter risco baixo e preservar a arquitetura: as mudancas sao ativadas por flags/configs, com defaults conservadores.

## 2. Grid automatico de throughput

Novo script:

```text
scripts/run_drm_throughput_grid.py
```

Ele executa multiplas chamadas a:

```text
scripts/train_drm_memmap.py
```

e salva:

```text
runs/<grid>/summary.json
runs/<grid>/summary.csv
```

Grid padrao:

| Caso | Geometria | Aux | Naturalizacao | Batch | Compile | Metric diversity |
|---|---:|---:|---:|---:|---|---:|
| `baseline_g16_a8_n4_b2` | 16 | 8 | 4 | 2 | nao | config |
| `g32_a16_n8_b2` | 32 | 16 | 8 | 2 | nao | config |
| `g32_a16_n8_b4` | 32 | 16 | 8 | 4 | nao | config |
| `g64_a16_n8_b4` | 64 | 16 | 8 | 4 | nao | config |
| `g32_a16_n8_b2_compile` | 32 | 16 | 8 | 2 | sim | config |
| `g32_a16_n8_b2_no_metric_div` | 32 | 16 | 8 | 2 | nao | 0.0 |

Opcionalmente, com `--include-metric-rank-grid`, adiciona:

| Caso | Metric rank |
|---|---:|
| `g32_a16_n8_b2_rank32` | 32 |
| `g32_a16_n8_b2_rank16` | 16 |

## 3. Comando de grid recomendado

Smoke curto:

```powershell
.\.venv\Scripts\python.exe scripts\run_drm_throughput_grid.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\drm_throughput_grid_125m_smoke `
  --device cuda `
  --precision bf16 `
  --steps 5 `
  --seq-len 512 `
  --profile-steps 5 `
  --prefetch-batches 1 `
  --pin-memory `
  --fused-adamw
```

Grid com ranks reduzidos:

```powershell
.\.venv\Scripts\python.exe scripts\run_drm_throughput_grid.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\drm_throughput_grid_125m_rank `
  --device cuda `
  --precision bf16 `
  --steps 10 `
  --seq-len 512 `
  --profile-steps 10 `
  --prefetch-batches 1 `
  --pin-memory `
  --fused-adamw `
  --include-metric-rank-grid
```

## 4. Batch maior por GPU

O grid testa `batch_size=4` diretamente nos casos:

```text
g32_a16_n8_b4
g64_a16_n8_b4
```

Para testar batch ainda maior sem editar o script:

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\drm_125m_b8_probe `
  --device cuda `
  --precision bf16 `
  --steps 20 `
  --batch-size 8 `
  --grad-accum-steps 1 `
  --seq-len 512 `
  --geometry-update-interval 32 `
  --aux-loss-interval 16 `
  --naturalization-interval 8 `
  --forward-chunk-size 64 `
  --log-interval 1 `
  --profile-steps 20 `
  --pin-memory `
  --prefetch-batches 1 `
  --fused-adamw `
  --metrics-format jsonl `
  --no-save-final-checkpoint
```

## 5. Metric diversity desligavel no pretrain

`train_drm_memmap.py` agora aceita:

```text
--lambda-metric-diversity 0
```

Isso permite pretrain rapido sem reter a serie de `metric_diag_steps`. A arquitetura DRM continua igual; so a regularizacao auxiliar e removida ou alterada para a fase.

Exemplo:

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\drm_125m_no_metric_div_10m `
  --target-tokens 10000000 `
  --device cuda `
  --precision bf16 `
  --batch-size 2 `
  --grad-accum-steps 8 `
  --seq-len 512 `
  --lambda-metric-diversity 0 `
  --pin-memory `
  --prefetch-batches 1 `
  --fused-adamw `
  --metrics-format jsonl
```

## 6. Compile com warmup controlado

`--compile-drm-step` ja existia apos o report 009. Agora ele foi incluido no grid padrao:

```text
g32_a16_n8_b2_compile
```

Observacao importante: no smoke CPU, o caso compile foi muito lento por custo de compilacao inicial. Em CUDA, deve-se medir apos warmup e comparar `rolling_tokens_per_sec`, nao o primeiro step.

Comando isolado:

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\drm_125m_compile_probe `
  --device cuda `
  --precision bf16 `
  --steps 30 `
  --batch-size 2 `
  --grad-accum-steps 1 `
  --seq-len 512 `
  --geometry-update-interval 32 `
  --aux-loss-interval 16 `
  --naturalization-interval 8 `
  --compile-drm-step `
  --log-interval 1 `
  --profile-steps 30 `
  --pin-memory `
  --prefetch-batches 1 `
  --fused-adamw `
  --metrics-format jsonl `
  --no-save-final-checkpoint
```

## 7. Prefetch simples do dataset

`src/drm_language_emitter/data.py` agora separa:

```python
make_batch_cpu(...)
move_batch_to_device(...)
```

`scripts/train_drm_memmap.py` adiciona:

```text
--prefetch-batches N
```

Com `N > 0`, uma thread prepara batches CPU/pinned antes do step seguinte. A copia para CUDA continua no thread principal, evitando comportamento arriscado com CUDA em thread secundaria.

Isso e baixo risco porque:

- o caminho antigo permanece quando `--prefetch-batches 0`
- o prefetch usa o mesmo `torch.Generator`
- a preparacao segue sequencial em um worker
- o modelo nao muda

## 8. Logging JSONL para runs longos

Novo argumento:

```text
--metrics-format json|jsonl|both
```

Recomendado para runs longos:

```text
--metrics-format jsonl
```

Nesse modo:

- cada linha de historico entra em `metrics_history.jsonl`
- `metrics_latest.json` guarda apenas `latest` e `profile`
- evita regravar um JSON gigante a cada log

Isso reduz risco de degradacao em runs de muitos dias.

## 9. Metric rank por fase

Novo argumento:

```text
--metric-rank N
```

Uso recomendado:

- pretrain exploratorio com `--metric-rank 16` ou `32`
- fine-tune final com `metric_rank=64`

Isso muda parametrizacao e contagem de parametros, entao deve ser medido separadamente. Ainda respeita a arquitetura DRM porque mantem metrica relacional low-rank, apenas com rank menor.

## 10. Validacoes executadas

### Testes focados

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_token_shards.py tests\test_config.py tests\test_forward.py
```

Resultado:

```text
22 passed, 15 warnings
```

### Suite completa

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Resultado:

```text
37 passed, 15 warnings
```

### Compileall

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Resultado: sem erros.

### Dry-run com overrides novos

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\smoke_throughput_prefetch_dryrun `
  --device cpu `
  --precision fp32 `
  --seq-len 8 `
  --dry-run `
  --dry-run-forward `
  --aux-loss-interval 2 `
  --naturalization-interval 2 `
  --geometry-update-interval 4 `
  --forward-chunk-size 4 `
  --lambda-metric-diversity 0 `
  --metric-rank 16 `
  --metrics-format jsonl `
  --prefetch-batches 1 `
  --profile-steps 1
```

Resultado:

```text
parameter_count=103135622
dry_run_loss=5.593062
```

### Step real tiny com prefetch e JSONL

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\tiny.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\smoke_prefetch_jsonl_train `
  --device cpu `
  --precision fp32 `
  --steps 1 `
  --batch-size 1 `
  --grad-accum-steps 1 `
  --seq-len 8 `
  --log-interval 1 `
  --eval-tokens-interval 1000000000 `
  --checkpoint-tokens-interval 1000000000 `
  --metrics-format jsonl `
  --prefetch-batches 1 `
  --lambda-metric-diversity 0 `
  --metric-rank 1 `
  --no-save-final-checkpoint
```

Resultado:

```text
tokens_per_sec=137.34
instant_tokens_per_sec=138.13
```

### Smoke do grid

```powershell
.\.venv\Scripts\python.exe scripts\run_drm_throughput_grid.py `
  --config configs\tiny.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\smoke_grid_cpu `
  --device cpu `
  --precision fp32 `
  --steps 1 `
  --seq-len 8 `
  --profile-steps 1 `
  --prefetch-batches 1 `
  --no-pin-memory `
  --no-fused-adamw
```

Resultado:

```text
saved=runs\smoke_grid_cpu\summary.json
saved=runs\smoke_grid_cpu\summary.csv
```

## 11. Proxima avaliacao recomendada

Rodar primeiro:

```powershell
.\.venv\Scripts\python.exe scripts\run_drm_throughput_grid.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\drm_throughput_grid_125m_cuda `
  --device cuda `
  --precision bf16 `
  --steps 10 `
  --seq-len 512 `
  --profile-steps 10 `
  --prefetch-batches 1 `
  --pin-memory `
  --fused-adamw `
  --include-metric-rank-grid
```

Depois escolher o melhor caso e rodar 10M tokens antes de qualquer 150M.

## 12. Status

Implementado:

- grid automatico
- batch maior no grid
- metric diversity override
- compile no grid
- prefetch CPU/pinned
- JSONL para historico longo
- metric rank override
- testes e smokes

Pendente apenas a avaliacao longa em CUDA 125M, que deve ser rodada como experimento separado por tempo de GPU.
