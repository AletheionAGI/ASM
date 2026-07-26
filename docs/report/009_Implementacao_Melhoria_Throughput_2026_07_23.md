# Implementacao Da Melhoria De Throughput DRM

Data: 2026-07-23  
Escopo: primeira implementacao das alternativas do plano `008_Plano_De_Melhoria_Throughput_2026_26_07.md`.

## 1. Resumo

Foram implementadas as alternativas principais de throughput sem trocar a arquitetura DRM por Transformer:

- geometria amortizada por intervalo/agenda
- perdas geometricas em stride
- naturalizacao agendada
- acumuladores online para reduzir `torch.stack` e retencao de tensores
- loader memmap sem `bytes -> list -> tensor`
- treino com menos sincronizacao CPU/GPU por microbatch
- `DRMStep` extraido e opcionalmente compilavel
- chunking do forward por config
- trunk geometrico compartilhado opcional
- telemetria de throughput instantaneo, janela movel, fases de step e memoria CUDA

O comportamento antigo fica preservado por padrao:

```yaml
geometry_update_interval: 1
aux_loss_interval: 1
naturalization_interval: 1
forward_chunk_size: 0
compile_drm_step: false
use_shared_geometry_trunk: false
```

As configs existentes continuam validas. Foi adicionada uma config experimental:

```text
configs/drm_125m_4090_throughput.yaml
```

Ela mantem a contagem de parametros do DRM 125M atual, mas ativa uma agenda conservadora:

```yaml
geometry_update_interval: 16
aux_loss_interval: 8
naturalization_interval: 4
forward_chunk_size: 64
```

## 2. Arquivos alterados

### `src/drm_language_emitter/config.py`

Novos campos:

```python
compile_drm_step: bool = False
use_shared_geometry_trunk: bool = False
aux_loss_interval: int = 1
naturalization_interval: int = 1
forward_chunk_size: int = 0
```

Validacao adicionada:

- `aux_loss_interval >= 1`
- `naturalization_interval >= 1`
- `forward_chunk_size >= 0`
- flags booleanas para compile e trunk compartilhado

### `src/drm_language_emitter/direction_field.py`

`DirectionField.forward` agora aceita `hidden` opcional:

```python
directions, gates = self.direction_field(z, hidden)
```

Quando `use_shared_geometry_trunk=False`, o caminho antigo usa o trunk interno.

Quando `use_shared_geometry_trunk=True`, o trunk interno nao e criado e o modulo exige hidden externo vindo do `GeometryEncoder`.

### `src/drm_language_emitter/metric.py`

`RelationalMetric.forward` tambem aceita `hidden` opcional:

```python
metric_diag, metric_u = self.metric(z, hidden)
```

Isso permite compartilhar uma unica MLP geometrica entre campo direcional e metrica, preservando heads separados.

### `src/drm_language_emitter/model.py`

Mudancas centrais:

1. Adicionado `GeometryEncoder` opcional.
2. Extraido `_drm_step`.
3. Adicionado `_run_step` com fallback se `torch.compile` falhar.
4. Adicionado `_geometry` para computar direcoes, gates, metrica e risco juntos.
5. Substituidas listas densas de losses auxiliares por acumuladores online.
6. Adicionado `aux_loss_interval`.
7. Adicionado `naturalization_interval`.
8. Adicionado `forward_chunk_size`.

O forward antigo acumulava varias listas:

```python
action_values = []
dim_values = []
entropy_values = []
metric_regs = []
condition_values = []
u_norm_values = []
```

Agora o caminho quente usa somas online:

```python
action_sum
dim_sum
dim_square_sum
entropy_sum
metric_reg_sum
condition_sum
active_050_sum
u_norm_sum
u_norm_square_sum
u_floor_sum
```

Isso reduz alocacoes, reduz `torch.stack` no fim do forward e permite calcular losses auxiliares so a cada N ticks.

### `src/drm_language_emitter/data.py`

O loader memmap deixou de criar lista Python por janela:

Antes:

```python
values = torch.tensor(list(raw), dtype=torch.long)
```

Agora:

```python
values = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.long)
```

`make_batch` agora prealoca `x_cpu` e `y_cpu`, suporta `pin_memory` e copia para CUDA com `non_blocking` quando aplicavel.

### `scripts/train_drm_memmap.py`

Novas flags:

```text
--geometry-update-interval N
--aux-loss-interval N
--naturalization-interval N
--forward-chunk-size N
--compile-drm-step
--torch-compile
--shared-geometry-trunk
--pin-memory
--fused-adamw
--profile-steps N
--no-save-final-checkpoint
```

O loop de treino deixou de fazer:

```python
float(out["aux_losses"].get("ce", out["loss"]).detach().cpu())
```

a cada microbatch. Agora acumula tensors detached e so sincroniza para CPU quando precisa logar.

Novas metricas no `metrics_latest.json`:

```json
{
  "instant_tokens_per_sec": "...",
  "rolling_tokens_per_sec": "...",
  "step_elapsed_sec": "...",
  "data_elapsed_sec": "...",
  "forward_backward_elapsed_sec": "...",
  "optimizer_elapsed_sec": "...",
  "max_memory_mb": "...",
  "memory_allocated_mb": "...",
  "memory_reserved_mb": "..."
}
```

## 3. Config experimental adicionada

Arquivo:

```text
configs/drm_125m_4090_throughput.yaml
```

Objetivo:

- manter o desenho 125M atual
- amortizar geometria
- calcular losses geometricas com stride
- aplicar naturalizacao com stride
- resetar cache em chunks de 64 tokens

Config relevante:

```yaml
geometry_update_interval: 16
aux_loss_interval: 8
naturalization_interval: 4
forward_chunk_size: 64
compile_drm_step: false
use_shared_geometry_trunk: false
```

`compile_drm_step` ficou desligado na config inicial porque `torch.compile` pode ter warmup alto e recompilacoes dependentes do ambiente. A flag existe para experimentos controlados.

`use_shared_geometry_trunk` tambem ficou desligado na config inicial porque muda a parametrizacao e a compatibilidade de checkpoint. A implementacao esta pronta para uma branch/experimento proprio.

## 4. Testes executados

### Testes focados

Comando:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_forward.py tests\test_token_shards.py
```

Resultado:

```text
21 passed, 15 warnings
```

### Suite completa

Comando:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Resultado:

```text
36 passed, 15 warnings
```

### Validacao de `pyproject.toml`

Comando:

```powershell
.\.venv\Scripts\python.exe -c "import tomllib, pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text()); print('pyproject ok')"
```

Resultado:

```text
pyproject ok
```

### Compile check

Comando:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
```

Resultado: sem erros.

## 5. Dry-run funcional

Comando:

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\smoke_throughput_dryrun `
  --device cpu `
  --precision fp32 `
  --seq-len 8 `
  --dry-run `
  --dry-run-forward `
  --aux-loss-interval 2 `
  --naturalization-interval 2 `
  --geometry-update-interval 4 `
  --forward-chunk-size 4 `
  --profile-steps 1
```

Resultado:

```text
parameter_count=125161862
dataset_manifest=data\tokens_5b\manifest.json
world_size=1
train_tokens_available=5014643124
val_tokens_available=5019662
dry_run_loss=5.558307
```

## 6. Smoke de throughput em CUDA

Ambiente observado:

```text
torch.cuda.is_available() = True
device_count = 1
torch.version.cuda = 12.8
```

Os smokes abaixo usam 3 steps, `batch_size=2`, `grad_accum_steps=1`, `seq_len=512`, sem checkpoint final. Eles nao substituem benchmark longo. Servem para confirmar que o caminho rapido executa e medir custo inicial.

### Baseline 125M 4090

Comando:

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\throughput_smoke_baseline `
  --device cuda `
  --precision bf16 `
  --steps 3 `
  --batch-size 2 `
  --grad-accum-steps 1 `
  --seq-len 512 `
  --log-interval 1 `
  --eval-tokens-interval 1000000000 `
  --checkpoint-tokens-interval 1000000000 `
  --profile-steps 3 `
  --pin-memory `
  --no-save-final-checkpoint
```

Resultado resumido:

| Step | Instant tok/s | Rolling tok/s | Forward+backward s | Pico MB |
|---:|---:|---:|---:|---:|
| 1 | 318,3 | 318,3 | 3,1519 | 3013,7 |
| 2 | 409,3 | 358,1 | 2,4999 | 3932,2 |
| 3 | 409,0 | 373,6 | 2,5014 | 3932,2 |

### Config throughput 125M 4090

Comando:

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\throughput_smoke_fast `
  --device cuda `
  --precision bf16 `
  --steps 3 `
  --batch-size 2 `
  --grad-accum-steps 1 `
  --seq-len 512 `
  --log-interval 1 `
  --eval-tokens-interval 1000000000 `
  --checkpoint-tokens-interval 1000000000 `
  --profile-steps 3 `
  --pin-memory `
  --fused-adamw `
  --no-save-final-checkpoint
```

Resultado resumido:

| Step | Instant tok/s | Rolling tok/s | Forward+backward s | Pico MB |
|---:|---:|---:|---:|---:|
| 1 | 776,4 | 776,4 | 1,2976 | 1880,9 |
| 2 | 1080,1 | 903,4 | 0,9469 | 2528,2 |
| 3 | 1083,0 | 956,2 | 0,9444 | 2528,2 |

### Interpretacao

No mesmo protocolo curto, o modo throughput saiu de aproximadamente 409 tok/s instantaneo no baseline aquecido para aproximadamente 1083 tok/s. Isso e cerca de 2,65x no smoke inicial.

O pico de memoria caiu de aproximadamente 3932 MB para 2528 MB. Isso indica que `aux_loss_interval`, `geometry_update_interval`, `naturalization_interval` e acumuladores online estao reduzindo trabalho e grafo retido.

Ainda nao alcancou a meta de 2k tok/s nesse smoke de 3 steps. A proxima rodada deve testar:

- `geometry_update_interval=32`
- `aux_loss_interval=16`
- `naturalization_interval=8`
- `batch_size=4` ou `8`
- `compile_drm_step`
- `use_shared_geometry_trunk`

## 7. Comandos de reproducao recomendados

### 7.1 Baseline curto 10M tokens

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\eval_baseline_125m_10m `
  --target-tokens 10000000 `
  --batch-size 2 `
  --grad-accum-steps 8 `
  --seq-len 512 `
  --device cuda `
  --precision bf16 `
  --log-interval 10 `
  --eval-tokens-interval 5000000 `
  --checkpoint-tokens-interval 1000000000 `
  --profile-steps 20 `
  --pin-memory `
  --no-save-final-checkpoint
```

### 7.2 Throughput curto 10M tokens

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\eval_throughput_125m_10m `
  --target-tokens 10000000 `
  --batch-size 2 `
  --grad-accum-steps 8 `
  --seq-len 512 `
  --device cuda `
  --precision bf16 `
  --log-interval 10 `
  --eval-tokens-interval 5000000 `
  --checkpoint-tokens-interval 1000000000 `
  --profile-steps 20 `
  --pin-memory `
  --fused-adamw `
  --no-save-final-checkpoint
```

### 7.3 Grid agressivo para buscar 2k tok/s

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\eval_throughput_125m_grid_g32_a16_n8 `
  --target-tokens 10000000 `
  --batch-size 4 `
  --grad-accum-steps 4 `
  --seq-len 512 `
  --device cuda `
  --precision bf16 `
  --geometry-update-interval 32 `
  --aux-loss-interval 16 `
  --naturalization-interval 8 `
  --forward-chunk-size 64 `
  --log-interval 10 `
  --eval-tokens-interval 5000000 `
  --checkpoint-tokens-interval 1000000000 `
  --profile-steps 20 `
  --pin-memory `
  --fused-adamw `
  --no-save-final-checkpoint
```

### 7.4 Experimento com step compilado

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\eval_throughput_125m_compile_step `
  --target-tokens 10000000 `
  --batch-size 2 `
  --grad-accum-steps 8 `
  --seq-len 512 `
  --device cuda `
  --precision bf16 `
  --compile-drm-step `
  --log-interval 10 `
  --eval-tokens-interval 5000000 `
  --checkpoint-tokens-interval 1000000000 `
  --profile-steps 20 `
  --pin-memory `
  --fused-adamw `
  --no-save-final-checkpoint
```

### 7.5 Experimento com trunk geometrico compartilhado

Este experimento muda a parametrizacao e nao e compativel com checkpoints antigos.

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\eval_throughput_125m_shared_geometry `
  --target-tokens 10000000 `
  --batch-size 2 `
  --grad-accum-steps 8 `
  --seq-len 512 `
  --device cuda `
  --precision bf16 `
  --shared-geometry-trunk `
  --log-interval 10 `
  --eval-tokens-interval 5000000 `
  --checkpoint-tokens-interval 1000000000 `
  --profile-steps 20 `
  --pin-memory `
  --fused-adamw `
  --no-save-final-checkpoint
```

## 8. Comando para run longo de verificacao

Depois de escolher o melhor grid curto, rodar 150M tokens:

```powershell
.\.venv\Scripts\python.exe scripts\train_drm_memmap.py `
  --config configs\drm_125m_4090_throughput.yaml `
  --dataset-manifest data\tokens_5b\manifest.json `
  --output-root runs\drm_125m_4090_throughput_150m `
  --target-tokens 150000000 `
  --batch-size 2 `
  --grad-accum-steps 8 `
  --seq-len 512 `
  --device cuda `
  --precision bf16 `
  --log-interval 10 `
  --eval-tokens-interval 10000000 `
  --checkpoint-tokens-interval 50000000 `
  --profile-steps 20 `
  --pin-memory `
  --fused-adamw
```

Comparar contra o run anterior:

```text
runs\drm_125m_4090_base\metrics_latest.json
```

Metricas principais:

- `best_val_ce`
- `tokens_per_sec`
- `rolling_tokens_per_sec`
- `instant_tokens_per_sec`
- `forward_backward_elapsed_sec`
- `max_memory_mb`

## 9. Criterios de aceite

Para considerar a mudanca aprovada em 125M:

1. Suite completa continua passando.
2. Run curto 10M mostra ganho >= 2x em `rolling_tokens_per_sec`.
3. `best_val_ce` no run curto nao degrada de forma gritante.
4. Geracao qualitativa em checkpoint curto nao colapsa.
5. Run 150M fica abaixo de 24h ou mostra caminho claro para isso.
6. Se `shared_geometry_trunk` for usado, comparar qualidade separadamente porque a parametrizacao muda.

## 10. Status

Implementado e validado:

- config e validacao
- stride de losses auxiliares
- naturalizacao em intervalo
- acumuladores online
- loader memmap otimizado
- pin memory/non-blocking
- telemetria de treino
- fused AdamW opcional
- no-save final checkpoint para benchmarks curtos
- step DRM extraido e compilavel com fallback
- forward chunking
- trunk compartilhado opcional
- config throughput 125M
- testes unitarios e smoke CUDA

Pendente de avaliacao longa:

- run 10M baseline versus throughput
- grid agressivo para buscar 2k tok/s
- run 150M final
- comparacao qualitativa de geracao
- experimento separado com trunk compartilhado
