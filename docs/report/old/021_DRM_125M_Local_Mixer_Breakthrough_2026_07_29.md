# DRM 125M - Virada do Anderson b8 Para Local Mixer Causal

Data do relatorio: 2026-07-29  
Escopo: documentar a solucao encontrada para o gargalo do DRM 125M, os resultados obtidos ate agora e as melhorias futuras mais promissoras.  
Status: resultado experimental forte, ainda com 150M multi-seed em andamento.

## 1. Resumo executivo

O caminho que funcionou em 37M parametros nao escalou bem para 125M.

No regime pequeno, o DRM `causal Anderson b8` mostrou uma troca interessante: throughput bruto menor que GPT-2, mas melhor time-to-quality. Em 125M, a mesma familia ficou lenta demais. Profiling mostrou que o problema principal nao era apenas o solve do Anderson, mas a granularidade do grafo: milhares de kernels pequenos, muitas chamadas `mm/addmm/add_/mul/copy_`, e baixa eficiencia por bloco.

A solucao que mudou o resultado foi substituir o papel pratico do Anderson local por um **mixer causal local barato**:

```text
directional_block_cumsum
block_size = 64
cumsum_step_mode = velocity
anderson_iterations = 0
local_mixer = causal_conv
local_mixer_hidden_size = 256
local_mixer_kernel_size = 8
local_mixer_layers = 2
local_mixer_scale = 0.2
```

Essa configuracao preserva a ideia central: uma trajetoria latente causal com mistura local de curto alcance. Mas troca o mecanismo caro de ponto fixo/Anderson por uma correcao convolucional causal paralelizavel.

Resultado em 10M tokens, 3 seeds, 125M pareado:

| Familia | Params | Best val CE medio | Tokens/s medio | Target reached |
|---|---:|---:|---:|---:|
| DRM local mixer | 127.27M | 1.6464 | 11.9k | 2/3 pelo target global estrito |
| GPT-2 real | 126.08M | 2.9365 | 42.1k | 0/3 |

Resultado parcial em 150M:

| Run | Tokens | Best val CE | Observacao |
|---|---:|---:|---|
| DRM seed 1 | 150.0M | 1.3134 | concluido |
| DRM seed 2 | 104.4M informado no terminal | 1.3159 | ainda rodando, ja replica seed 1 |

A conclusao atual e clara:

```text
O DRM local mixer nao vence GPT-2 em throughput bruto por token.
Ele vence por eficiencia estatistica e time-to-quality.
```

## 2. O que falhou

### 2.1 Causal Anderson b8 em 37M

O caminho `b8 + Anderson causal` foi importante porque corrigiu a causalidade do Anderson blockwise e mostrou, em escala pequena, que o DRM podia atingir qualidade que o GPT-2 pareado nao atingia no mesmo protocolo.

Esse resultado sustentou a hipotese:

```text
Talvez a mistura local da trajetoria latente entregue mais qualidade por token,
mesmo custando mais computacao por token.
```

### 2.2 Escala 125M

Ao escalar para 125M, o b8 Anderson ficou caro demais. Probes observados:

| Configuracao | Throughput aproximado | Leitura |
|---|---:|---|
| b8 iter2 candidate | ~0.7k tok/s | impraticavel |
| b8 iter2 velocity | ~0.9k tok/s | melhora pequena |
| b8 iter2 velocity stride4 | ~1.5k tok/s | ainda longe |
| super64 local8 iter2 | ~4.1k tok/s | melhor, mas insuficiente |
| b64 velocity iter0 | ~13k tok/s | rapido, mas CE ruim sem mixer |

O profiler reforcou a leitura:

- `linalg_lu_solve` e `linalg_lu_factor_ex` aparecem, mas nao dominam sozinhos.
- O topo e dominado por milhares de `aten::mm`, `aten::addmm`, `aten::add_`, `aten::mul`, `aten::copy_` e kernels elementwise.
- A combinacao de blocos pequenos, Anderson local e autograd fragmentado torna o caminho caro.

Conclusao:

```text
Nao era um problema de flag. Era um problema de representacao computacional.
```

## 3. A mudanca de hipotese

O insight foi separar duas coisas:

1. A semantica matematica exata do Anderson.
2. O efeito funcional observado: mistura causal local da trajetoria latente.

Se o Anderson estava ajudando por atuar como um mixer causal local, entao talvez fosse possivel trocar o mecanismo exato por um mixer neural causal barato.

Isso gerou a nova hipotese:

```text
Um block scan rapido pode produzir uma trajetoria latente base.
Um mixer causal local pode corrigir essa trajetoria usando apenas passado e presente.
```

## 4. Implementacao atual

Arquivos principais:

- `src/drm_language_emitter/config.py`
- `src/drm_language_emitter/model.py`
- `scripts/train_drm_memmap.py`
- `scripts/run_125m_150m_multiseed_competition.ps1`
- `scripts/run_125m_local_mixer_probe.ps1`
- `scripts/run_125m_local_mixer_validation_sequence.ps1`
- `scripts/check_125m_local_mixer_causality.py`

### 4.1 CausalLocalMixer

O mixer recebe, por posicao:

- estado latente atual do bloco;
- residual local em relacao ao estado anterior;
- delta local calculado pelo campo direcional;
- embedding do token;
- dimensionalidade ativa;
- massa de risco.

Depois aplica:

1. Projecao linear para hidden local.
2. Uma ou mais camadas `Conv1d` depthwise causais.
3. `Conv1d` pointwise.
4. Projecao de volta para `d_state`.
5. Residual escalado sobre o estado.

Forma conceitual:

```text
z_mixed[t] = z_base[t] + scale * causal_conv(features[<=t])
```

Isso e importante: o mixer nao ve futuro.

### 4.2 Config vencedora ate agora

```powershell
.\scripts\run_125m_150m_multiseed_competition.ps1 `
  -OutputRoot "runs\competition_125m_local_mixer_h256_l2_s02_150m" `
  -Seeds 1,2,3 `
  -TargetTokens 150000000 `
  -SeqLen 512 `
  -BatchSize 2 `
  -GradAccumSteps 8 `
  -EvalTokensInterval 1000000 `
  -CheckpointTokensInterval 50000000 `
  -DrmSequenceMode directional_block_cumsum `
  -DrmBlockSize 64 `
  -DrmAndersonIterations 0 `
  -DrmCumsumStepMode velocity `
  -DrmLocalMixer causal_conv `
  -DrmLocalMixerHiddenSize 256 `
  -DrmLocalMixerKernelSize 8 `
  -DrmLocalMixerLayers 2 `
  -DrmLocalMixerScale 0.2
```

## 5. Validacao de causalidade

Foi criado o script:

```text
scripts/check_125m_local_mixer_causality.py
```

Ele altera tokens futuros e compara logits/estados no prefixo. No checkpoint real do seed 1 em 10M:

```json
{
  "passed": true,
  "max_logit_abs_diff": 0.0,
  "max_state_abs_diff": 0.0
}
```

Prefixos testados incluem bordas internas relevantes:

```text
1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 511
```

Isso nao prova todas as propriedades do modelo, mas remove o principal risco imediato: vazamento simples de futuro pelo mixer ou pelo block scan.

## 6. Resultados 10M multi-seed

Fonte: `runs/competition_125m_local_mixer_h256_l2_s02_10m/time_to_quality_status.json`

### 6.1 DRM local mixer

| Seed | Best val CE | Tokens/s | Tokens |
|---:|---:|---:|---:|
| 1 | 1.6571 | 11,938.7 | 10.0M |
| 2 | 1.6435 | 11,960.5 | 10.0M |
| 3 | 1.6388 | 11,858.2 | 10.0M |

Agregado:

```text
best_val_ce_mean = 1.6464
best_val_ce_std  = 0.0077
tokens_per_sec_mean = 11.9k
```

### 6.2 GPT-2 125M real

| Seed | Best val CE | Tokens/s | Tokens |
|---:|---:|---:|---:|
| 1 | 2.9489 | 41,933.2 | 10.0M |
| 2 | 2.9105 | 42,297.8 | 10.0M |
| 3 | 2.9501 | 42,185.2 | 10.0M |

Agregado:

```text
best_val_ce_mean = 2.9365
tokens_per_sec_mean = 42.1k
```

### 6.3 Leitura

GPT-2 processa cerca de 3.5x mais tokens por segundo nesse setup. Mesmo assim, em 10M tokens, sua qualidade fica muito atras:

```text
DRM mean CE  = 1.6464
GPT-2 mean CE = 2.9365
Delta CE = -1.2901 a favor do DRM
```

Em perplexidade aproximada:

```text
exp(1.6464) ~= 5.19
exp(2.9365) ~= 18.85
```

## 7. Resultado 150M em andamento

Fonte parcial: `runs/competition_125m_local_mixer_h256_l2_s02_150m/time_to_quality_status.json` e logs de terminal.

### 7.1 Seed 1 concluido

```text
tokens_seen = 150,003,712
best_val_ce = 1.3134306073
final logged val_ce = 1.3881737292
seconds_to_target = 13,257s
tokens_to_target = 132,005,888
elapsed = 15,775s
```

### 7.2 Seed 2 em andamento

Log informado:

```text
tokens_seen = 104,366,080
best_val_ce = 1.3158806860
tokens_per_sec = 10,156 tok/s
```

Isso e crucial porque replica o seed 1 antes mesmo de concluir 150M:

```text
seed 1 best = 1.3134
seed 2 best parcial = 1.3159
delta = 0.0025 CE
```

## 8. Por que isso funcionou

Interpretacao atual:

1. O block64 velocity scan fornece uma trajetoria latente causal barata.
2. Sem mixer, essa trajetoria e rapida mas fraca em qualidade.
3. O mixer causal local recupera a capacidade de interacao curta que o Anderson b8 parecia trazer.
4. Ao contrario do Anderson, o mixer e altamente paralelizavel e tem autograd simples.
5. A arquitetura fica mais proxima de uma mistura entre state-space/local-conv/latent-trajectory model do que de uma recorrencia pura.

Em termos praticos:

```text
Anderson b8 era bom como laboratorio.
Local mixer causal e melhor como caminho escalavel.
```

## 9. Riscos e cautelas

Ainda nao devemos formular claims absolutos como "DRM e melhor que Transformer".

Claims seguros agora:

- Em 125M, neste dataset e protocolo, o DRM local mixer mostra time-to-quality muito superior ao GPT-2 pareado.
- O resultado se repetiu em 3 seeds no probe 10M.
- O resultado se repetiu em 150M seed 1 e esta se repetindo no seed 2.
- O caminho testado e causal no teste de prefix invariance implementado.

Claims ainda pendentes:

- Superioridade geral contra Transformers modernos.
- Robustez em outros datasets.
- Robustez em tokenizadores diferentes.
- Robustez em escalas maiores, como 350M/500M.
- Qualidade de geracao humana alem de CE.

## 10. Melhorias futuras

### 10.1 Validacao experimental

Prioridade alta:

1. Concluir 150M para seeds 1, 2, 3.
2. Concluir GPT-2 150M pareado para seeds 1, 2, 3.
3. Rodar causality check nos checkpoints finais de todos os seeds DRM.
4. Gerar tabela final com media, mediana, desvio, tempo ate alvo e tokens ate alvo.
5. Rodar amostras de geracao com os checkpoints finais.

### 10.2 Controle de avaliacao

Melhorias recomendadas:

- Aumentar `EvalBatches` para reduzir variancia do CE.
- Fixar e documentar o conjunto de validacao usado.
- Salvar tambem CE em um subset de validacao maior ao final de cada run.
- Rodar uma avaliacao offline separada para todos os checkpoints finais.

### 10.3 Ablacoes do mixer

Executar sweep controlado:

| Eixo | Valores |
|---|---|
| hidden | 64, 128, 256, 384 |
| layers | 1, 2, 3 |
| kernel | 4, 8, 16 |
| scale | 0.05, 0.1, 0.2, 0.3 |
| block size | 32, 64, 128 |

Perguntas:

- O ganho vem do mixer em si ou do aumento de parametros?
- `h128` e suficiente para preservar quase todo o CE?
- `kernel=16` melhora qualidade ou apenas custa mais?
- `block128 + mixer` aumenta throughput mantendo CE?

### 10.4 Igualar budget de parametros

O DRM local mixer tem `127.27M` parametros contra `126.08M` do GPT-2 real. A diferenca e pequena, mas para rigor:

- reduzir `hidden_size` do mixer;
- reduzir `d_state` ou `hidden_size` de algum submodulo;
- ou criar GPT-2 pareado com numero de parametros mais proximo.

### 10.5 Throughput

O caminho atual ainda perde em tokens/s bruto. Possiveis melhorias:

1. `torch.compile` seletivo para `_forward_directional_cumsum`.
2. Fusar calculos do block scan e features do mixer.
3. Testar `block_size=128`.
4. Testar mixer com `Conv1d` grouped/pointwise reordenado.
5. Remover diagnosticos e perdas auxiliares desnecessarias durante treino longo.
6. Avaliar impacto de checkpoint/eval interval no throughput medio.

### 10.6 Arquitetura

Possiveis proximas variantes:

- Mixer gated causal: `z + gate * correction`.
- Mixer residual normalizado: RMSNorm antes/depois do mixer.
- Mixer multi-kernel: kernels 4/8/16 em paralelo.
- Mixer state-space curto em vez de Conv1d.
- Hibrido: mixer rapido em todos os blocos e Anderson real esparso apenas como regularizador/checkpoint.
- Distilacao offline do b8 Anderson para o mixer.

### 10.7 Kernel especializado

O kernel/scan especializado para Anderson ainda nao esta descartado, mas sua prioridade caiu.

Antes, ele era necessario para salvar o b8. Agora ele vira uma opcao de segunda fase:

- implementar se o mixer saturar;
- usar como professor local;
- ou fundir o caminho `block scan + mixer` para aumentar throughput.

## 11. Comandos de reproducao

### 11.1 Validacao completa 10M + causalidade + 150M

```powershell
.\scripts\run_125m_local_mixer_validation_sequence.ps1
```

### 11.2 Check causal manual

```powershell
.\.venv\Scripts\python.exe scripts\check_125m_local_mixer_causality.py `
  --run-dir runs\competition_125m_local_mixer_h256_l2_s02_10m\drm_125m_real_causal_anderson_b64_stepvelocity_andcandidate_trajectory_s1_mixh256_k8_l2_scale0.2_seed_1 `
  --batch-size 2 `
  --seq-len 512 `
  --device cuda `
  --precision bf16 `
  --drm-block-size 64 `
  --drm-anderson-iterations 0 `
  --drm-cumsum-step-mode velocity `
  --drm-local-mixer causal_conv `
  --drm-local-mixer-hidden-size 256 `
  --drm-local-mixer-kernel-size 8 `
  --drm-local-mixer-layers 2 `
  --drm-local-mixer-scale 0.2
```

## 12. Conclusao

A reviravolta principal foi perceber que o Anderson b8 nao precisava ser preservado literalmente.

O que precisava ser preservado era a propriedade funcional:

```text
mistura causal local sobre uma trajetoria latente.
```

O local mixer causal entrega essa propriedade com custo muito menor e resultados muito melhores em 125M. No estado atual, ele e a linha principal do projeto para escala, enquanto Anderson b8 passa a ser ferramenta de estudo, professor/local target ou inspiracao para kernels futuros.

