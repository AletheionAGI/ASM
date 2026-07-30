# DRM Language Emitter

**Um laboratório de modelos de linguagem orientado por geometria, sem
attention, sem Q/K/V e sem blocos Transformer.**

![Banner de variedade do DRM Language Emitter](assets/drm-language-emitter-banner.svg)

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-pure%20torch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](pyproject.toml)
[![Não Transformer](https://img.shields.io/badge/arquitetura-não--Transformer-14B8A6?style=for-the-badge)](ARCHITECTURE.md)
[![Sem Attention](https://img.shields.io/badge/attention-nenhuma-0F172A?style=for-the-badge)](tests/test_no_transformer.py)
[![Licença](https://img.shields.io/badge/licença-AGPL--3.0-64748B?style=for-the-badge)](LICENSE)

O DRM Language Emitter representa geração de linguagem como movimento
controlado em um espaço relacional aprendido. Direções ativas determinam onde
o estado pode se mover, uma métrica aprendida atribui custo ao movimento e um
emissor transforma o estado resultante em logits de tokens.

Este repositório é uma infraestrutura de pesquisa. Não é um modelo de produção
e não sustenta uma alegação geral de superioridade sobre Transformers ou world
models.

Versão principal em inglês: [README.md](README.md).

## Links rápidos

- [Arquitetura](ARCHITECTURE.md)
- [Roadmap formal em português](roadmap_ptbr.md)
- [Roadmap formal em inglês](roadmap.md)
- [Paper DRM v6](docs/paper/drm_v6.tex)
- [Notas matemáticas](docs/math.md)
- [Protocolo de competição](docs/competition.md)
- [Metodologia e FAQ técnico](docs/TECHNICAL_QA.md)
- [Artefatos de benchmark](docs/benchmarks/README.md)
- [Índice cronológico dos scripts](scripts/INDEX.md)
- [Model card](MODEL_CARD.md)
- [Limitações](docs/limitations.md)
- [Referência da API](docs/api.md)
- [Checklist de conformidade](docs/compliance_checklist.md)
- [Licenças e proveniência de dados](docs/third_party_licenses.md)

## O que diferencia o projeto

O DRM Language Emitter não usa:

- blocos Transformer;
- self-attention;
- attention Q/K/V;
- `nn.MultiheadAttention`;
- KV cache.

O caminho recorrente básico é:

```text
token e_t
  → estado latente z_t
  → direções D(z_t)
  → gates direcionais a_i(z_t)
  → métrica G(z_t) = diag + U Uᵀ
  → velocidade dz no span das direções
  → estado z_{t+1}
  → logits
```

O caminho 125M atualmente mais forte usa blocos causais de 64 tokens. Deltas
direcionais de velocidade são calculados em paralelo, prefixos são
reconstruídos com soma cumulativa e um mixer convolucional causal aplica uma
correção local barata.

```text
input_ids
  → TokenEmbedding
  → blocos causais de 64 tokens
  → deltas direcionais paralelos
  → prefix cumsum
  → causal local mixer
  → LanguageEmitter
  → logits
```

Essa arquitetura continua autoregressiva: tokens futuros não podem alterar os
logits ou estados do prefixo. Existem testes específicos de causalidade.

## Instalação

```bash
pip install -e .
```

Ferramentas de desenvolvimento:

```bash
pip install -e ".[dev]"
```

O projeto funciona em CPU para testes pequenos. CUDA é recomendada para os
benchmarks memmap de 125M parâmetros.

## Início rápido

Treinar um modelo DRM pequeno:

```bash
python scripts/train_tiny.py --config configs/tiny.yaml --text data/tiny.txt
```

Gerar texto:

```bash
python scripts/generate.py \
  --checkpoint runs/tiny/drm_tiny.pt \
  --prompt "DRM "
```

Executar diagnósticos geométricos:

```bash
python scripts/eval_geometry.py --checkpoint runs/tiny/drm_tiny.pt
python scripts/eval_geodesic_paths.py --checkpoint runs/tiny/drm_tiny.pt
```

Se `data/tiny.txt` não existir, o treinamento pequeno cria um corpus mínimo.
O tokenizer padrão opera sobre bytes UTF-8.

## Componentes principais

- `src/drm_language_emitter/config.py`: configuração validada do DRM.
- `src/drm_language_emitter/direction_field.py`: direções e gates ativos.
- `src/drm_language_emitter/metric.py`: métrica `diag + U Uᵀ`.
- `src/drm_language_emitter/dynamics.py`: fluxo e atualização de estado.
- `src/drm_language_emitter/deer.py`: fixed point e Anderson causal.
- `src/drm_language_emitter/model.py`: modelo de linguagem causal.
- `src/drm_language_emitter/generation.py`: geração autoregressiva.
- `transformer/`: baseline Transformer.
- `world_model/`: baseline simbólico seq2seq.

## Diagnósticos

O código registra:

- cross-entropy e perplexidade aproximada;
- ação métrica;
- soma dos gates direcionais `dimD`;
- entropia e frações ativas dos gates;
- norma low-rank e proxy de condição da métrica;
- proxies de recorrência e estabilidade;
- diagnósticos de trajetórias de baixa ação.

Ressalvas:

- `dimD` não é o rank métrico formal do paper. A métrica atual é estritamente
  positiva definida e tem rank matemático completo.
- `eval_geodesic_paths.py` não é um solver geodésico formal.
- diagnósticos blockwise são aproximações de engenharia.
- a geração atual ainda precisa alcançar paridade completa com o caminho
  block-cumsum/local-mixer usado no treinamento 125M.

## Benchmark 125M versionado

O benchmark já publicado contém três seeds por modelo e 150 milhões de tokens
por seed:

| Modelo | Parâmetros | Melhor CE de validação, média | Desvio | Tokens/s |
|---|---:|---:|---:|---:|
| DRM block64 + mixer causal | 127,27M | 1,3116 | 0,0019 | 10.678,7 |
| GPT-2 real | 126,08M | 1,7305 | 0,0259 | 41.224,4 |

Artefatos:

```text
docs/benchmarks/competition_125m_local_mixer_h256_l2_s02_150m/
```

Esse resultado demonstra um sinal experimental sob o protocolo antigo, mas
não constitui superioridade geral. Os melhores checkpoints DRM exatos não
foram preservados naquele experimento.

## Validação independente 125M

O protocolo atual usa:

- treino e validação da Wikipédia separados por documento;
- remoção de duplicatas antes da tokenização;
- PG-19 oficial como teste externo congelado;
- auditorias de contaminação com zero blocos compartilhados;
- três seeds DRM e três seeds GPT-2;
- orçamento fixo de 150 milhões de tokens por run;
- `checkpoint_best.pt`;
- uma única avaliação externa por checkpoint selecionado.

Arquivos principais:

```text
configs/independent_125m_protocol.json
scripts/run_independent_125m_smoke.sh
scripts/run_independent_125m_benchmark.sh
scripts/evaluate_frozen_test.py
docs/report/023_Proposta_Validacao_Independente_125M_2026_07_30.md
```

Executar o smoke test:

```bash
./scripts/run_independent_125m_smoke.sh
```

Executar ou retomar os seis treinamentos:

```bash
./scripts/run_independent_125m_benchmark.sh
```

Não consultar o PG-19 antes de concluir os runs e selecionar checkpoints pela
validação.

Observação metodológica: avaliações intermediárias de apenas quatro batches
são ruidosas. Antes do teste externo, os checkpoints candidatos devem ser
comparados sobre uma varredura determinística e idêntica da validação para
DRM e GPT-2. Qualquer ajuste ao protocolo congelado deve ser documentado.

## Outros benchmarks

### Time-to-quality 37M

No experimento versionado de aproximadamente 37M parâmetros, DRM causal
Anderson b8 atingiu o target nas três seeds; o GPT-2 36M não o atingiu dentro
do piso de tokens testado.

```text
docs/benchmarks/tta/
```

### DRM versus Transformer pequeno

```bash
python scripts/sweep_drm_transformer.py \
  --steps 1000 2000 3000 \
  --seeds 1 2 3 \
  --output-root runs/sweep_drm_transformer
```

### World model simbólico

```bash
python scripts/make_tiny_world_dataset.py \
  --output-root data/tiny_world \
  --seed 1 \
  --grid-size 5 \
  --num-train 20000 \
  --num-val 2000 \
  --max-rollout-len 8

python scripts/sweep_world_model_competition.py \
  --steps 1000 2000 3000 \
  --seeds 1 2 3 \
  --dataset-root data/tiny_world \
  --output-root runs/world_model_competition
```

Os resultados simbólicos ainda têm baixa acurácia absoluta e são apenas
diagnósticos.

## Testes

```bash
python -m pytest -q
```

Testes CUDA são condicionais e só executam quando
`torch.cuda.is_available()` é verdadeiro.

## Mapa do repositório

```text
configs/                  configurações do DRM e benchmarks
docs/                     paper, matemática, relatórios e benchmarks
scripts/                  treino, avaliação e preparação de dados
src/drm_language_emitter/ pacote principal do DRM
tests/                    testes e invariantes
transformer/              baseline Transformer
world_model/              baseline simbólico
```

## Estado científico

Afirmações permitidas:

- o projeto implementa um protótipo funcional de LM não Transformer;
- a geometria neural é explícita, mensurável e treinável;
- o caminho blockwise causal preserva causalidade de prefixo;
- os benchmarks versionados mostram vantagem de CE do DRM nos protocolos
  específicos testados.

Afirmações não permitidas:

- DRM é superior a Transformers em geral;
- `dimD` atual é igual ao rank formal `rank(g)` do paper;
- o modelo implementa kernel, estratos, anchor, conexão ou holonomia formal;
- baixa ação prova geodésicas emergentes;
- boundedness ou recorrência prova topologia toroidal;
- checkpoints base são modelos de chat;
- o sistema está pronto para produção ou avaliado em segurança.

## Limitações

- o caminho recorrente básico é lento frente a kernels Transformer otimizados;
- o DRM 125M ainda é mais lento que GPT-2 em tokens por segundo;
- benchmarks dependem do corpus, tokenizer, hardware e protocolo;
- geração e forward do local mixer ainda precisam de uma API unificada;
- a métrica atual é SPD e não possui kernel formal;
- não há SFT, RLHF, alignment ou safety evaluation;
- as Fases 1–6 do paper ainda não estão implementadas.

## Roadmap

O projeto está no **Gate 0: validação independente e fidelidade de runtime**.
A próxima fase formal é rank métrico, kernel, fibra efetiva e estratos.

Leia [roadmap_ptbr.md](roadmap_ptbr.md).

## Licença

Copyright © 2026 Felipe Maya Muniz
SPDX-License-Identifier: AGPL-3.0-only

Licenciamento duplo:

- GNU AGPL v3.0 only; ou
- licença comercial separada.

Consulte [LICENCE-COMMERCIAL.md](LICENCE-COMMERCIAL.md).
