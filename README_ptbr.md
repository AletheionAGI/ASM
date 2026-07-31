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

- [Arquitetura em português](ARCHITECTURE_ptbr.md)
- [Arquitetura em inglês](ARCHITECTURE.md)
- [Roadmap formal em português](roadmap_ptbr.md)
- [Roadmap formal em inglês](roadmap.md)
- [Paper DRM v6](docs/paper/drm_v6.tex)
- [Notas matemáticas](docs/math.md)
- [Protocolo de competição](docs/competition.md)
- [Metodologia e FAQ técnico](docs/TECHNICAL_QA_ptbr.md)
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

O candidato 125M atual é a variante J. Ela usa blocos causais de 64 tokens,
deltas direcionais, naturalização métrica, mixer causal curto, residual
token→estado e memória seletiva forget/write. A memória seletiva é uma adição
posterior inspirada pela literatura de SSMs, não parte do DRM original.

```text
input_ids
  → TokenEmbedding
  → blocos causais de 64 tokens
  → deltas direcionais paralelos
  → prefix cumsum
  → causal local mixer
  → residual token→estado
  → memória seletiva forget/write
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

- `src/drm_language_emitter/config.py`: schema validado do `DRMConfig`.
- `src/drm_language_emitter/model.py`: montagem do modelo e forward recorrente principal.
- `src/drm_language_emitter/model_components.py`: inicializador, mixer causal, transição direta de controle e memória seletiva.
- `src/drm_language_emitter/selective_control.py`: controle de memória seletiva sem geometria.
- `src/drm_language_emitter/mqar.py`: dados sintéticos de associative recall.
- `src/drm_language_emitter/direction_field.py`: campos direcionais ativos e gates.
- `src/drm_language_emitter/metric.py`: métrica relacional `diag + U Uᵀ` e naturalização.
- `src/drm_language_emitter/dynamics.py`: fluxo direcional e atualização de estado.
- `src/drm_language_emitter/directional_forward.py`: forward directional cumsum, losses e diagnósticos.
- `src/drm_language_emitter/directional_blocks.py`: construção de trajetórias em blocos e superblocos.
- `src/drm_language_emitter/directional_solvers.py`: fixed point, Anderson causal e helpers de transição.
- `src/drm_language_emitter/geometric_steps.py`: limitação de estado, refinamento geodésico e candidatos direcionais.
- `src/drm_language_emitter/deer.py`: solvers reutilizáveis de trajetória.
- `src/drm_language_emitter/emitter.py`: embedding de tokens e cabeça emissora.
- `src/drm_language_emitter/generation.py`: geração autoregressiva.
- `src/drm_language_emitter/data.py`: datasets em memória e memory-mapped.
- `src/drm_language_emitter/checkpoint.py`: carregamento validado de checkpoints weights-only.
- `transformer/`: baselines Transformer.
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

## Comparações GPT-2 retratadas

As comparações anteriormente publicadas de 125M/150M tokens e time-to-quality
36M contra GPT-2 estão **deprecated e retratadas como evidência comparativa**.

O treinamento histórico deslocava os targets antes de fornecê-los à
implementação causal da Hugging Face, que já realiza seu próprio shift interno.
Assim, o GPT-2 foi treinado para prever \(x_{t+2}\), não \(x_{t+1}\). Esse
double-shift piorou artificialmente seu CE. O bug foi corrigido em
`scripts/train_gpt2_memmap.py` e possui testes de regressão.

Consequências:

- as alegações antigas de vitória do DRM sobre GPT-2 36M ou 125M foram
  retiradas;
- conclusões antigas de target e time-to-quality são inválidas;
- os artefatos permanecem apenas para auditoria e histórico;
- curvas DRM podem descrever aqueles runs, mas não sustentam a comparação;
- este README não afirma superioridade sobre GPT-2.

Artefatos históricos, inválidos para comparação:

```text
docs/benchmarks/competition_125m_local_mixer_h256_l2_s02_150m/
docs/benchmarks/tta/
```

## Evidência interna atual

O resultado controlado atual usa 5M tokens, três seeds e validação contínua
determinística sobre 4.834.787 targets:

| Variante | Parâmetros | CE médio | Desvio | Interpretação |
|---|---:|---:|---:|---|
| I | 127,01M | 1,878244 | 0,000647 | baseline geométrico |
| J | 126,08M | **1,760581** | 0,003057 | geometria + memória seletiva |
| SSM_CONTROL | 126,08M | 1,806518 | 0,006191 | memória sem geometria |

J venceu SSM_CONTROL nas três seeds por 0,045937 CE em média, enquanto o
controle treinou aproximadamente 2,5 vezes mais rápido. Isso sustenta uma
contribuição do sistema geométrico nesse controle interno; não é comparação
com Mamba ou GPT-2.

Veja o
[relatório 027](docs/report/027_Contribuicao_Geometrica_J_vs_SSM_Control_e_Proximas_Ablacoes_2026_07_31.md).

## Pipeline independente 125M

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

Executar ou retomar os treinamentos. Os baselines corrigidos usam diretórios
`gpt2_125m_real_next_token_seed_*`; checkpoints antigos
`gpt2_125m_real_seed_*` não devem ser retomados:

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
- as comparações antigas com GPT-2 estão retratadas por double-shift;
- J venceu SSM_CONTROL nas três seeds do controle interno de 5M tokens.

Afirmações não permitidas:

- DRM é superior a Transformers em geral;
- os benchmarks GPT-2 retratados demonstram superioridade do DRM;
- `dimD` atual é igual ao rank formal `rank(g)` do paper;
- o modelo implementa kernel, estratos, anchor, conexão ou holonomia formal;
- baixa ação prova geodésicas emergentes;
- boundedness ou recorrência prova topologia toroidal;
- checkpoints base são modelos de chat;
- o sistema está pronto para produção ou avaliado em segurança.

## Limitações

- o caminho recorrente básico é lento frente a kernels Transformer otimizados;
- ainda não existe comparação multiseed corrigida e concluída que estabeleça
  vantagem do DRM sobre GPT-2;
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
