# Proposta de Validação Independente do Benchmark 125M

Data: 2026-07-30  
Status: corpora preparados e auditados; protocolo aguarda hardware e target CE
para ser congelado.

## Objetivo

Confirmar se a vantagem observada pelo DRM 125M com local mixer causal se mantém
em dados externos e nunca utilizados para treinamento, seleção de checkpoint ou
ajuste de hiperparâmetros.

O benchmark só será considerado validado quando DRM e GPT-2 forem avaliados
sobre os mesmos tokens de teste, com protocolo definido antes das execuções e
sem sobreposição detectável entre treino, validação e teste.

## Princípios

1. Separar documentos antes da tokenização.
2. Usar fontes ou partições independentes para treino, validação e teste.
3. Usar validação somente para selecionar checkpoints.
4. Acessar o test set uma única vez por checkpoint selecionado.
5. Avaliar janelas determinísticas idênticas para todos os modelos.
6. Somar cross-entropy por token, evitando médias de médias.
7. Registrar hashes de checkpoint, manifest e shards.
8. Congelar seeds, target, budget e regra de parada antes do treinamento.

## Layout proposto

```text
data/benchmark_125m/
├── train/manifest.json
├── validation/manifest.json
├── test/manifest.json
└── provenance.json

docs/benchmarks/independent_125m/
├── protocol.json
├── contamination_report.json
├── runs.csv
├── aggregate.csv
├── test_results.json
└── README.md
```

## Protocolo mínimo

- Modelos: DRM 125M real local mixer e GPT-2 125M real.
- Seeds: 1, 2 e 3.
- Budget: 150 milhões de tokens por seed.
- Seleção: menor CE no validation set.
- Teste: 10 milhões de tokens congelados.
- Métrica primária: test CE.
- Métricas secundárias: perplexidade, throughput, memória, GPU-horas e
  time-to-quality na validação.
- Parada: budget fixo de tokens.
- Contaminação permitida: zero blocos exatos compartilhados de 512 bytes,
  medidos com stride 256.

O template versionado está em:

```text
configs/independent_125m_protocol.json
```

Todos os campos `null` devem ser preenchidos, o commit deve ser fixado e o
status deve mudar de `draft` para `frozen` antes do início dos treinamentos.

## Infraestrutura implementada

### Preparação documental

O script `scripts/prepare_independent_benchmark.py`:

- recebe entradas explícitas para train, validation e test;
- aceita arquivos TXT, diretórios recursivos e JSONL;
- interpreta blocos separados por linha vazia como documentos em TXT;
- normaliza NUL e whitespace sem remover caracteres Unicode;
- calcula SHA-256 do documento normalizado;
- remove duplicatas dentro de cada split e entre splits;
- aplica precedência conservadora `train -> validation -> test`;
- rejeita documentos abaixo do tamanho mínimo;
- tokeniza somente depois da deduplicação;
- gera shards com nomes e labels próprios para cada split;
- registra inputs, hashes, contagens e decisões em `provenance.json`;
- registra cada documento aceito ou descartado em `document_records.jsonl`.

Exemplo:

```bash
python scripts/prepare_independent_benchmark.py \
  --train-input corpora/train/ \
  --validation-input corpora/validation/ \
  --test-input corpora/test.jsonl \
  --output-root data/benchmark_125m \
  --min-doc-chars 200 \
  --source-name "nome do corpus" \
  --source-version "versao congelada" \
  --source-url "https://origem.example/dataset" \
  --license "identificador da licenca"
```

Se o JSONL usar outro campo, deve-se passar `--text-field`. O diretório de
saída não é sobrescrito sem `--overwrite`.

### Auditoria de contaminação

O script `scripts/audit_dataset_contamination.py`:

- recebe dois ou mais manifests nomeados;
- lê shards em streaming, inclusive através de suas fronteiras;
- calcula SHA-256 de blocos sobrepostos;
- usa SQLite temporário para limitar uso de RAM;
- rejeita path traversal nos manifests;
- gera relatório JSON e retorna código 2 quando encontra sobreposição.

Exemplo:

```bash
python scripts/audit_dataset_contamination.py \
  --manifest train data/benchmark_125m/train/manifest.json \
  --manifest validation data/benchmark_125m/validation/manifest.json \
  --manifest test data/benchmark_125m/test/manifest.json \
  --block-size 512 \
  --stride 256 \
  --output docs/benchmarks/independent_125m/contamination_report.json
```

### Avaliação congelada

O script `scripts/evaluate_frozen_test.py`:

- suporta checkpoints DRM e GPT-2;
- usa `weights_only=True`;
- valida a integridade dos shards;
- percorre o dataset sequencialmente;
- respeita um limite exato de tokens;
- calcula CE pela soma das losses individuais;
- registra SHA-256, commit, versões e dispositivo;
- não seleciona checkpoint nem altera hiperparâmetros.

Exemplo DRM:

```bash
python scripts/evaluate_frozen_test.py \
  --family drm \
  --checkpoint runs/drm_seed_1/checkpoint_best.pt \
  --manifest data/benchmark_125m/test/manifest.json \
  --split test \
  --seq-len 512 \
  --max-tokens 10000000 \
  --device cuda \
  --output docs/benchmarks/independent_125m/drm_seed_1_test.json
```

## Sequência de execução

1. Escolher e documentar as fontes de dados.
2. Separar documentos e remover duplicatas antes da tokenização.
3. Gerar manifests independentes.
4. Executar a auditoria de contaminação.
5. Preencher e congelar o protocolo.
6. Treinar os seis runs sem consultar o test set.
7. Selecionar cada `checkpoint_best.pt` pela validação.
8. Executar uma avaliação congelada por checkpoint.
9. Agregar média, desvio-padrão e diferença DRM menos GPT-2.
10. Publicar resultados positivos ou negativos sem alterar o protocolo.

## Critérios de aprovação

- auditoria de contaminação aprovada;
- três seeds completos por modelo;
- mesmo conjunto e mesma ordem de tokens no teste;
- vantagem de test CE consistente entre seeds;
- intervalo de confiança da diferença sem cruzar zero;
- resultados reproduzíveis pelo commit e hashes registrados.

## Execução realizada

### PG-19 como fonte externa

Foi escolhido o split oficial de teste do PG-19, publicado pelo Google DeepMind
em `gs://deepmind-gutenberg/test/`. O novo
`scripts/download_pg19_test.py` exige os 100 livros esperados e verifica tamanho
e MD5 de cada objeto.

Resultado:

- 100 livros e 41.289.101 bytes nos objetos oficiais;
- SHA-256 do JSONL:
  `4988d639d6f1ccbb5918557f0e3c4ac8fc76bb7678c6b90fb140cd00d8aa44de`;
- 39.898.769 byte-tokens após normalização;
- SHA-256 do corpus:
  `610d4f67cca125e220a6612928cfc7dd396736f84008af3cb0e64f73c6a96e5e`.

### Split documental da Wikipédia

O novo `scripts/prepare_wikipedia_document_split.py` atribui documentos antes
da tokenização usando os primeiros 64 bits do SHA-256 normalizado. Com fração
de validação `0,001`, foram obtidos:

- treino: 1.531.257 documentos e 5.014.760.375 byte-tokens;
- validação: 1.538 documentos e 4.834.788 byte-tokens;
- 83 duplicatas exatas removidas;
- SHA-256 de treino:
  `7c0a78cebc7e744b31034c4e25cefef1456144d95c3aa86b7114b43d5bc95127`;
- SHA-256 de validação:
  `9629998457a98582f275f86c41a56915d3b25e688f217bb079fbfc28c0afb581`.

### Resultado das auditorias

| Comparação | Blocos A | Blocos B | Sobreposições |
|---|---:|---:|---:|
| Wikipédia treino × validação | 19.588.906 | 18.884 | 0 |
| Wikipédia original × PG-19 teste | 19.608.056 | 155.853 | 0 |

Ambas passaram com blocos de 512 bytes e stride 256. Os relatórios JSON estão
em `docs/benchmarks/independent_125m/`. Os corpora gerados são ignorados pelo
Git devido ao tamanho e podem ser reconstruídos pelos scripts.

## Bloqueios atuais

Os dados, manifests e auditorias estão prontos. Antes dos seis treinamentos,
restam valores que não podem ser inferidos com segurança:

- descrever a GPU/CPU/RAM efetivamente reservadas;
- fixar antecipadamente o target CE de validação;
- registrar o commit final e mudar o protocolo para `status=frozen`.

Depois disso, devem ser executadas três seeds por arquitetura, selecionando
checkpoints somente pela validação e consultando PG-19 uma única vez por
checkpoint escolhido.

## Validação da implementação

Foram executados:

```text
python -m compileall -q src transformer world_model scripts tests
python -m json.tool configs/independent_125m_protocol.json
.venv/bin/python -m pytest
git diff --check
```

Resultado:

```text
73 passed, 1 skipped
```

Os testes novos cobrem corpus sem contaminação, detecção de blocos
compartilhados, determinismo da avaliação, respeito ao limite exato de tokens
preparação ponta a ponta de TXT/JSONL com deduplicação antes da tokenização e
split documental determinístico da Wikipédia.
O teste ignorado depende de CUDA e já fazia parte da suíte anterior.
