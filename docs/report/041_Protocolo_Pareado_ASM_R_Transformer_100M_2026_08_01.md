# Protocolo pareado ASM-R versus Transformer em 100M tokens

## Questões

A suíte responde separadamente:

1. Qual arquitetura possui a melhor curva de CE sob os mesmos marcos de tokens?
2. Qual possui maior throughput de prefill?
3. Qual possui maior throughput de decode token a token com cache?
4. Qual consome menos VRAM em inferência?
5. Como CE e throughput variam com o comprimento do contexto?
6. Que diferenças qualitativas aparecem sob prompts e seeds idênticos?

## Controles

- ASM-R: 83.206.400 parâmetros, checkpoint de 100M tokens.
- Transformer: 83.001.240 parâmetros, checkpoint de 100M tokens.
- Wikipedia byte-level, vocabulário 256 e contexto de treino 512.
- Seed de pré-treinamento 1 e precisão BF16.
- Mesmos checkpoints em 1M, 2M, 5M, 10M, 20M, 30M, 50M e 100M.
- Rescoring sobre a mesma sequência contínua de validação.

## Comando único

```bash
./scripts/run_asm_transformer_paired_suite.sh
```

Saídas:

```text
runs/asm_transformer_paired_suite/
├── milestone_rescoring.json
├── paired_benchmark.json
└── report.md
```

O rescoring é a etapa dominante. Na RTX 4090, a estimativa total é de 12 a 20
minutos. O benchmark de inferência deve consumir mais 3 a 8 minutos.

## PG-19 opcional

Se existir um manifest PG-19 preparado no formato memmap do projeto:

```bash
PG19_MANIFEST=data/benchmark_125m_external/test/manifest.json \
./scripts/run_asm_transformer_paired_suite.sh
```

Essa execução acrescenta `pg19_context.json`. Contextos acima de 512 são
extrapolação apenas para ASM-R: o Transformer pareado usa embeddings posicionais
absolutos com limite 512 e deve ser marcado como não suportado, não penalizado
como se tivesse produzido uma previsão comparável.

## Limites

- Uma seed ainda não estabelece significância estatística.
- Parâmetros pareados não significam FLOPs pareados.
- Geração qualitativa serve para inspeção, não ranking isolado.
- O resultado de contexto longo precisa distinguir capacidade arquitetural de
  treinamento efetivo em sequências longas.
