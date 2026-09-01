# Suíte de avaliação pós-promoção do ASM-R

**Data:** 1 de agosto de 2026

## Objetivo

Transformar a validação do ASM-R em um processo reproduzível de um único
comando. A suíte não realiza novo treinamento de linguagem. Ela avalia um
checkpoint promovido e executa uma adaptação sintética curta para medir MQAR.

## Ordem de execução

1. testes de regressão de checkpoint e inferência;
2. causalidade de prefixo no checkpoint real;
3. rescoring determinístico da validação;
4. auditoria de finitude e identidade ASM-R;
5. geração com três prompts e seeds fixos;
6. CE em múltiplos comprimentos de contexto;
7. adaptação supervisionada MQAR;
8. benchmark de decode incremental e paridade numérica;
9. consolidação em JSON e Markdown.

## Interpretação correta

O teste MQAR começa medindo o checkpoint sem adaptação e depois treina uma cópia
em memória por um orçamento curto. O valor posterior mede facilidade de
adaptação e associative recall sob supervisão. Ele não deve ser descrito como
MQAR zero-shot aprendido no corpus Wikipedia.

Comprimentos acima de 512 também são probes de extrapolação. O modelo foi
treinado com sequências de 512 tokens; bom resultado acima disso não equivale a
treinamento explícito em contexto longo.

## Modos

`--quick` valida a infraestrutura com rescoring parcial, poucos batches de
contexto, geração curta e dez passos MQAR.

`--full` usa toda a validação disponível, 16 batches por comprimento, contextos
até 2.048 tokens, geração maior e 200 passos MQAR.

## Comandos

Primeiro execute o smoke rápido:

```bash
./scripts/run_asm_r_post_promotion_suite.sh \
  --checkpoint runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt \
  --output-root runs/asm_r_post_promotion_quick \
  --device cuda \
  --quick
```

Se ele terminar sem erro, execute a avaliação completa:

```bash
./scripts/run_asm_r_post_promotion_suite.sh \
  --checkpoint runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt \
  --output-root runs/asm_r_post_promotion_full \
  --device cuda \
  --full
```

As seeds 2 ou 3 podem ser avaliadas trocando apenas o caminho do checkpoint.

## Curva MQAR ampliada

Após a suíte completa, a curva contínua de adaptação MQAR pode ser executada
com:

```bash
./scripts/run_asm_r_mqar_curve.sh
```

Ela avalia o mesmo modelo nos passos 0, 200, 500, 1.000, 2.000 e 5.000 sem
reiniciar os pesos. O resultado é salvo em:

```text
runs/asm_r_mqar_curve_5k/results.json
```

Outro checkpoint ou diretório pode ser selecionado por variáveis de ambiente:

```bash
CHECKPOINT=runs/asm_confirmation_100m_seed2/variant_j_no_direction_seed_2/checkpoint_milestone_100000000.pt \
OUTPUT=runs/asm_r_mqar_curve_seed2/results.json \
./scripts/run_asm_r_mqar_curve.sh
```

## Artefatos

```text
runs/asm_r_post_promotion/
├── causality.json
├── validation.json
├── checkpoint_evaluation.json
├── incremental_decode.json
├── summary.json
└── report.md
```

O arquivo `configs/asm_r_125m.json` congela a configuração canônica promovida,
derivada do `resolved_config.json` da execução ASM-R confirmada.
