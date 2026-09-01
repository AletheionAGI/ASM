# Confirmação independente do ASM-C2-FW-LM

## Estado do candidato

A primeira suíte ASM-C2-FW-LM terminou com promoção aprovada por todos os gates
internos. Em sua avaliação principal, o candidato combinou:

- CE linguística `1,326738`, contra `1,342003` do ASM-R de origem;
- 100% de MQAR em 512, 4.096 e 32.768 tokens;
- 4.096 respostas corretas entre 4.096 targets no teste de 32K;
- cache constante de `0,1367 MiB`;
- VRAM e throughput estáveis entre 4K e 32K;
- divergência BF16 de argmax de `0,195%` e erro absoluto médio `0,01846`.

Esse resultado promove o modelo ao estado de **candidato real**, mas não encerra
a validação. As três especializações anteriores partiram do mesmo checkpoint
ASM-R seed 1, e o rescoring linguístico decisório avaliou somente a seed 1 em um
milhão de tokens.

## Protocolo de confirmação

A nova suíte elimina essas duas limitações:

1. usa checkpoints ASM-R independentes das seeds 1, 2 e 3, todos treinados em
   100 milhões de tokens;
2. especializa um ASM-C2-FW-LM separado a partir de cada linhagem;
3. treina ou reutiliza Transformers pareados de 100M tokens para as mesmas três
   seeds;
4. faz rescoring contínuo no split completo da Wikipedia, com 4.834.787 tokens;
5. repete MQAR de 32K com 4.096 targets para cada seed;
6. repete cache, VRAM, throughput e paridade BF16 para cada seed;
7. mede prefill e decode contra o Transformer em comprimentos compartilhados.

O treinamento de especialização adiciona batches MQAR e uma pequena quantidade
de replay linguístico ao checkpoint-base. Portanto, a comparação registra essa
computação adicional e não a descreve como orçamento de treino perfeitamente
idêntico. O orçamento de pré-treinamento dos três backbones permanece pareado em
100M tokens.

## Regra de promoção

A promoção oficial exige simultaneamente:

- três linhagens independentes;
- mesmo hash do corpus congelado e pelo menos 4,8M tokens avaliados;
- currículo MQAR aprovado nas três seeds;
- regressão média e individual contra ASM-R de no máximo `0,05` CE;
- pelo menos 80% de MQAR em 32K nas três seeds;
- pelo menos 4.096 targets MQAR por seed;
- cache limitado, crescimento de VRAM de no máximo 10% e retenção de throughput
  de pelo menos 80%;
- divergência BF16 de argmax de no máximo 1% e erro absoluto médio de no máximo
  `0,02`, em todas as seeds.

A CE do Transformer é comparativa, não um gate. O objetivo da promoção é provar
que o ASM-C2-FW-LM acrescenta memória associativa durável ao ASM-R sem regressão
linguística inaceitável. O protocolo não converte isso numa alegação de que o
ASM supera Transformers em modelagem geral de linguagem.

## Comando único

```bash
./scripts/run_asm_c2_fw_lm_confirmation.sh
```

O runner é reiniciável por artefato: checkpoints e avaliações já concluídos são
reutilizados. As saídas finais ficam em:

```text
runs/asm_c2_fw_lm_confirmation/report.md
runs/asm_c2_fw_lm_confirmation/decision.json
runs/asm_c2_fw_lm_confirmation/charts/
```

Se `promote` for `true`, o resultado autoriza atualizar o ASM Website, os
READMEs, `ARCHITECTURE.md`, a família de modelos e os benchmarks públicos. Se
for `false`, o relatório identifica precisamente o gate que bloqueou a
promoção, sem alterar a arquitetura canônica ASM-R.

## Correção de paridade BF16

A primeira confirmação independente passou todos os gates exceto erro absoluto
médio BF16: `0,01846`, `0,02031` e `0,02835` nas seeds 1, 2 e 3, contra limite
pré-registrado de `0,02`. A divergência de argmax permaneceu abaixo de 1% nas
três seeds.

A análise identificou uma diferença de acumulação dependente da forma: o
forward completo executava blocos e emitter sobre matrizes maiores, enquanto o
decode incremental recomputava o bloco aberto e emitia uma posição. Sob BF16,
kernels com formas diferentes podem selecionar ordens de acumulação diferentes.

A correção mantém treinamento e pesos intactos. Somente em inferência compacta
com memória fast-weight FP32, o núcleo recorrente e o emitter executam fora do
autocast, em FP32. O restante da interface continua aceitando o contexto BF16.
Isso estabelece o mesmo contrato numérico para recomposição e decode.

Para repetir exclusivamente os três testes baratos de paridade e reaplicar a
decisão oficial:

```bash
./scripts/rerun_asm_c2_fw_lm_parity.sh
```
