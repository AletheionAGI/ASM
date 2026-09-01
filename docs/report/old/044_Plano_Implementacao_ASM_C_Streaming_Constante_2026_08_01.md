# Plano de implementação ASM-C — streaming de memória constante

## Nome da variante

ASM-C significa:

> **Aletheion Compact State Model**

Em português:

> **Modelo de Estado Compacto Aletheion**

O sufixo `C` representa diretamente as propriedades que a variante introduz ou
investiga:

- estado compacto;
- cache limitado;
- emissão local;
- streaming contínuo;
- custo incremental potencialmente constante.

A família experimental fica organizada assim:

| Sigla | Nome |
|---|---|
| ASM-R | Aletheion Relational State Model |
| ASM-S | Aletheion Selective State Model |
| ASM-F | Aletheion Relational Frame Model |
| ASM-X | Aletheion Explicit Relational Model |
| ASM-C | Aletheion Compact State Model |

Uma alternativa seria **Aletheion Constant-Memory State Model**, mas esse nome
prometeria uma propriedade ainda não confirmada. `Compact` é a denominação mais
precisa enquanto a suíte de validação está em andamento.

Se todos os critérios de promoção forem satisfeitos, a definição pública poderá
ser ampliada para:

> **ASM-C is a compact recurrent state model for bounded-memory continuous
> streaming.**

Se algum critério falhar, ASM-C continuará corretamente descrito como uma
variante compacta experimental, sem alegar memória ou custo estritamente
constantes antes da evidência.

## Objetivo

ASM-C será a variante experimental que transforma a recorrência conceitual do
ASM-R em uma propriedade operacional verificável: cache limitado, temporários
limitados e custo por token independente do comprimento já processado.

## Fase 1 — emitter local

Remover a matriz zero de forma `[batch, prefix_length, d_state]` criada em
`decode_step`. O emitter deverá receber somente o último estado necessário para
o próximo token, com forma `[batch, 1, d_state]`.

Como a implementação anterior preserva o formato do GEMM para paridade BF16, a
mudança exige três verificações:

1. paridade exata em FP32;
2. erro BF16 explicitamente quantificado;
3. ausência de mudança de argmax em uma amostra representativa, ou tolerância
   documentada quando a diferença ocorrer perto de empates numéricos.

## Fase 2 — estado compacto

Substituir `InferenceState.input_ids` completos por:

```text
tokens_seen: int
completed_state: Tensor[batch, d_state]
block_tokens: Tensor[batch, < block_size]
block_index: int
block_size: int
```

O fallback que requer prefixo completo deve continuar disponível para modos não
streaming, mas ASM-C deverá recusar silenciosamente nenhum fallback: se a
configuração não puder operar compactamente, deverá produzir erro explícito.

## Fase 3 — invariantes e testes

- `cache_tensor_bytes` estabiliza após completar o primeiro bloco.
- Decode em 512, 4K e 32K mantém os mesmos logits de referência dentro da
  tolerância definida.
- O estado nunca contém tensor com dimensão proporcional a `tokens_seen`.
- O pico temporário entre 4K e 32K cresce no máximo 10%.
- Throughput de 32K preserva pelo menos 90% do throughput medido em 4K.
- Testes cobrem fechamento de bloco, bloco aberto e batch maior que um.

## Fase 4 — MQAR corrigido

Após 5.000 passos de adaptação curta, avaliar primeiro comprimento 40. A suíte
só poderá falar em esquecimento se o controle atingir um limiar pré-definido,
inicialmente 80% de acurácia.

Depois, avaliar 512, 1K, 2K, 4K, 8K, 16K e 32K com pelo menos 4.096 targets por
ponto. Para acaso `p = 1/64`, isso reduz o erro-padrão para aproximadamente 0,19
ponto percentual, contra cerca de 1,55 ponto com apenas 64 targets.

Também registrar:

- acurácia e CE;
- curva por distância;
- número de pares e interferência;
- degradação relativa ao controle curto;
- intervalo de confiança binomial.

## Fase 5 — comparação justa

Comparar ASM-C com:

1. ASM-R atual, para isolar a correção de inferência;
2. Transformer 512 com KV cache, na região compartilhada;
3. Transformer de contexto longo com RoPE ou posições treinadas, em 4K–32K;
4. controle recorrente simples com estado do mesmo tamanho.

Métricas principais: CE, MQAR, prefill, decode, cache persistente, pico de VRAM e
FLOPs aproximados por token.

## Ordem de execução

1. implementar emitter local;
2. introduzir `tokens_seen` e estado compacto;
3. adicionar testes de paridade e invariantes;
4. corrigir MQAR e aumentar a amostra;
5. repetir a suíte 512–32K;
6. promover o nome ASM-C somente se os critérios de memória e throughput forem
   satisfeitos.

Essa fase não promete vencer o Transformer em CE. Sua hipótese falsificável é
mais específica: oferecer streaming longo com cache e custo incremental
limitados, preservando uma quantidade mensurável de informação distante.

## Estado de implementação

As cinco fases foram implementadas em 1º de agosto de 2026 e permanecem
experimentais até a execução da suíte:

- `compact_streaming_inference` preserva o ASM-R como comportamento padrão;
- `build_compact_streaming` expõe ASM-C como variante nomeada;
- o emitter compacto recebe somente `[batch, 1, d_state]` no decode;
- o estado compacto retém contador, estado concluído e bloco aberto;
- modos sem fronteira fixa falham explicitamente;
- testes verificam paridade FP32, forma do emitter e cache limitado;
- a paridade BF16 do checkpoint real é medida e salva;
- MQAR inclui controle em 40 tokens, 4.096 targets por ponto e IC Wilson de 95%;
- o runner compara ASM-R, ASM-C e o Transformer pareado.

Comando de validação:

```bash
./scripts/run_asm_c_validation_suite.sh
```
