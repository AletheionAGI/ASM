# Cache incremental de inferência ASM

**Data:** 31 de julho de 2026
**Estado:** implementado e validado, ainda sem commit
**Arquitetura medida:** ASM-R, checkpoint de 100M tokens

## Objetivo

Eliminar a recomposição de todo o histórico a cada token gerado sem criar uma
dinâmica diferente daquela usada no treinamento.

## Estratégia

As variantes promovidas usam blocos causais fixos de 64 tokens. O novo
`InferenceState` mantém:

- prefixo de tokens para proveniência e fallback;
- estado final do último bloco concluído;
- tokens do bloco ainda aberto;
- índice do bloco;
- tamanho fixo do bloco.

Ao gerar um token:

1. o token é acrescentado ao bloco aberto;
2. somente esse bloco é recomposto pelo mesmo `_directional_cumsum_block()` usado
   no forward de treino;
3. ao completar 64 tokens, seu estado final é congelado como início do próximo
   bloco;
4. o buffer causal volta a ficar vazio.

Assim, o custo da transição é limitado ao tamanho do bloco, em vez de crescer com
todo o contexto.

Modos sem tamanho de bloco explícito permanecem automaticamente no caminho de
referência que recompõe o prefixo inteiro.

## Paridade BF16 do emitter

O primeiro benchmark encontrou estados de fronteira idênticos, mas diferença de até
0,125 nos logits BF16. A causa não era o cache: kernels GEMM BF16 usavam arredondamento
diferente quando o emitter recebia uma única linha em vez da forma completa
`[batch, sequence, state]`.

A correção preserva a forma matricial completa na projeção do emitter, preenchendo
com zeros as linhas anteriores. Como o emitter opera independentemente por posição,
essas linhas não alteram o último logit, mas fazem o kernel usar a mesma geometria
matricial do forward completo.

Após a correção:

```text
max_abs_error  = 0.0
mean_abs_error = 0.0
```

## Benchmark na RTX 4090

Checkpoint:

```text
runs/asm_scaling_law_100m_seed1/
variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt
```

Protocolo:

- batch 1;
- prompt de 128 tokens;
- decode de 32 tokens conhecidos;
- BF16;
- mesma continuação para referência e cache;
- sincronização CUDA antes e depois de cada medição.

Resultado:

| Caminho | Tempo | Tokens/s |
|---|---:|---:|
| Recomposição completa | 0,218819 s | 146,240 |
| Cache incremental | 0,075333 s | 424,778 |

Ganho:

```text
2,905 vezes
```

O ganho cresce conceitualmente com o contexto até que outros custos, especialmente
o emitter com forma completa e movimentação de buffers, passem a dominar.

## Cobertura

Os testes agora verificam:

- igualdade com o forward completo em cada passo;
- cruzamento de múltiplas fronteiras de bloco;
- buffer aberto sempre menor que o tamanho do bloco;
- ASM-R, ASM-S e ASM-F;
- fallback correto para modos sem bloco fixo;
- geração das sete variantes públicas ASM.

Resultado global:

```text
109 passed, 1 skipped
```

## Reprodução

```bash
.venv/bin/python scripts/benchmark_incremental_decode.py \
  --checkpoint runs/asm_scaling_law_100m_seed1/variant_j_no_direction_seed_1/checkpoint_milestone_100000000.pt \
  --prompt-tokens 128 \
  --decode-tokens 32 \
  --batch-size 1 \
  --precision bf16 \
  --device cuda
```

## Limitações

- O prefixo completo ainda é preservado no estado para fallback e depuração.
- O emitter conserva a forma completa para paridade BF16, portanto essa parte ainda
  cresce com o comprimento do contexto.
- A medição apresentada usa um único comprimento, batch e checkpoint.
- O cache não acelera modos cujo tamanho de bloco é dinâmico ou zero.
- Ainda não há benchmark separado de latência de prefill.

## Próxima otimização

Investigar uma projeção do emitter numericamente estável e independente da forma do
batch, ou aceitar uma tolerância BF16 explicitamente validada. Isso permitiria
eliminar o padding de estados e tornar todo o decode limitado ao bloco aberto.

Antes dessa mudança, a implementação atual deve permanecer como referência, pois
oferece paridade BF16 exata e ganho real de aproximadamente 2,9 vezes.
