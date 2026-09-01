# Confirmação multiseed em 100M tokens e promoção do ASM-R

**Data:** 1 de agosto de 2026

**Arquitetura promovida:** ASM-R — Aletheion Relational State Model

**Variante de implementação:** `J_NO_DIRECTION`

**Estado da decisão:** promoção concluída para qualidade por token neste protocolo

## Resumo executivo

ASM-R completou três treinamentos independentes até 100 milhões de tokens e foi
rescoreado, em todos os marcos, sobre a mesma sequência contínua de 4.834.787
targets de validação. O CE final foi:

| Seed | CE em 100M | PPL em 100M |
|---:|---:|---:|
| 1 | 1,344849 | 3,8376 |
| 2 | 1,343751 | 3,8334 |
| 3 | 1,345016 | 3,8382 |
| Média | **1,344538** | — |
| Desvio-padrão populacional | **0,000561** | — |

A concordância entre seeds é excepcionalmente estreita. Todas as execuções
ASM-R chegaram a 100M com parâmetros finitos e curvas monotonicamente melhores
nos marcos congelados.

O resultado confirma a decisão antes provisória: **ASM-R passa a ser a
arquitetura principal da família ASM para qualidade por token no protocolo
Wikipedia byte-level de contexto 512**.

## Curva multiseed consolidada

| Tokens | CE médio | Desvio-padrão populacional |
|---:|---:|---:|
| 1M | 2,232584 | 0,002435 |
| 2M | 2,014519 | 0,006462 |
| 5M | 1,750925 | 0,000363 |
| 10M | 1,611494 | 0,000648 |
| 20M | 1,511525 | 0,001572 |
| 30M | 1,465967 | 0,000794 |
| 50M | 1,411406 | 0,001656 |
| 100M | **1,344538** | **0,000561** |

Essa curva substitui a leitura anterior baseada apenas na seed 1. O crossover
observado contra o controle seletivo ASM-S deixou de ser apenas uma hipótese de
triagem: ASM-R já havia vencido ASM-S em três seeds pareadas aos 30M e agora
repete sua própria trajetória até 100M em três seeds.

## O que exatamente foi promovido

ASM-R remove o catálogo explícito de direções e o fluxo fatorado. Seu caminho é:

```text
token
  → estado causal
  → transição contextual direta T(z,x)
  → naturalização pela métrica relacional G(z)
  → mixer causal e residual token→estado
  → memória seletiva forget/write
  → emitter
```

O movimento continua sendo um vetor de transição, mas a direção é aprendida
implicitamente por `T(z,x)`. A métrica relacional permanece no caminho e
condiciona a atualização. Portanto, o resultado não elimina toda geometria; ele
elimina a necessidade, nesta arquitetura, de fatorar explicitamente o movimento
em um catálogo de direções, gates e coeficientes.

## Situação do ASM-F

ASM-F (`J_METRIC_ORTHONORMAL_DIRECTION`) foi a alternativa direcional mais
próxima na seed 1: CE 1,346046 em 100M contra 1,344849 do ASM-R. A confirmação
multiseed, porém, revelou uma falha decisiva:

- seed 2 tornou-se não finita aproximadamente em 69,39M tokens;
- seed 3 tornou-se não finita aproximadamente em 62,18M tokens;
- ambos os checkpoints ASM-F de 100M possuem 126.080.896 parâmetros não finitos;
- na seed 3, o rescoring já devolveu CE não finito nos marcos 30M e 50M.

Robustez faz parte do desempenho arquitetural. ASM-F geração 1 não possui um
resultado multiseed válido em 100M e não pode ser apresentado como empate
estatístico com ASM-R.

A fatorização foi posteriormente protegida com Cholesky por amostra, jitter
adaptativo, fallback finito e gradientes fail-fast. Uma nova execução com essas
mudanças será chamada **ASM-F geração 2**. Ela constitui um novo experimento e
não uma recuperação retroativa dos checkpoints contaminados.

## Decisão de nomenclatura

A família permanece **ASM — Aletheion State Models**. Dentro dela:

- **ASM-R — Relational State Model:** arquitetura principal promovida;
- **ASM-S — Selective State Model:** variante eficiente por tempo;
- **ASM-F — Relational Frame State Model:** pesquisa geométrica de segunda geração;
- **ASM-X — Explicit DRM State Model:** referência histórica e experimental da teoria DRM.

O nome DRM permanece associado à teoria e à variante explícita ASM-X. Ele não é
mais imposto à arquitetura principal. O nome público descritivo do ASM-R é
**Relational State Emitter**.

## Escopo da promoção

A promoção significa:

- melhor evidência reproduzível de qualidade por token dentro da família testada;
- estabilidade confirmada em três seeds até 100M;
- escolha padrão para novos experimentos de CE, contexto e geração;
- arquitetura de referência para documentação e APIs novas.

Ela não significa:

- superioridade sobre Transformers, Mamba ou outros SSMs;
- validação em outros datasets, tokenizadores ou contextos;
- melhor CE por hora: ASM-S continua muito mais rápido;
- comprovação física ou matemática da teoria DRM;
- prontidão para produção, chat, segurança ou alinhamento.

## Próximos testes da arquitetura promovida

1. Avaliar geração qualitativa com o caminho incremental já corrigido.
2. Repetir MQAR e associative recall após as correções de runtime.
3. Medir prefill, decode e memória por comprimento de contexto.
4. Comparar ASM-R com ASM-S por tokens, tempo e energia.
5. Adicionar baselines externos corretos: Transformer, Mamba e SSM moderno.
6. Testar contexto longo e retenção além de 512 tokens.
7. Executar ASM-F geração 2 separadamente, com monitoramento de finitude.

## Proveniência

Os valores consolidados e hashes dos resumos locais estão em:

- `docs/benchmarks/asm_r_confirmation_100m_multiseed/summary.json`;
- `docs/benchmarks/asm_r_confirmation_100m_multiseed/README.md`.

Os checkpoints permanecem em `runs/` e não são versionados devido ao tamanho.
O diagnóstico numérico completo do ASM-F está no relatório 036.
