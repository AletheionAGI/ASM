# Protocolo diagnóstico MQAR do ASM-C

## Objetivo

A validação streaming confirmou cache, VRAM e throughput limitados até 32K,
mas o controle MQAR de 40 tokens alcançou somente 32,25%. Antes de modificar
novamente a arquitetura ou investir em redução de CE, este protocolo separa
três hipóteses:

1. **subtreinamento:** o ASM-C atual precisa de mais passos de adaptação;
2. **capacidade insuficiente:** a memória forget/write é estreita demais;
3. **limitação arquitetural:** mesmo com mais treino ou capacidade, a
   recorrência não aprende recuperação associativa de forma eficiente.

## Variantes

| Variante | Inicialização | Memória | Pergunta |
|---|---|---|---|
| `ASM_C_PRETRAINED` | checkpoint ASM-R de 100M | largura 1x | mais passos resolvem? |
| `ASM_C_MEMORY_2X` | pesos ASM-R compartilhados | largura 2x, reinicializada | mais capacidade resolve? |
| `ASM_S_PRETRAINED` | checkpoint ASM-S de 100M | capacidade redistribuída | geometria ajuda ou atrapalha? |
| `TRANSFORMER_PRETRAINED` | checkpoint pareado de 100M | attention | qual é o controle competitivo? |

ASM-C e ASM-R compartilham o mesmo forward completo durante a adaptação curta;
o modo compacto altera o estado incremental, não os logits do treino em lote.
O rótulo ASM-C é mantido porque esta é a arquitetura cuja capacidade de
streaming e retenção estamos qualificando.

Na variante 2x, somente o módulo de memória seletiva é reinicializado porque a
mudança de largura torna seus tensores incompatíveis. Todos os demais pesos
compatíveis são carregados do checkpoint de 100M. Isso deve ser considerado ao
interpretar velocidade inicial de aprendizado.

## Protocolo pareado

- milestones: 5.000, 10.000 e 20.000 passos;
- batch: 4 exemplos;
- oito pares e oito queries por exemplo;
- 128 batches fixos de avaliação;
- 4.096 targets de validação por milestone;
- precisão BF16;
- mesmos batches de treino e validação para todas as variantes;
- gate: acurácia mínima de 80% no controle curto.

## Comando

```bash
./scripts/run_asm_c_mqar_diagnostic.sh
```

Saída padrão:

```text
runs/asm_c_mqar_diagnostic_20k/results.json
runs/asm_c_mqar_diagnostic_20k/report.md
```

## Regras de decisão

### ASM-C atual supera 80%

O problema principal era subtreinamento da adaptação MQAR. Repetir então o
teste de retenção em 512, 1K, 2K, 4K, 8K, 16K e 32K usando o primeiro
checkpoint que ultrapassar o gate.

### Somente ASM-C 2x supera 80%

Existe evidência de gargalo de capacidade da memória. O próximo estudo deve
variar larguras 1x, 2x e 4x, comparando acurácia por parâmetro e por tempo.

### ASM-S supera ASM-C

A geometria relacional não está ajudando esta tarefa de recuperação, ou a
capacidade é melhor empregada diretamente na memória. Isso não resolve sozinho
o mérito da geometria para linguagem, mas orienta uma memória streaming
especializada.

### Transformer aprende muito mais rápido

A attention continua superior para associação explícita nesta escala. O ASM-C
precisará de memória endereçável, slots ou mecanismo equivalente antes de uma
nova avaliação longa.

### Nenhum ASM supera 80%

Não repetir ainda o MQAR longo. Implementar uma segunda geração, provisoriamente
ASM-C2, combinando estado recorrente compacto com um pequeno banco de memória
endereçável e leitura/escrita seletivas.

## Relação com CE

CE de linguagem permanece uma métrica obrigatória de regressão, mas não é o
alvo desta rodada. A próxima mudança só deve ser promovida se melhorar memória
sem degradar de forma material o CE, throughput ou limite de memória já
confirmados.

## Resultado obtido

Em 20.000 passos, Transformer, ASM-C, ASM-C memória 2x e ASM-S obtiveram,
respectivamente, 100,00%, 36,21%, 35,96% e 35,42% de acurácia. Dobrar a largura
da memória reduziu o CE do ASM-C, mas não melhorou sua acurácia final. Nenhuma
variante ASM atingiu o gate de 80% ou sequer 50%.

O resultado acionou a última regra de decisão deste protocolo. O próximo
desenvolvimento será o ASM-C2 com banco de memória endereçável de tamanho fixo,
detalhado no
[report 047](047_Plano_Implementacao_ASM_C2_Memoria_Enderecavel_2026_08_01.md).
