# Protocolo decisivo de comparação MQAR das arquiteturas ASM

**Data:** 1 de agosto de 2026

## Perguntas

O experimento foi construído para responder:

1. O pré-treinamento linguístico ajuda a adaptação MQAR?
2. A memória seletiva é causalmente necessária para o resultado do ASM-R?
3. A geometria relacional ajuda ou prejudica recuperação associativa?
4. A alocação ampliada de memória do ASM-S aprende mais rapidamente?
5. Um Transformer de orçamento próximo aprende a tarefa com maior eficiência?

## Condições

| Condição | Inicialização | Geometria | Memória seletiva | Finalidade |
|---|---|---|---|---|
| `ASM_R_PRETRAINED` | checkpoint Wikipedia 100M | sim | sim | arquitetura promovida |
| `ASM_R_RANDOM` | aleatória | sim | sim | efeito do pré-treinamento |
| `ASM_R_NO_MEMORY` | pesos comuns do checkpoint 100M | sim | removida | intervenção causal na memória |
| `ASM_S_PRETRAINED` | checkpoint Wikipedia 100M | não | ampliada | geometria versus memória |
| `TRANSFORMER_RANDOM` | aleatória | não | attention | baseline externo de aproximadamente 85M |

O controle sem memória carrega todos os pesos compatíveis do ASM-R pré-treinado
e descarta somente os tensores `selective_memory.*`. Isso é mais informativo que
reiniciar toda a rede, embora a remoção também altere a contagem de parâmetros.

O Transformer é pareado aproximadamente por parâmetros, mas não possui
pré-treinamento linguístico. Seu resultado mede aprendizagem MQAR supervisionada,
não qualidade linguística comparativa.

## Protocolo pareado

- mesmos batches contínuos de treinamento para todas as condições;
- mesmos batches congelados de validação em todos os marcos;
- seed 1234;
- batch 4;
- 8 pares e 8 consultas por exemplo;
- 32 chaves e 64 valores possíveis;
- AdamW, LR `1e-4`, weight decay `0,01`;
- BF16 em CUDA;
- marcos 0, 200, 500, 1.000, 2.000, 5.000, 10.000 e 20.000.

O runner salva `partial.json` depois de cada arquitetura, além de `results.json`
e `report.md` ao final.

## Comando

```bash
./scripts/run_asm_r_mqar_architecture_comparison.sh
```

Saída padrão:

```text
runs/asm_r_mqar_architecture_comparison_20k/
├── partial.json
├── results.json
└── report.md
```

## Critérios de leitura

- pré-treinado acima do aleatório: transferência positiva da linguagem;
- sem memória abaixo do ASM-R: contribuição causal da memória seletiva;
- ASM-R acima do ASM-S: possível contribuição da métrica;
- ASM-S acima do ASM-R: capacidade melhor alocada à memória;
- Transformer acima de todos: gargalo específico dos modelos de estado;
- CE menor sem aumento de acurácia: aprendizagem da distribuição dos valores,
  não recuperação correta das associações.

Os pontos de 50%, 80% e 90% são registrados automaticamente. Nenhum resultado
isolado prova superioridade geral fora desta tarefa sintética.
