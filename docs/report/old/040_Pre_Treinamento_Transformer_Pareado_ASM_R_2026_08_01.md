# Pré-treinamento do Transformer pareado ao ASM-R

## Objetivo

Produzir um controle Transformer pré-treinado adequado para a comparação MQAR.
Os resultados históricos de GPT-2 não servem para esse papel porque não
preservam um checkpoint compatível e pertencem a protocolos anteriores.

## Pareamento

| Item | ASM-R | Transformer pareado |
|---|---:|---:|
| Parâmetros | 83.206.400 | 83.001.240 |
| Diferença | — | -0,247% |
| Tokens de treino | 100.000.000 | 100.000.000 |
| Vocabulário | 256 bytes | 256 bytes |
| Contexto | 512 | 512 |
| Batch por GPU | 2 | 2 |
| Acumulação | 8 | 8 |
| Tokens por atualização | 8.192 | 8.192 |
| Seed | 1 | 1 |
| Precisão | BF16 | BF16 |
| Learning rate | 3e-4 | 3e-4 |
| Weight decay | 0,01 | 0,01 |

O Transformer possui 12 camadas, largura 756 e 12 cabeças. A diferença de
205.160 parâmetros é pequena o bastante para um controle parameter-near, mas
deve continuar explícita nos resultados.

O treinador calcula a cross-entropy diretamente entre os logits e os targets
`y` já deslocados pelo dataset. Ele não utiliza o deslocamento interno de labels
do Hugging Face, evitando o erro histórico de deslocamento duplo.

## Execução

```bash
./scripts/run_transformer_asm_r_matched_100m.sh
```

O runner salva checkpoints em 1M, 2M, 5M, 10M, 20M, 30M, 50M e 100M tokens,
além de executar rescoring congelado sobre toda a validação ao final.

Checkpoint final esperado:

```text
runs/transformer_asm_r_matched_100m_seed1/checkpoint_last.pt
```

Depois do pré-treinamento, a comparação MQAR completa passa a incluir tanto o
Transformer pré-treinado quanto sua contraparte aleatória:

```bash
./scripts/run_asm_r_mqar_architecture_comparison.sh
```

## Limites de interpretação

- O pareamento controla dados, tokens, contexto, seed e hiperparâmetros
  principais; as arquiteturas continuam tendo custos computacionais distintos.
- A proximidade de parâmetros não implica igualdade de FLOPs por token.
- O resultado em MQAR mede adaptação a associative recall, não qualidade geral
  de linguagem.
- A comparação entre Transformer pré-treinado e aleatório mede o efeito do
  pré-treinamento dentro da arquitetura Transformer.
