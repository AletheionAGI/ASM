# Revalidação pós-FP32 e promoção do ASM-CM

## Objetivo

Revalidar, sem retreinamento, as três linhagens congeladas do ASM-C2-FW-LM após a correção do núcleo recorrente FP32. A publicação definitiva deve usar apenas CE, throughput, VRAM, cache e paridade medidos pelo caminho numérico corrigido.

## Nome público

O identificador técnico permanece **ASM-C2-FW-LM**, preservando a linhagem experimental. O nome público proposto é **ASM-CM — Aletheion Compact Memory Model**.

O nome curto descreve a propriedade promovida — memória associativa compacta e durável — sem expor no nome público toda a sequência de versões internas. Ele também evita confusão com ASM-C, ASM-C2, ASM-F e ASM-M.

## Protocolo congelado

- checkpoints finais já treinados das seeds 1, 2 e 3;
- nenhuma atualização de pesos;
- mesma validação Wikipedia byte-level contínua;
- rescoring completo de aproximadamente 4,83 milhões de targets;
- streaming real em 512, 4K e 32K tokens;
- throughput segmentado, pico de VRAM e tamanho tensorial do cache;
- paridade BF16 contra recomposição completa;
- resultado anterior de MQAR preservado como evidência de memória, sem nova adaptação nesta medição de desempenho.

## Critério de promoção final

A promoção exige que a confirmação independente anterior continue aprovada e que todas as seeds satisfaçam os gates pós-FP32 de CE, cache, VRAM, throughput e paridade. O Transformer continua sendo o controle superior de CE; a promoção do ASM-CM é pela combinação de memória associativa durável e estado limitado, não por superioridade geral em linguagem.

## Sequência operacional

1. Executar `scripts/run_asm_cm_post_fp32_validation.sh` na RTX 4090.
2. Revisar `runs/asm_cm_post_fp32_validation/summary.json` e `report.md`.
3. Se `promote` for verdadeiro, consolidar os artefatos em `docs/benchmarks`.
4. Promover ASM-CM em README, arquitetura e família de modelos, mantendo ASM-C2-FW-LM como identificador técnico.
5. Atualizar o ASM Website com os números pós-FP32 e as limitações explícitas.
6. Executar testes de ambos os repositórios, commitar e enviar as alterações.

## Comando

```bash
./scripts/run_asm_cm_post_fp32_validation.sh
```
