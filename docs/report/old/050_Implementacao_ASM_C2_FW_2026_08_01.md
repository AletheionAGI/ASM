# Implementação do ASM-C2-FW

## Decisão experimental

O controle isolado terminou com 100% de acurácia e CE 0,000148 para fast
weights. O oráculo também atingiu 100%, enquanto a memória de slots densa
atingiu somente 12,18% após 10 mil passos. O armazenamento associativo é
aprendível; o roteamento discreto anterior era o gargalo dominante.

## Arquitetura implementada

ASM-C2-FW preserva a transição relacional e o cache limitado do ASM-C. A
memória adicional é uma matriz de tamanho fixo, independente do prefixo. Uma
única projeção produz a chave usada na escrita do token anterior e na consulta
do token atual. A atualização usa regra delta e gates aprendidos de escrita,
leitura e retenção.

Ao contrário do microteste, a variante completa não recebe IDs de chave,
faixas de valores ou marcadores interpretados por código. Ela deve aprender
quando escrever e ler exclusivamente a partir do estado e dos embeddings
causais.

## Protocolo de promoção

1. comparar ASM-C2-FW, ASM-C e Transformer até 5 mil passos;
2. exigir pelo menos 80% no MQAR curto;
3. desligar leitura, desligar escrita e embaralhar a memória;
4. exigir queda causal mínima de cinco pontos percentuais nas ablações;
5. somente então executar streaming até 32K;
6. confirmar pelo menos duas de três seeds;
7. medir paridade BF16 e regressão de CE em linguagem;
8. manter ASM-C2-FW como candidata se qualquer gate falhar.

## Comando

```bash
./scripts/run_asm_c2_fw_suite.sh
```

Saída principal:

```text
runs/asm_c2_fw_suite/
```
