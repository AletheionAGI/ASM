# Protocolo de streaming longo ASM-R até 32K

## Hipótese

O ASM-R só oferece uma vantagem estrutural relevante sobre attention se o custo
incremental e a memória de inferência crescerem mais lentamente com o histórico,
preservando simultaneamente informação associativa distante.

## Experimento

A suíte percorre uma única sequência por `decode_step` e registra em 512, 1K,
2K, 4K, 8K, 16K e 32K:

- tokens por segundo no segmento;
- comprimento lógico do estado;
- bytes ocupados pelos tensores do cache;
- memória CUDA corrente e pico;
- quantidade de tokens no bloco local aberto.

Em seguida, o checkpoint é recarregado, adaptado por 5.000 passos em MQAR curto
e avaliado nas mesmas extensões com filler entre escrita e consulta. Isso mede
retenção associativa à distância, não qualidade geral de linguagem.

## Comando

```bash
./scripts/run_asm_r_long_streaming_suite.sh
```

Saídas incrementais:

```text
runs/asm_r_long_streaming_32k/
├── partial.json
└── results.json
```

`partial.json` é atualizado atomicamente em cada milestone. Assim, resultados
menores sobrevivem mesmo que 16K ou 32K encontrem OOM ou outra limitação.

## Critérios

- Cache constante: `cache_tensor_bytes` deve estabilizar depois do bloco local.
- Decode constante: `segment_tokens_per_sec` não deve cair sistematicamente.
- Retenção: MQAR deve manter acurácia com aumento da distância.
- Extrapolação operacional: 32K deve terminar sem OOM ou estado inválido.

O código atual ainda retém os IDs completos do prefixo e o emitter pode criar
temporários proporcionais ao comprimento para preservar paridade BF16. Portanto,
é provável que o teste rejeite memória estritamente constante. Esse resultado
servirá como especificação quantitativa para ASM-Streaming/ASM-C.
