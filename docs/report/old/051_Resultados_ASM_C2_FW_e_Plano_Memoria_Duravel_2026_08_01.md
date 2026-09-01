# Resultados ASM-C2-FW e implementação de memória durável

## Resultado da primeira suíte completa

ASM-C2-FW resolveu integralmente o MQAR curto:

| Modelo | Acurácia em 5 mil passos | CE |
|---|---:|---:|
| ASM-C | 33,64% | 2,004768 |
| ASM-C2-FW | 100,00% | 0,000009 |
| Transformer pareado | 99,78% | 0,021904 |

ASM-C2-FW alcançou 100% já aos 500 passos, quando o Transformer estava em
68,68%. As três seeds de confirmação terminaram em 100%. Desligar leitura ou
escrita reduziu a acurácia para 33,64%; embaralhar a memória reduziu para
2,93%. Portanto, o ganho é causalmente dependente da memória fast-weight.

## Falha de retenção

O controle de 40 tokens atingiu 100%, mas a recuperação colapsou para o nível
do acaso a partir de 512 tokens:

| Distância total | Acurácia |
|---:|---:|
| 40 | 100,00% |
| 512 | 1,93% |
| 1.024 | 1,95% |
| 4.096 | 1,54% |
| 32.768 | 1,76% |

O acaso é 1,5625%. A memória é recuperável, mas não durável: escritas em tokens
distratores e retenção multiplicativa degradam as associações.

## Resultados de engenharia

O streaming computacional permaneceu limitado e estável até 32K:

- cache: 77.824 bytes em todas as distâncias;
- pico CUDA: aproximadamente 391 MiB;
- throughput: 84,59 tokens/s em 4K e 84,00 tokens/s em 32K;
- retenção de throughput: aproximadamente 99,3%.

Entretanto, o ajuste exclusivo em MQAR provocou regressão de CE de linguagem
de 1,3420 para 2,7909. A paridade BF16 passou em argmax (0,39% de divergência),
mas falhou no erro absoluto médio (0,0454 contra limite de 0,02).

## Correções implementadas

### 1. Gates de retenção longa reais

O critério ambíguo `long_mqar_control` foi renomeado. O controle curto agora
somente autoriza interpretar a avaliação longa. Gates independentes exigem 80%
em 512, 4.096 e 32.768 tokens.

### 2. Curva do ponto de colapso

O currículo e as avaliações incluem 40, 80, 160, 320, 512, 1.024 e 4.096
tokens. A avaliação final acrescenta 32K.

### 3. Memória rápida e consolidada

O estado passa a conter duas matrizes de capacidade fixa:

$$
M_t^{\mathrm{fast}}
$$

para associações recentes, e

$$
M_t^{\mathrm{slow}}
$$

para consolidação seletiva. A leitura combina as duas escalas.

### 4. Escrita seletiva

Um gate hard com estimador straight-through impede que toda entrada produza
necessariamente uma escrita. O currículo com distratores ensina o controlador
a preservar a memória quando o token não contém informação consolidável.

### 5. Consolidação

Um gate causal separado transfere atualizações selecionadas para a matriz
lenta. Isso implementa memória durável interna; armazenamento episódico externo
permanece a extensão recomendada para mundos com histórico ilimitado.

### 6. Currículo de distância

O runner padrão usa:

```text
40:1000,80:500,160:500,320:300,512:200,1024:100,4096:25
```

Cada par indica `comprimento:passos`.

### 7. Replay de linguagem

Por padrão, 20% das atualizações usam janelas reais de 128 tokens da Wikipedia.
O objetivo é reduzir esquecimento catastrófico durante a especialização MQAR.

### 8. Estado FP32

As matrizes fast-weight rápida e consolidada permanecem em FP32 mesmo sob
autocast BF16. Isso testa se a acumulação numérica era responsável por parte da
divergência incremental.

## Protocolo decisivo

1. executar o currículo em três seeds;
2. exigir 80% em todas as distâncias até 4K em pelo menos duas seeds;
3. avaliar 512, 4K e 32K com gates explícitos;
4. medir cache, VRAM e throughput;
5. medir paridade BF16;
6. limitar regressão de CE de linguagem a 0,05;
7. promover somente se todos os critérios passarem.

## Comando

```bash
./scripts/run_asm_c2_fw_durable_suite.sh
```

Saída:

```text
runs/asm_c2_fw_durable_suite/report.md
runs/asm_c2_fw_durable_suite/decision.json
```
