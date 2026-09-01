# Implementação do ASM-C2-FW-LM

## Motivação

A suíte durável demonstrou que o ASM-C2-FW recupera associações com 100% de
acurácia até 4.096 tokens e 99,78% em 32.768 tokens, mantendo cache e VRAM
constantes. A arquitetura, porém, não foi promovida porque sua CE de linguagem
regrediu de 1,3420 para 1,5003 e a paridade BF16 excedeu os limites definidos.

O ASM-C2-FW-LM testa a compatibilidade entre memória associativa durável e a
competência linguística do checkpoint ASM-R de 100 milhões de tokens. Ele não é
uma arquitetura promovida; é a candidata da próxima rodada experimental.

## Implementação

### Inicialização

Cada seed começa novamente no mesmo checkpoint ASM-R de 100M. Os componentes
preexistentes são migrados integralmente e somente a memória fast-weight e seus
gates são inicializados como componentes novos.

### Treinamento misto

Por padrão, cada atualização é sorteada entre:

- 80% de batches de linguagem da Wikipedia;
- 20% de batches MQAR do currículo de distância.

A fração efetivamente observada e a contagem de batches de cada tarefa são
gravadas em `results.json`.

### Learning rates por componente

O otimizador possui dois grupos disjuntos:

| Grupo | Learning rate padrão |
|---|---:|
| Backbone linguístico | `1e-5` |
| Memória fast-weight e gates | `1e-4` |

Isso permite adaptar rapidamente a memória sem deslocar os pesos linguísticos
na mesma velocidade.

### Distilação do ASM-R

O checkpoint ASM-R permanece congelado como professor. Nos batches de
linguagem, o objetivo do estudante é:

$$
L = L_{\mathrm{CE}} + \lambda_{\mathrm{KD}}L_{\mathrm{KD}},
$$

com peso padrão `0.5` e temperatura `2.0`. A divergência é normalizada por
token, evitando que sequências maiores alterem implicitamente a escala do
objetivo.

### Caminho fast-weight FP32

Além de manter as matrizes rápida e consolidada em FP32, a variante executa em
FP32:

- projeções de chave e valor;
- gates de leitura, escrita, esquecimento e consolidação;
- predição e atualização delta-rule;
- acumulação das matrizes;
- token anterior armazenado no estado da memória.

O estado devolvido ao restante do modelo retorna ao dtype do chamador. Essa
separação mantém a interface BF16 enquanto protege a recorrência numericamente
sensível.

## Gates

A suíte executa três seeds e exige aprovação do currículo MQAR em pelo menos
duas. Somente então executa:

1. paridade BF16 entre recomposição e streaming compacto;
2. MQAR em 512, 4K e 32K;
3. cache, VRAM e throughput de streaming;
4. rescoring linguístico congelado contra o ASM-R original;
5. decisão conjunta usando os mesmos limites da suíte durável.

A promoção exige simultaneamente:

- pelo menos 80% de MQAR em todas as distâncias;
- cache limitado;
- crescimento de VRAM de no máximo 10%;
- retenção de throughput de pelo menos 80%;
- regressão de CE de linguagem de no máximo 0,05;
- divergência de argmax BF16 de no máximo 1%;
- erro absoluto médio BF16 de no máximo 0,02.

## Comando

```bash
./scripts/run_asm_c2_fw_lm_suite.sh
```

Saídas principais:

```text
runs/asm_c2_fw_lm_suite/report.md
runs/asm_c2_fw_lm_suite/decision.json
runs/asm_c2_fw_lm_suite/bf16_parity.json
runs/asm_c2_fw_lm_suite/long_32k/results.json
runs/asm_c2_fw_lm_suite/language_regression/
```

Os hiperparâmetros podem ser alterados sem editar o runner:

```bash
LANGUAGE_PROBABILITY=0.8 \
BACKBONE_LR=1e-5 \
MEMORY_LR=1e-4 \
DISTILLATION_WEIGHT=0.5 \
./scripts/run_asm_c2_fw_lm_suite.sh
```

## Interpretação

Se todos os gates passarem, a evidência sustentará uma variante que combina
memória associativa durável, estado limitado e preservação linguística dentro
dos limites do protocolo. Se MQAR cair, a proporção 80/20 será insuficiente para
ensinar a memória. Se apenas a CE falhar, o próximo teste deverá congelar uma
parte maior do backbone ou elevar a pressão de distilação. Se apenas a paridade
falhar, o problema restante será o desacordo entre execução em blocos e decode
incremental, não a precisão interna da memória fast-weight.
