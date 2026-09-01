# AdamM e ASM-VR

## Decisão

AdamM pode ser útil ao ASM-VR como **otimizador opcional e ablação de treino**,
mas não fornece nenhum componente semântico da arquitetura de rank variável.
Ele não deve entrar no estado causal, no controlador de rank, no transporte ou
no teste de ausência de bypass.

## Onde pode ajudar

- comparar estabilidade quando gates hard usam um estimador straight-through;
- reagir a mudanças locais do gradiente por momentum adaptativo;
- reutilizar o protocolo de comparação pareada, múltiplas seeds, tempo até uma
  loss-alvo e medição separada de memória do otimizador;
- inspirar, sem copiar diretamente, uma feature causal de "surpresa" para
  estudos futuros do controlador de rank.

## Onde não ajuda na Fase 1

A Fase 1 é um teste de semântica de inferência: depois do colapso, informação
descartada não pode reaparecer sob inputs futuros. AdamM só atua na atualização
de parâmetros durante treino. Ele não implementa frame, máscara hard, projetor,
transporte, cache causal ou estado efetivo.

Por isso, o teste decisivo da Fase 1 deve rodar sem passo do otimizador. AdamM
só deve entrar depois, em uma matriz controlada contra AdamW, quando houver
controller treinável.

## Riscos de uma ablação futura

- `beta1_prod` adiciona um tensor FP32 por parâmetro em relação ao AdamW;
- a implementação atual não é fused/foreach e pode piorar throughput;
- o benchmark do repositório AdamM depende de um módulo `adamv` ausente neste
  workspace e contém um caminho antigo no README;
- momentum adaptativo não é rank adaptativo e não deve ser apresentado assim.

## Matriz mínima recomendada

Compare AdamW e AdamM com inicialização, batches, tokens, scheduler e seeds
idênticos. Registre CE de validação, estabilidade dos gates, distribuição de
rank, tempo até a loss-alvo, tokens/s e pico de memória. O tempo do optimizer
deve ser medido separadamente do forward ASM-VR.


## Resultado empírico na Fase 3A.1

A ablação controlada foi executada no scaffold mixer+residual projetado, com
`lr=3e-4`, 489 steps e seeds 17/29/43. AdamM melhorou test CE em todas as seis
comparações: `-0.0225` nat no fixed-32
e `-0.0246` nat no adaptativo. A
interação de apenas `-0.0021` nat mostra que o
ganho foi geral, não específico do controller. O adaptativo AdamM permaneceu
`+0.0655` nat pior que fixed-32 AdamM.
AdamM é uma ablação de otimizador promissora, com custo de estado 50% maior, mas
não altera a reprovação do controller. Artefatos:
[`benchmarks/asm_vr_phase3a1_adamm/`](benchmarks/asm_vr_phase3a1_adamm/).
