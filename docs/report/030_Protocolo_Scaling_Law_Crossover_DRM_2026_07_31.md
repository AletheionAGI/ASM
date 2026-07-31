# Protocolo de scaling law e crossover arquitetural do DRM

Data: 31 de julho de 2026

## 1. Motivação

As ablações DRM apresentaram uma inversão de ranking entre 5 e 30 milhões de
tokens.

Em 5M:

| Variante | CE médio |
|---|---:|
| J_DIRECT_CONTROL_MATCHED | 1,728162 |
| J_NO_DIRECTION | 1,751030 |

Em 30M:

| Variante | CE médio |
|---|---:|
| J_NO_DIRECTION | 1,477576 |
| J_DIRECT_CONTROL_MATCHED | 1,487258 |

O controle direto começou 0,022868 CE à frente, mas `J_NO_DIRECTION` terminou
0,009682 CE à frente. Entre os dois pontos, a redução de CE foi 0,273455 para
`J_NO_DIRECTION` e 0,240904 para o controle.

Isso indica regimes de aprendizado diferentes. O resultado em 5M mede
principalmente eficiência de otimização inicial; ele não determina sozinho a
arquitetura que escala melhor.

## 2. Por que 5M e 30M não definem uma scaling law

Uma curva empírica comum é:

$$
L(N)=L_\infty+AN^{-\alpha},
$$

onde $N$ é o número de tokens, $L_\infty$ é a perda assintótica, $A$ é a
distância de escala e $\alpha$ é o expoente de scaling.

Dois pontos não determinam de forma confiável os três parâmetros. Também não
mostram platôs, mudanças de regime ou um segundo crossover.

Cada arquitetura deve receber uma curva própria. Ajustar uma única lei sobre
arquiteturas diferentes esconderia justamente a diferença que está sendo
investigada.

## 3. Protocolo

Após a triagem atual de 30M, serão escolhidas duas ou três variantes. Cada uma
será treinada uma única vez, continuamente, até 100M tokens com seed 1.

Checkpoints serão preservados somente nos marcos:

```text
1M, 2M, 5M, 10M, 20M, 30M, 50M e 100M
```

Cada checkpoint será avaliado sobre a mesma sequência contínua de validação. A
seleção não usará `best_val_ce`, pois mínimos sobre amostras diferentes são
enviesados.

O protocolo produz duas leituras:

- **CE por tokens:** eficiência amostral;
- **CE por horas de GPU:** eficiência computacional.

## 4. Implementação

`train_drm_memmap.py` agora aceita:

```text
--checkpoint-token-milestones
```

Os checkpoints são gravados atomicamente. `checkpoint_latest.pt` e
`checkpoint_last.pt` tornam-se hard links para o último marco, evitando
duplicar arquivos de aproximadamente 1 a 1,5 GB. O runner desabilita
`checkpoint_best.pt`, pois ele não participa deste protocolo.

O rescoring gera, para cada arquitetura e marco:

- CE e perplexidade congelados;
- hash do checkpoint;
- tokens de validação;
- tempo acumulado e horas de GPU;
- ajuste separado de $L(N)$;
- crossovers observados entre pontos consecutivos.

O ajuste é exploratório. Uma única seed permite mapear curvas, mas não fornece
intervalos de confiança populacionais.

## 5. Comando recomendado

Se a triagem de 30M mantiver a variante ortonormal como melhor candidata
métrica-primeiro, execute:

```bash
VARIANTS=J_NO_DIRECTION,J_DIRECT_CONTROL_MATCHED,J_METRIC_ORTHONORMAL_DIRECTION \
OUTPUT_ROOT=runs/drm_scaling_law_100m_seed1 \
./scripts/run_drm_scaling_law_100m.sh
```

Se `J_METRIC_SUBSPACE` superar a ortonormal em 30M, substitua o terceiro nome:

```bash
VARIANTS=J_NO_DIRECTION,J_DIRECT_CONTROL_MATCHED,J_METRIC_SUBSPACE \
OUTPUT_ROOT=runs/drm_scaling_law_100m_seed1 \
./scripts/run_drm_scaling_law_100m.sh
```

Para executar somente as duas referências confirmadas:

```bash
./scripts/run_drm_scaling_law_100m.sh
```

## 6. Saídas

O resultado consolidado será:

```text
runs/drm_scaling_law_100m_seed1/scaling_law_summary.json
```

Cada diretório de variante conterá oito arquivos:

```text
checkpoint_milestone_1000000.pt
checkpoint_milestone_2000000.pt
checkpoint_milestone_5000000.pt
checkpoint_milestone_10000000.pt
checkpoint_milestone_20000000.pt
checkpoint_milestone_30000000.pt
checkpoint_milestone_50000000.pt
checkpoint_milestone_100000000.pt
```

Os aliases `checkpoint_latest.pt` e `checkpoint_last.pt` não consomem o espaço
integral novamente quando o filesystem suporta hard links.

## 7. Critérios de decisão

1. Comparar CE nos mesmos marcos, nunca em amostras diferentes.
2. Identificar todos os crossovers observados.
3. Comparar os expoentes $\alpha$ sem interpretar o ajuste como lei física.
4. Verificar se a vantagem final compensa o custo em horas de GPU.
5. Promover somente as duas melhores curvas para três seeds.
6. Repetir os marcos decisivos nas seeds 2 e 3.

Uma arquitetura pode ser escolhida por eficiência amostral, eficiência de
compute ou qualidade assintótica. Essas decisões devem ser apresentadas
separadamente.

## 8. Estimativa operacional

Com duas variantes atualmente confirmadas, a execução até 100M deve levar
aproximadamente 2,5 a 3 horas de treinamento, mais rescoring. Com uma terceira
variante geométrica de 126M parâmetros, a faixa provável é de 4,5 a 6 horas.

Oito checkpoints exigem aproximadamente 8 a 12 GB por variante. Com três
variantes, recomenda-se reservar de 30 a 40 GB, além de margem temporária para
gravação atômica.

## 9. Interpretação esperada

O crossover não é ruído a ser descartado. Ele pode indicar que mecanismos
geométricos possuem custo de aquisição inicial, seguido de maior eficiência em
escalas posteriores.

Também pode ser apenas uma mudança transitória antes de um novo platô. A curva
até 100M separará melhor essas hipóteses e impedirá que uma arquitetura seja
promovida ou descartada somente por sua velocidade nos primeiros milhões de
tokens.
