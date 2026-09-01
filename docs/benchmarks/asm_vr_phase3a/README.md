# ASM-VR Phase 3A — linguagem em pequena escala

Este gate compara seis variantes em um corpus Wikipedia byte-level, com split
90/5/5 por documento e SHA-256, três seeds e aproximadamente 2M tokens por run.
A matriz completa contém 18 execuções.

## Resultado principal

Todos os nove gates operacionais passaram após calibração do threshold hard em
validação (`0.8`, sem consultar rótulos de test):

| Variante | Test CE médio | Rank médio | Tokens/s | Pico CUDA MiB |
|---|---:|---:|---:|---:|
| ASM-R referência | 2.5721 | 64.00 | 150564 | 85.7 |
| VR full | 3.1704 | 64.00 | 273409 | 69.3 |
| VR fixo 16 | 3.1811 | 16.00 | 272687 | 69.3 |
| VR fixo 32 | 3.1753 | 32.00 | 271293 | 69.3 |
| VR fixo 48 | 3.1755 | 48.00 | 271799 | 69.3 |
| VR adaptativo | 3.1869 | 30.63 | 274078 | 72.2 |

O adaptativo ficou apenas `0.0116` nat pior que o rank fixo 32 em média e
respeitou o orçamento hard nas três seeds. A paridade streaming FP32 ficou entre
`1.9e-06` e `2.4e-06` nos runs adaptativos.

## Interpretação científica

O resultado **não demonstra vantagem Pareto**. O rank fixo 16 obteve CE menor
que o adaptativo usando menos rank. A correlação descritiva entre rank e CE de
bloco também foi baixa (`0.059`).
Assim, a Phase 3A valida treino, controle de orçamento e inferência em linguagem,
mas não justifica promover o controller atual como superior.

ASM-R tem CE muito melhor porque a receita prática mantém mixer, residual e
memória seletiva, enquanto o scaffold VR os proíbe para preservar o teste de
não-bypass. `vr_full` é o controle justo para medir rank dentro do scaffold.
A diferença de throughput ASM-R×VR mede essa diferença arquitetural, não ganho
de rank esparso. Todos os caminhos VR continuam densos.

## Gráficos

- [Curvas de CE por tokens](validation_ce_by_tokens.png)
- [CE final por variante e seed](final_test_ce_by_variant.png)
- [Qualidade versus rank médio](quality_vs_mean_rank.png)
- [Distribuição de rank](rank_distribution.png)
- [Throughput e memória observados](observed_cost.png)
- [Deltas pareados por seed](paired_seed_deltas.png)
- [Dashboard HTML offline](index.html)

Cada gráfico também está disponível em SVG para zoom e documentação.

## Transparência de desenvolvimento

A primeira matriz falhou orçamento e mediu streaming sob BF16. Uma segunda
matriz adaptativa ainda falhou o orçamento hard. O threshold final `0.8` foi
calibrado usando apenas distribuições de score da validação e então aplicado aos
checkpoints congelados para a avaliação final. Os resultados intermediários
foram preservados em `development_*.json`.

## Reprodução

```bash
.venv/bin/python scripts/run_asm_vr_phase3a.py \
  --steps 489 --batch-size 16 --sequence-length 256 \
  --evaluation-batches 16 --seeds 17 29 43 \
  --output docs/benchmarks/asm_vr_phase3a \
  --run-root runs/asm_vr_phase3a

.venv/bin/python scripts/calibrate_asm_vr_phase3a.py --threshold 0.8
```

Este é um gate de pequena escala. Não é confirmação de scaling nem evidência de
speedup por rank.
