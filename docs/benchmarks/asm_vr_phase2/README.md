# ASM-VR Phase 2 — rank adaptativo sintético

A Phase 2 treina o controller causal com máscara hard no forward e estimador
straight-through no backward. O benchmark usa `Variable-Capacity Copy`, com
regimes de um e três itens e marcador causal no primeiro token de cada bloco.

## Protocolo final

- variantes: ASM-R, VR full-rank, ranks fixos 4 e 8, e VR adaptativo;
- seeds: `17`, `29`, `43`;
- 200 updates por execução, batch 64;
- estado 16-D, rank mínimo adaptativo 4 e alvo médio 8;
- máscaras prefixadas/aninhadas;
- nenhuma memória, mixer, residual ou solver auxiliar;
- execução densa: não há claim de speedup.

## Resultado

Todos os gates pré-definidos passaram:

- acurácia média ASM-R/VR-full: `0.4871`;
- acurácia média rank fixo 8: `0.4578`;
- acurácia média adaptativa: `0.4201`;
- rank médio adaptativo: `9.67`;
- desvio médio de rank: `5.67`;
- correlação rank × dificuldade: `1.0`;
- gradiente do controller presente em 100% dos updates medidos após warm-up.

O adaptativo preservou qualidade dentro do limite contra rank fixo 8, mas não o
superou e usou rank médio maior que 8. Portanto, o resultado demonstra
**controlabilidade e adaptação**, não vantagem Pareto de qualidade/capacidade.

## Desenvolvimento e transparência

Duas matrizes exploratórias com rank mínimo 2 falharam o gate de qualidade. A
primeira permitia máscaras livres; a segunda usava máscaras prefixadas. O
protocolo final elevou o piso para 4 porque a contagem de itens lógicos não é um
lower bound matemático do rank interno. Esses ensaios não foram usados como
resultado final e estão declarados no `manifest.json`.

## Reprodução

```bash
.venv/bin/python scripts/run_asm_vr_phase2.py \
  --steps 200 \
  --batch-size 64 \
  --evaluation-batches 16 \
  --seeds 17 29 43 \
  --output docs/benchmarks/asm_vr_phase2/summary.json
```

Arquivos:

- `manifest.json`: protocolo congelado, ambiente e disclosure;
- `summary.json`: métricas de todas as 15 execuções e gates de aceite.
