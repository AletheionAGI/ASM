# ASM-VR Phase 3A.1 — scaffold projetado

A Phase 3A.1 reintegrou componentes úteis do ASM-R com projeção hard em cada
aresta, sem Transition Memory e sem acesso ao complemento descartado.

- [Relatório e dashboard 3A.1-A](stage_a/README.md)
- [Relatório e dashboard 3A.1-B](stage_b/README.md)

## Decisão

A etapa A selecionou **mixer causal + residual token-state**. Ela recuperou quase
toda a qualidade do ASM-R prático com custo observado menor. A etapa B mostrou
que o controller adaptativo atual ainda é dominado por ranks fixos. Assim, o
scaffold projetado é uma base melhor, mas o controller não deve ser promovido à
Fase 3B sem uma nova hipótese de alocação.

Abrir dashboards:

```bash
xdg-open docs/benchmarks/asm_vr_phase3a1/stage_a/index.html
xdg-open docs/benchmarks/asm_vr_phase3a1/stage_b/index.html
```

Reprodução:

```bash
.venv/bin/python scripts/run_asm_vr_phase3a1.py
.venv/bin/python scripts/finalize_asm_vr_phase3a1.py
```
