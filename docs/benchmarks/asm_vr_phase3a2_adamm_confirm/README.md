# ASM-VR-S — confirmação AdamM em seeds novas

Após a seleção AdamW, a política comum full-rank e as bases R/S foram congeladas.
A confirmação usa AdamM com seeds novas `71`, `89`, `107`, `lr=3e-4` e
2.002.944 tokens por run. Ela confirma o efeito da base; não estima uma interação
causal AdamM×base porque não há AdamW contemporâneo nessas seeds.

## Resultado

| Seed | VR-R full | VR-S full | S−R |
|---|---:|---:|---:|
| 71 | 2.5699 | 2.5227 | -0.0472 |
| 89 | 2.5621 | 2.5200 | -0.0421 |
| 107 | 2.5443 | 2.4982 | -0.0461 |


Delta médio: **`-0.0451` nat** favorável ao ASM-VR-S,
com a mesma direção em 3/3 seeds novas. Combinando seleção e confirmação, a base
S venceu R em 6/6 seeds full-rank e em todas as políticas da matriz inicial.

## Gates

- `complete`: **PASS**
- `finite`: **PASS**
- `s_quality_superior`: **PASS**
- `streaming`: **PASS**


AdamM está pinado no commit `980d84ce96825c3d11d6bc8dd98f0c5168897643` e SHA-256
`79495581868147a5bed69acc3e3a85e838634c3ced0aeb9ab98b35223722c877`.

## Gráficos

- [Dashboard offline](index.html)
- [Curvas em seeds novas](validation_ce_adamm_new_seeds.png)
- [Deltas pareados S−R](paired_s_minus_r_adamm.png)
- [Contexto AdamW/AdamM](quality_optimizer_context.png)
- [Custo da confirmação](adamm_confirmation_cost.png)

Cada gráfico também está disponível em SVG.

## Reprodução

```bash
.venv/bin/python scripts/run_asm_vr_phase3a2_adamm_confirm.py
```
