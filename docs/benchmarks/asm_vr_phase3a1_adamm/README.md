# ASM-VR Phase 3A.1 — ablação AdamM

Esta ablação troca somente o otimizador no scaffold congelado **mixer causal +
residual token-state projetados**. Foram executados fixed-32 e adaptive-32 com
seeds 17/29/43, 489 steps e 2.002.944 tokens por run. Os seis controles AdamW
são os artefatos pareados da Phase 3A.1-B.

## Tratamento

- AdamW: PyTorch, `lr=3e-4`, betas `(0.9, 0.999)`, weight decay `0.01`;
- AdamM: `lr=3e-4`, mesmos beta2/weight decay, `beta1_min=0.5`,
  `shock_ratio=1.5`, `adapt_strength=0.03`;
- fonte AdamM: commit `980d84ce96825c3d11d6bc8dd98f0c5168897643`;
- SHA-256: `79495581868147a5bed69acc3e3a85e838634c3ced0aeb9ab98b35223722c877`.

Usar `3e-4` evita confundir a regra de momentum com o default AdamM `1e-3`.

## Resultado

| Rank | AdamW test CE | AdamM test CE | Delta AdamM−AdamW |
|---|---:|---:|---:|
| Fixed-32 | 2.6092 | 2.5867 | -0.0225 |
| Adaptativo | 2.6768 | 2.6522 | -0.0246 |

AdamM melhorou test CE nas seis comparações pareadas. O ganho médio foi
`0.0225` nat no fixed-32 e
`0.0246` nat no adaptativo.
Entretanto, a interação foi apenas
`-0.0021` nat. Isso
indica um ganho geral de otimização, não um resgate específico do controller.

Com AdamM, o adaptativo calibrado ficou em rank médio `32.24`
e CE `2.6522`, contra CE `2.5867` do
fixed-32. A diferença de `+0.0655` nat
mantém o controller fora da fronteira Pareto.

## Calibração e sensibilidade

O threshold AdamM principal `0.648` foi selecionado
somente pelas distribuições de score de validation. Ao congelar o threshold
AdamW histórico `0.672`, o AdamM adaptativo obteve rank médio `31.64` e
CE médio `2.6541`. A conclusão não muda.

## Custo observado

- throughput fixed-32: `-9.3%` versus AdamW;
- throughput adaptativo: `-7.2%` versus AdamW;
- pico CUDA: `+0.81` e `+0.82 MiB`;
- estado fixed-32: `1.61` MiB AdamW contra `2.42` MiB AdamM;
- estado adaptativo: `1.64` MiB AdamW contra `2.46` MiB AdamM.

O estado do otimizador cresce exatamente 50% por causa de `beta1_prod` FP32.

Throughput é descritivo porque os controles AdamW foram executados antes, não
intercalados com os runs AdamM.

## Gates

- `adaptive_optimizer_noninferior`: **PASS**
- `adaptive_optimizer_superior`: **PASS**
- `budget`: **PASS**
- `complete`: **PASS**
- `controller_gradient`: **PASS**
- `finite`: **PASS**
- `near_adamm_fixed32`: **FAIL**
- `pareto_vs_adamm_fixed32`: **FAIL**
- `streaming`: **PASS**
- `variation`: **PASS**


AdamM foi aprovado como ablação de otimizador com melhora pareada, mas **não
aprovou o controller adaptativo**.

## Gráficos

- [Dashboard offline](index.html)
- [Qualidade versus rank](quality_vs_rank_optimizer.png)
- [Deltas pareados](paired_optimizer_deltas.png)
- [Curvas de validação](validation_ce_optimizer.png)
- [Throughput, pico CUDA e estado](optimizer_observed_cost.png)

Cada gráfico também está disponível em SVG.

## Reprodução

```bash
.venv/bin/python scripts/run_asm_vr_phase3a1_adamm.py
```
