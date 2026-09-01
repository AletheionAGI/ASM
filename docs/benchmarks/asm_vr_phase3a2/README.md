# ASM-VR Phase 3A.2 — ASM-VR-R versus ASM-VR-S

Esta fase compara duas bases parameter-matched sob as mesmas cinco políticas de
rank, três seeds e 2.002.944 tokens por run. O test permaneceu selado até
checkpoint, thresholds e política comum serem congelados em validation.

- **ASM-VR-R:** núcleo relacional + mixer + residual, sem memória seletiva;
- **ASM-VR-S:** núcleo direto + mixer + residual + memória seletiva projetada,
  sem métrica/naturalização.

## Pareamento

| Base | Parâmetros | Memória seletiva hidden |
|---|---:|---:|
| VR-R | 223814 | desligada |
| VR-S | 223738 | 308 |

A diferença é somente `0.034%` (`−76`
parâmetros), o inteiro mais próximo permitido pelo degrau de 321 parâmetros da
memória seletiva.

## Matriz AdamW

| Base/rank | Test CE | Rank médio | Tokens/s | Pico CUDA MiB |
|---|---:|---:|---:|---:|
| vr_r_adaptive_32 | 2.6779 | 32.16 | 199897 | 79.9 |
| vr_r_fixed_16 | 2.6732 | 16.00 | 199591 | 79.9 |
| vr_r_fixed_32 | 2.6108 | 32.00 | 200400 | 79.9 |
| vr_r_fixed_48 | 2.5905 | 48.00 | 200284 | 79.9 |
| vr_r_full | 2.5803 | 64.00 | 199555 | 79.9 |
| vr_s_adaptive_32 | 2.6234 | 32.26 | 188781 | 74.8 |
| vr_s_fixed_16 | 2.6176 | 16.00 | 187280 | 74.8 |
| vr_s_fixed_32 | 2.5605 | 32.00 | 187025 | 74.8 |
| vr_s_fixed_48 | 2.5424 | 48.00 | 189771 | 74.8 |
| vr_s_full | 2.5318 | 64.00 | 189308 | 74.8 |


ASM-VR-S melhorou test CE em **todas as 15 comparações pareadas**:

- full: `-0.0485` nat;
- fixed-32: `-0.0503` nat;
- média por rank entre aproximadamente `−0.040` e `−0.061` nat.

O S full teve throughput `-5.1%` e pico de memória
`6.4%` menor que R full. Logo, foi promovido pelo caminho de
**superioridade de qualidade**, não pelo gate histórico de `1.25×` throughput.

Um resultado prático importante: S fixed-32 obteve CE `2.5605`
contra `2.5803` do R full. Ele melhora qualidade com metade
do rank lógico, embora o caminho físico ainda seja denso.

## Controller

Os thresholds validation-only foram `0.674` para R
e `0.689` para S. Ambos atingiram rank médio próximo
de 32, variaram e receberam gradiente. Porém, os adaptativos continuaram fora
da fronteira fixa:

- R adaptive − fronteira: `+0.0673` nat;
- S adaptive − fronteira: `+0.0631` nat.

Assim, **a base S foi aprovada; o controller adaptativo continua reprovado**.
A política comum congelada por validation foi full-rank.

## Integridade

- 30/30 runs completas e finitas;
- projeção hard de estado e `local_delta` antes do mixer;
- memória seletiva recebe e devolve somente estado projetado;
- streaming FP32 revalidado com bloco aberto causal de forma fixa;
- sem memória endereçável ou Transition Memory;
- execução continua densa, sem claim de speedup por rank.

## Gráficos

- [Dashboard offline](index.html)
- [Curvas por base e rank](validation_ce_by_base_rank.png)
- [Fronteiras de qualidade × rank](quality_vs_rank_frontiers.png)
- [Deltas pareados S−R](paired_s_minus_r_deltas.png)
- [Custo observado](observed_dense_cost.png)
- [Faixas adaptativas](adaptive_rank_ranges.png)
- [Heatmap de test CE](test_ce_heatmap.png)
- [Qualidade × throughput](quality_vs_observed_throughput.png)

Cada gráfico também está disponível em SVG.

## Reprodução

```bash
.venv/bin/python scripts/run_asm_vr_phase3a2.py
.venv/bin/python scripts/finalize_asm_vr_phase3a2.py
```

## Continuação

- [Confirmação AdamM em seeds novas](../asm_vr_phase3a2_adamm_confirm/README.md)
- [ASM-VR-RS full: comparação R/S/RS](../asm_vr_phase3a3_rs/README.md)
