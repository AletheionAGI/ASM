# ASM-VR Phase 3A.1-B — rank no scaffold selecionado

Esta etapa congelou o scaffold **mixer + residual**, selecionado sem usar test,
e comparou full, ranks fixos 16/32/48 e adaptativo alvo 32. Foram 5 braços × 3
seeds × 2.003M tokens.

## Resultado

| Variante | Test CE | Rank médio | Desvio de rank | Tokens/s | Pico CUDA MiB |
|---|---:|---:|---:|---:|---:|
| selected_adaptive_32 | 2.6768 | 32.21 | 21.74 | 201308 | 78.9 |
| selected_fixed_16 | 2.6706 | 16.00 | 0.00 | 202509 | 78.9 |
| selected_fixed_32 | 2.6092 | 32.00 | 0.00 | 202433 | 78.9 |
| selected_fixed_48 | 2.5897 | 48.00 | 0.00 | 203455 | 78.9 |
| selected_full | 2.5803 | 64.00 | 0.00 | 202134 | 78.9 |


O threshold global `0.672` foi calibrado somente
nas distribuições de score de validation. O adaptativo atingiu rank médio
`32.21` e desvio
`21.74`.

## Resultado científico

A etapa técnica passou orçamento, variação, gradiente, finitude e streaming.
Porém, o controller **não passou os gates de qualidade**:

- adaptativo − fixed-32: `+0.0676` nat;
- adaptativo − fronteira fixa interpolada: `+0.0678` nat;
- fixed-32 teve CE menor com rank praticamente igual;
- fixed-16 também teve CE ligeiramente menor usando metade do rank.

Portanto, não há vantagem Pareto nem vantagem adaptativa. O rank variável é
amplo, mas a alocação aprendida continua pior que máscaras fixas.

## Gates

- `adaptive_budget`: **PASS**
- `adaptive_frontier_advantage`: **FAIL**
- `adaptive_variation`: **PASS**
- `complete_matrix`: **PASS**
- `controller_gradient`: **PASS**
- `finite_runs`: **PASS**
- `pareto`: **FAIL**
- `quality_near_fixed32`: **FAIL**
- `streaming_parity`: **PASS**


Streaming FP32 máximo: `9.54e-06`.
Os caminhos continuam densos; throughput não é speedup causal por rank.

## Gráficos e dashboard

- [Dashboard offline](index.html)
- [Curvas de aprendizado](validation_ce_by_tokens.png)
- [Qualidade versus rank](quality_vs_mean_rank.png)
- [Faixa do rank adaptativo](adaptive_rank_range.png)
- [Deltas pareados](paired_adaptive_deltas.png)
- [Custo observado denso](observed_dense_cost.png)

Cada gráfico também está disponível em SVG.
