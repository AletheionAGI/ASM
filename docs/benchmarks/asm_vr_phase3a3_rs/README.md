# ASM-VR-RS full — Relational Selective State

Taxonomia:

- **ASM-RS:** mistura do núcleo ASM-R com o núcleo ASM-S;
- **ASM-VR-RS:** aplicação de Variable Rank sobre ASM-RS;
- braço desta fase: `vr_rs_full`, controle full-rank 64.

Nome completo:

> **ASM-VR-RS — Variable-Rank Relational Selective State Emitter**

O modelo combina métrica/naturalização relacional, mixer causal, residual
token-state e memória seletiva, com projeção entre todos os componentes. Não usa
memória endereçável nem Transition Memory.

## Resultado

| Base full | Test CE | Parâmetros | Tokens/s | Pico CUDA MiB |
|---|---:|---:|---:|---:|
| vr_r_full | 2.5803 | 223814 | 199555 | 79.9 |
| vr_rs_full | 2.5721 | 244550 | 128876 | 95.7 |
| vr_s_full | 2.5318 | 223738 | 189308 | 74.8 |


ASM-VR-RS ficou `-0.0082` nat contra R e `+0.0403` nat contra S.
Ele reproduz mecanicamente a receita relacional+seletiva já usada pelo antigo
ASM-R prático, agora com taxonomia explícita.

O RS **não é parameter-matched**: possui
`9.3%`
mais parâmetros que R/S. Mesmo assim, S teve qualidade melhor, mais throughput,
menos memória e menos parâmetros. Portanto, a composição RS foi validada como
modelo, mas não foi promovida nesta escala.

## Gates

- `complete`: **PASS**
- `finite`: **PASS**
- `rs_beats_r`: **FAIL**
- `rs_beats_s`: **FAIL**
- `streaming`: **PASS**


## Gráficos

- [Dashboard offline](index.html)
- [Curvas full-rank](validation_ce_full_bases.png)
- [CE final R/S/RS](full_base_test_ce.png)
- [Custo observado](full_base_observed_cost.png)
- [Qualidade versus parâmetros](quality_vs_parameters.png)

Cada gráfico também está disponível em SVG.

## Reprodução

```bash
.venv/bin/python scripts/run_asm_vr_phase3a3_rs.py
```
