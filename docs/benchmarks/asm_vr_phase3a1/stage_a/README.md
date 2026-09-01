# ASM-VR Phase 3A.1-A — seleção do scaffold projetado

Esta etapa executou o fatorial completo `2³`: mixer causal (M), residual
token-state (R) e memória seletiva (S), sempre em full-rank 64 e com projeção
hard após cada componente. Foram 8 braços × 3 seeds × 2.003M tokens.

## Resultado

| Scaffold | Validation CE | Test CE descritivo | Tokens/s | Pico CUDA MiB |
|---|---:|---:|---:|---:|
| all_projected | 2.5603 | 2.5721 | 137959 | 94.7 |
| mixer | 2.5901 | 2.6045 | 205286 | 77.3 |
| mixer_residual | 2.5676 | 2.5803 | 200525 | 78.9 |
| mixer_selective | 2.5803 | 2.5938 | 141791 | 93.1 |
| residual | 2.7449 | 2.7591 | 246722 | 72.9 |
| residual_selective | 2.6686 | 2.6819 | 157808 | 88.8 |
| selective | 2.8319 | 2.8502 | 162248 | 87.2 |
| strict | 3.1529 | 3.1704 | 255212 | 71.3 |


A regra congelada selecionou **mixer + residual**: qualquer braço até `0.02`
nat do melhor usa desempate por simplicidade. `all_projected` foi o menor CE de
validação (`2.5603`), mas mixer+residual ficou apenas
`0.0073` nat atrás sem memória
seletiva.

Mixer+residual recuperou `0.5853`
nat frente ao scaffold estrito e ganhou nas três seeds. Contra ASM-R histórico,
teve apenas `+0.0082` nat de test CE, aproximadamente
`33.2%` mais throughput observado,
`8.0%` menos pico de memória e
`6.9%` menos parâmetros.

## Efeitos fatoriais em validation CE

| Termo | Efeito médio; negativo melhora |
|---|---:|
| mixer | -0.2750 |
| mixer:residual | +0.1322 |
| mixer:residual:selective | -0.0605 |
| mixer:selective | +0.0951 |
| residual | -0.1534 |
| residual:selective | +0.0618 |
| selective | -0.1036 |


Os efeitos principais de M, R e S foram favoráveis, mas as interações positivas
mostram retornos redundantes. Isso explica por que S não foi necessário no
scaffold selecionado.

## Gates

- matriz completa: PASS;
- runs finitas: PASS;
- streaming FP32 `≤1e-4`: PASS; máximo `2.38e-05`;
- recuperação de qualidade `≥0.05 nat` em pelo menos 2/3 seeds: PASS.

## Gráficos e dashboard

- [Dashboard offline](index.html)
- [Curvas de aprendizado](validation_ce_by_tokens.png)
- [CE por scaffold](scaffold_validation_ce.png)
- [Efeitos fatoriais](factorial_effects.png)
- [Custo observado denso](observed_dense_cost.png)

Cada gráfico também está disponível em SVG. O test foi calculado para relato,
mas não participou da seleção, que usou somente validation.
