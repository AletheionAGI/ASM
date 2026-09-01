# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Esclarecer em que o ASM-R foi melhor e considerar que o ASM-VR adaptativo parece mais atraente em custo e distribuição de rank.

## Summary

Foi esclarecido que ASM-R foi melhor apenas em qualidade linguística (CE), enquanto ASM-VR adaptativo teve melhor throughput, pico de memória e contagem de parâmetros no sistema medido. A atribuição causal foi separada: quase todo o ganho físico vem do scaffold VR sem mixer/residual/memória, não da redução adaptativa de rank, pois VR full/fixed/adaptive têm custos semelhantes. A distribuição ampla de rank é estruturalmente interessante, mas apresentou baixa correlação com dificuldade e não superou fixed-16.

## Modified files

- [docs/report/0010_interpretacao-qualidade-custo-asm-r-asm-vr_2026-09-01.md](0010_interpretacao-qualidade-custo-asm-r-asm-vr_2026-09-01.md)

## Changes

- Quantificada a diferença entre qualidade, custo do scaffold e benefício específico do rank adaptativo.
- Definido o experimento necessário para isolar custo e qualidade do mecanismo de rank.

## Validation

- None recorded.
