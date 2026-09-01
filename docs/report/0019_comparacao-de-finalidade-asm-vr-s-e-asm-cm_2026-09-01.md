# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Avaliar se ASM-VR-S é melhor que ASM-CM e se os modelos têm finalidades diferentes.

## Summary

Foi esclarecido que os modelos têm objetivos distintos e que ainda não há evidência matched para declarar superioridade geral. ASM-VR-S investiga capacidade efetiva/rank projetado e venceu R/S/RS no protocolo pequeno; ASM-CM foi promovido por memória associativa durável e streaming com estado limitado até 32K. A comparação exige parâmetros, tokens, dados e tarefas pareados.

## Modified files

- [docs/report/0019_comparacao-de-finalidade-asm-vr-s-e-asm-cm_2026-09-01.md](0019_comparacao-de-finalidade-asm-vr-s-e-asm-cm_2026-09-01.md)

## Changes

- Separadas as finalidades científicas e operacionais de ASM-VR-S e ASM-CM.
- Definidos os eixos necessários para uma comparação causal futura.

## Validation

- Evidência ASM-CM verificada: CE 1.328496 ±0.000687 em protocolo próprio, cache retido 143360 bytes e throughput estável de 4K a 32K.
- Evidência ASM-VR-S verificada: CE 2.5318 no protocolo 2M small-scale e vitória pareada sobre VR-R; protocolos não são diretamente comparáveis.
