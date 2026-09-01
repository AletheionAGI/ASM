# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Avaliar se faria sentido criar um ASM-CM-VR, isto é, aplicar Variable Rank ao ASM-CM.

## Summary

A proposta é tecnicamente coerente e combina os objetivos complementares observados na PMCS-64: memória associativa explícita do ASM-CM e gargalo lógico/estabilidade de fixed-rank do ASM-VR-S. Para preservar o contrato anti-bypass, não basta mascarar apenas o estado principal: escrita, leitura e memória fast-weight precisam ser rank-aware, idealmente com eixo de valores alinhado ao estado e projeção da matriz nas reduções de rank. Recomenda-se começar com ranks fixos e estabilizar o streaming do CM antes de qualquer controller adaptativo.

## Modified files

- [docs/report/0021_viabilidade-conceitual-asm-cm-vr_2026-09-01.md](0021_viabilidade-conceitual-asm-cm-vr_2026-09-01.md)

## Changes

- Definido o escopo conceitual ASM-CM-VR e a diferença entre uma composição core-only, que permite bypass pela memória externa, e uma versão strict rank-aware.
- Proposta uma sequência de validação com full/fixed ranks, testes anti-bypass, MQAR e streaming antes de rank adaptativo.

## Validation

- Revisão direcionada da implementação FastWeightMemory confirmou matriz causal K×V, projeção de leitura para d_state e escrita derivada de estado+token; essa interface exige tratamento explícito do eixo de valor para no-bypass.
