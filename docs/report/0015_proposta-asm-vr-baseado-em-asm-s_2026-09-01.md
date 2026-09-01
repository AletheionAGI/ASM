# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Esclarecer se o ASM-VR usa memória do ASM-R e avaliar usar ASM-S como base em vez de ASM-R.

## Summary

Foi esclarecido que o scaffold ASM-VR selecionado usa mixer+residual e não usa memória seletiva nem endereçável; ASM-R prático no benchmark usava memória seletiva opcional. Usar ASM-S como base é uma hipótese forte de eficiência e deve ser uma Fase 3A.2 antes da 3B, produzindo ASM-VR-S com memória seletiva projetada, sem estado paralelo e com comparação full/fixed/adaptive no mesmo core.

## Modified files

- [docs/report/0015_proposta-asm-vr-baseado-em-asm-s_2026-09-01.md](0015_proposta-asm-vr-baseado-em-asm-s_2026-09-01.md)

## Changes

- Separada memória seletiva opcional do ASM-R do scaffold ASM-VR selecionado.
- Definida uma matriz causal ASM-VR-R versus ASM-VR-S antes da Transition Memory.

## Validation

- Referência histórica inspecionada: aos 100M tokens, ASM-S obteve CE 1.358291 em 0.7965 GPU-h versus ASM-R CE 1.344849 em 1.6762 GPU-h, seed única.
