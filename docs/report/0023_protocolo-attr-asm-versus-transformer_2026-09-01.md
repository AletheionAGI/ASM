# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Definir como próxima etapa, antes de retomar a Fase 3B, um experimento com o melhor baseline ASM versus um baseline Transformer existente na pasta transformer/, capaz de medir antecipação de transições perigosas, restrição do espaço de estados e previsibilidade; documentar como fazer isso de forma justa em ASM e Transformer.

## Summary

Documentado o ATTR — ASM–Transformer Transition Risk Benchmark em inglês e PT-BR. O protocolo cria HazardWorld, separa antecipação, previsibilidade e intervenção causal, usa heads e shield externos comuns, define pares parameter-matched ASM-X/Transformer-220K e ASM-R/GPT-2, leakage audits, telemetria, cinco seeds, métricas calibradas, contrafactuais clonados, OOD e gates sequenciais. O ATTR foi inserido no roadmap antes da Fase 3B. Nenhum claim de safety ou superioridade foi feito e nenhum experimento ATTR foi executado nesta solicitação.

## Modified files

- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL.md)
- [docs/ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md](../ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md)
- [docs/ARCHITECTURE_ASM_VR.md](../ARCHITECTURE_ASM_VR.md)
- [docs/report/0023_protocolo-attr-asm-versus-transformer_2026-09-01.md](0023_protocolo-attr-asm-versus-transformer_2026-09-01.md)

## Changes

- Criado ambiente proposto HazardWorld com hazards raros, falha atrasada, POMDP, recovery/stop, splits por mundo/dinâmica e teste selado.
- Definido par principal ASM-X directional_candidates 219.610 vs Transformer-220K 220.208, além de robustez ASM-R vs GPT-2 scratch.
- Definida interface comum de heads, probes congelados/end-to-end, shield externo hard e intervenções com simulador clonado/common random numbers.
- Pré-registradas métricas AUPRC, Brier, lead time útil, risco causal, CVaR, utilidade, calibração, OOD e gates G0–G5.
- Posicionado ATTR como interlúdio P0–P3 antes do retorno à Fase 3B; ASM-CM-VR permanece linha separada.

## Validation

- Inventário independente da pasta transformer/ e dos controles GPT-2 existentes — concluído.
- Inventário independente de RiskField, directional_candidates, telemetria ASM e benchmarks world-model — concluído.
- Revisão crítica independente de leakage, causalidade, métricas, estatística e claims — concluída.
- Links Markdown relativos, fences de código e git diff --check — passaram.
