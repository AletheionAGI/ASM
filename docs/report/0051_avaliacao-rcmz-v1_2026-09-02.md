# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Registrar a avaliação do novo ATTR-RTG-RCMZ V1: design preferido, pronto para revisão final, mas ainda não pronto para lock até definir exatamente a métrica H8 e a decision rule.

## Summary

## Evaluation recorded

The user strongly prefers the simplified ATTR-RTG-RCMZ V1 over the former RCM V10/V11 design.

- Clarity: excellent
- Auditability: very good
- Architectural comparison: much better
- Operational complexity: drastically better
- Overengineering risk: low
- Ready for final review: yes
- Ready for lock: no

Protocol reviewed: `00bd9242f4bfc1fcd53ac76cebf4e71b9bb709cfbdbda3f10a35152ee9b8833b`  
Evaluation SHA-256: `513c74c6b2a10c324f18bbde0126c98c1df018f658de7a10cd3d2aaf2362f348`

The accepted scientific question is: under the same conditions, which of R, CM, strict Z and Transformer produces better risk prediction, better calibration and lower selection of dangerous actions?

Two exact definitions remain before lock:

1. **H8 metric:** inclusive horizon, first candidate transition, continuation policy, failure event, terminal/censoring behavior, target encoding and aggregation unit.
2. **Frozen decision rule:** calibrated score used, threshold derivation restricted to calibration, candidate eligibility, BLOCK/ABSTAIN behavior, BRAKE mapping, tie-breaking and exact definition of unsafe selection.

The protocol remains unchanged and paused. The evaluation does not authorize edits, implementation, review execution, data generation, training, calibration, test opening or lock creation.

## Modified files

- [docs/review/attr_rtg_rcmz_v1/user_evaluation_v1.md](../review/attr_rtg_rcmz_v1/user_evaluation_v1.md)
- [docs/report/0051_avaliacao-rcmz-v1_2026-09-02.md](0051_avaliacao-rcmz-v1_2026-09-02.md)

## Changes

- Registrada a avaliação do usuário e a pergunta científica aceita.
- Registrados H8 e decision rule como definições pendentes antes do lock.
- Mantida a pausa e o protocolo sem alterações.

## Validation

- SHA-256 do protocolo avaliado e do registro de avaliação computados
