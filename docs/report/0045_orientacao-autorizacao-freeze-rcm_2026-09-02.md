# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Esclarecer se agora o usuário deve avaliar e autorizar o freeze do ATTR-RTG-RCM.

## Summary

## Resposta

Sim, agora o usuário pode avaliar o candidato local V10. Porém uma autorização verbal de freeze, sozinha, não torna o V10 globalmente elegível.

O estado atual é:

- quatro domínios `NORMATIVE READY`;
- quatro domínios `LOCAL FACTUAL READY`;
- `GLOBAL READY TO FREEZE: NO`;
- nenhum freeze ou seal criado.

## Sequência correta

1. Revisar `docs/ATTR_RTG_RCM_PREREGISTRATION.md`, `docs/review/attr_rtg_rcm_v1/local_readiness_v10.md`, `docs/review/attr_rtg_rcm_v1/external_freeze_requirements_v10.md` e o relatório 0044.
2. Aprovar ou rejeitar o desenho V10 e suas emendas.
3. Fornecer/autorizar os sete grupos externos: clean-room/supply chain, principals, HSM, beacon+ledger/TSA, watchdog/relay, WORM e continuous attestation/no-egress.
4. Executar o preflight real, arquivar receipts e obter nova revisão exata dos quatro domínios com `GLOBAL READY TO FREEZE: YES`.
5. Somente então emitir uma autorização explícita para criar o preregistration freeze manifest/hash. Commit/push exigem autorização separada.

Se o usuário quiser dispensar qualquer requisito externo, isso não pode ser chamado de freeze V10. Seria uma alteração normativa prospectiva, exigindo um novo DRAFT, nova matriz e nova revisão antes de eventual freeze.

## Modified files

- [docs/report/0045_orientacao-autorizacao-freeze-rcm_2026-09-02.md](0045_orientacao-autorizacao-freeze-rcm_2026-09-02.md)

## Changes

- Documentado o fluxo correto de avaliação, artifacts externos, revisão global e autorização final de freeze.

## Validation

- Status V10 considerado: normativa e factual local READY; global NOT READY
