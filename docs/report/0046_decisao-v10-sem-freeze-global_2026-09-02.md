# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Registrar a aprovação do V10 como desenho experimental local e a decisão explícita de não autorizar GLOBAL READY TO FREEZE nem freeze manifest/hash enquanto os sete grupos externos e quatro revisões globais não estiverem completos. Preservar V10/evidência e exigir V11/DRAFT para qualquer waiver.

## Summary

## Decisão registrada

A decisão do usuário foi registrada sem modificar o protocolo V10, Evidence V7 ou freeze candidate:

- `V10 DESIGN: APPROVED`
- `LOCAL READINESS: ACCEPTED`
- `GLOBAL READY TO FREEZE: NO`
- `FREEZE MANIFEST/HASH: NOT AUTHORIZED YET`

Protocol V10 SHA-256: `918851086373512855ffc653e292037dd9af5af6649c16d51e58dbe8e4d74422`  
Evidence V7 SHA-256: `d160335f9efa72fcf61d2b1cc364aa1e1a5bf4ee81eb4af4d676243d3291993b`  
User-disposition SHA-256: `7e25c21bcd5f7b3c1cc5fa572bd47fe14c825eb59674d3e6e297a32c992597da`

## Regra de não-waiver

Os sete grupos externos são requisitos materiais. Test keys, mocks, schemas ou serviços sob a mesma administração não podem substituí-los. Dispensar qualquer requisito exige um V11/DRAFT prospectivo, uma nova matriz e nova revisão; não pode ser chamado de freeze V10.

## Sequência autorizada

1. Preservar V10 e a evidência local byte-for-byte.
2. Resolver os sete grupos externos.
3. Executar o preflight real no ambiente final.
4. Coletar e arquivar os receipts.
5. Submeter o ambiente real e o snapshot exato às quatro revisões finais.
6. Exigir quatro `GLOBAL READY TO FREEZE: YES`.
7. Somente depois solicitar nova autorização explícita para criar o preregistration manifest/hash.

Nenhum freeze, seal, dado confirmatório, treino, calibration, test opening, commit ou push foi autorizado ou executado.

## Modified files

- [docs/review/attr_rtg_rcm_v1/user_disposition_v10.md](../review/attr_rtg_rcm_v1/user_disposition_v10.md)
- [docs/report/0046_decisao-v10-sem-freeze-global_2026-09-02.md](0046_decisao-v10-sem-freeze-global_2026-09-02.md)

## Changes

- Registrada a decisão formal do usuário sobre V10.
- Preservada a proibição de freeze global e de waiver silencioso dos requisitos externos.
- Documentada a sequência autorizada até uma futura decisão explícita de freeze.

## Validation

- Hashes de V10, Evidence V7 e freeze candidate V10 recomputados e inalterados
