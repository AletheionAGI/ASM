# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Registrar que o usuário não quer usar serviços externos nem pagar e explicar as opções legítimas para continuar.

## Summary

## Decisão operacional

Entendido: nenhum serviço externo será contratado ou usado e nenhum custo externo será assumido.

Consequência necessária: o V10 deve permanecer `LOCAL READINESS ACCEPTED — GLOBAL NOT READY — NOT FROZEN`. Não é possível representar mocks, chaves locais, arquivos locais ou processos sob a mesma administração como HSM atestado, TSA/ledger, watchdog independente, WORM externo ou continuous remote attestation.

## Opções legítimas

### Opção A — preservar V10 e parar antes do estudo confirmatório

- manter V10/Evidence V7 byte-for-byte;
- não criar freeze manifest/hash;
- não gerar dados confirmatórios, treinar, calibrar ou abrir test;
- conservar o trabalho como desenho experimental local aprovado.

### Opção B — criar V11/DRAFT local-only, sem custo externo

V11 seria uma alteração prospectiva explícita. Ele:

- removeria os requisitos externos do protocolo aplicável;
- substituiria-os por controles locais de reprodutibilidade, hashes, append-only local, processos separados e receipts locais;
- declararia que esses controles não oferecem independência, non-equivocation externa, trusted time, HSM custody, WORM externo ou attestation remota;
- usaria um rótulo inequívoco, como `LOCAL-ONLY PREREGISTRATION` ou `REPRODUCIBILITY SNAPSHOT`, e não `V10 global freeze`;
- exigiria nova matriz, quatro revisões do desenho V11 e uma nova autorização antes de gerar qualquer dado confirmatório.

A recomendação, se o objetivo é continuar sem custo, é a Opção B. O agente não deve iniciar V11 sem autorização explícita, pois isso altera o desenho já aprovado.

## Modified files

- [docs/report/0048_opcoes-rcm-sem-servicos-externos_2026-09-02.md](0048_opcoes-rcm-sem-servicos-externos_2026-09-02.md)

## Changes

- Documentadas as consequências de operar sem serviços externos e as opções V10 preservado ou V11 local-only.

## Validation

- Mantida a decisão vigente: V10 não pode ser congelado globalmente sem os requisitos externos
