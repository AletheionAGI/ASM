# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Alterar o status do Markdown de DRAFT para LOCAL PROTOCOL LOCK, sem revisão pesada, e gerar o novo manifest e SHA-256.

## Summary

Foi executado um reseal leve e estritamente status-only. O Markdown ativo agora declara `STATUS: LOCAL PROTOCOL LOCK — LOCAL-ONLY` e `Current state: LOCAL PROTOCOL LOCK — LOCAL-ONLY`.

A comparação textual confirma que somente metadados de status/autorização operacional foram alterados. Definições científicas, modelos, configs, CUDA evidence, estatística e resultados não mudaram. O lock anterior foi preservado em `locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v4_f3d772784742307f.json`.

Novos registros:

- Protocol SHA-256: `c37fa09bdad9715d82d5cb6b6108ce5d2147462c79738674abb75ca50dbc0f84`
- Manifest V5: `docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v5.json`
- Manifest V5 SHA-256: `0edca10830c63363bbfe5dd1253f488b25e961f784610caefa7b017c607551a6`
- Manifest V5 content SHA-256: `ec51589784d8320ad19999eca929aa84f57e56ac5bc47b7de0c7e4e6730614af`
- Manifest sidecar: `docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v5.sha256`
- Novo lock SHA-256: `6ed6bd2b39460695a891c27c51cc417453e59eb10ad519660219f84a8cb8950e`
- Lock: `locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json`

O manifest contém 107 artifacts. O guard revalidou todos os artifacts e o lock canônico. A execução oficial continua `NOT_STARTED` e atribuída ao usuário em outro terminal.

## Modified files

- [docs/ATTR_RTG_RCMZ_PREREGISTRATION.md](../ATTR_RTG_RCMZ_PREREGISTRATION.md)
- [docs/review/attr_rtg_rcmz_v1/locked_v1_r6_c37fa09bdad9715d.md](../review/attr_rtg_rcmz_v1/locked_v1_r6_c37fa09bdad9715d.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_status_only_reseal_v3.md](../review/attr_rtg_rcmz_v1/user_authorization_status_only_reseal_v3.md)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v5.json](../review/attr_rtg_rcmz_v1/local_lock_manifest_v5.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v5.sha256](../review/attr_rtg_rcmz_v1/local_lock_manifest_v5.sha256)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256)
- [locks/attr_rtg_rcmz_v1/manifest.json](../../locks/attr_rtg_rcmz_v1/manifest.json)
- [locks/attr_rtg_rcmz_v1/manifest.sha256](../../locks/attr_rtg_rcmz_v1/manifest.sha256)
- [locks/attr_rtg_rcmz_v1/status.json](../../locks/attr_rtg_rcmz_v1/status.json)
- [locks/attr_rtg_rcmz_v1/status.sha256](../../locks/attr_rtg_rcmz_v1/status.sha256)
- [locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v4_f3d772784742307f.json](../../locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v4_f3d772784742307f.json)
- [src/attr_rtg_rcmz/lock_anchor.py](../../src/attr_rtg_rcmz/lock_anchor.py)
- [tests/test_attr_rtg_rcmz_cli.py](../../tests/test_attr_rtg_rcmz_cli.py)
- [docs/report/0055_reseal-status-local-protocol-lock-rcmz_2026-09-02.md](0055_reseal-status-local-protocol-lock-rcmz_2026-09-02.md)

## Changes

- Alterado o status do Markdown ativo para LOCAL PROTOCOL LOCK.
- Criados snapshot locked R6, autorização status-only e manifest V5 com sidecar SHA-256.
- Arquivado o lock V4 e emitido novo lock canônico ancorado no manifest V5.
- Atualizados manifest/status pós-lock e seus sidecars.

## Validation

- Protocol status grep — passed
- 10 focused lock/CLI tests — passed
- verify_candidate_manifest — 107 artifacts passed
- verify_canonical_lock — passed
- sha256sum -c lock/manifest/status/V5 sidecars — passed
- git diff --check — passed
- SOLID source audit — zero RCMZ exceptions/violations
