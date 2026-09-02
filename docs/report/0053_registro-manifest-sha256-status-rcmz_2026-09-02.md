# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Registrar os arquivos .sha256, manifest JSON e mudar o status do ATTR-RTG-RCMZ-V1.

## Summary

O registro pós-lock foi concluído sem modificar os 106 artifacts selados. O status autoritativo agora é `LOCAL PROTOCOL LOCK`, com execução oficial `NOT_STARTED` e proprietário da execução `USER_IN_OTHER_TERMINAL`.

Foram criados `manifest.json`, `status.json` e sidecars SHA-256 para o receipt, manifest e status. Também foi criado o sidecar do candidate manifest V4. Todos passaram em `sha256sum -c`. O lock canônico permaneceu válido com SHA-256 `f3d772784742307fe9939000f29e5b0dd09e7f0d82861df4897d0dd3bbb5c661`.

Os campos `DRAFT` existentes dentro dos artifacts selados não foram reescritos, pois isso invalidaria o candidate manifest e o lock. O novo `locks/attr_rtg_rcmz_v1/status.json` é o registro autoritativo pós-lock.

## Modified files

- [locks/attr_rtg_rcmz_v1/manifest.json](../../locks/attr_rtg_rcmz_v1/manifest.json)
- [locks/attr_rtg_rcmz_v1/status.json](../../locks/attr_rtg_rcmz_v1/status.json)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256)
- [locks/attr_rtg_rcmz_v1/manifest.sha256](../../locks/attr_rtg_rcmz_v1/manifest.sha256)
- [locks/attr_rtg_rcmz_v1/status.sha256](../../locks/attr_rtg_rcmz_v1/status.sha256)
- [docs/review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v4.sha256](../review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v4.sha256)
- [docs/report/0053_registro-manifest-sha256-status-rcmz_2026-09-02.md](0053_registro-manifest-sha256-status-rcmz_2026-09-02.md)

## Changes

- Registrado manifest JSON pós-lock com protocolo, candidate manifest, autorização, reviews, receipt e source anchor.
- Alterado o status autoritativo para LOCAL PROTOCOL LOCK e execução oficial NOT_STARTED.
- Criados e validados os sidecars SHA-256.

## Validation

- sha256sum -c LOCAL_PROTOCOL_LOCK.sha256 manifest.sha256 status.sha256 — passed
- sha256sum -c prelock_candidate_manifest_v4.sha256 — passed
- verify_canonical_lock — passed
