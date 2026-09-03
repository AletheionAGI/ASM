# Request Report

- Status: completed
- Date: 2026-09-03

## User request

Confirmar a conclusão da execução oficial e finalizar os outputs sem repetir treino.

## Summary

A execução completou o treino de todos os cinco seeds e quatro braços: 20 checkpoints, cada um em update 2.000. O scoring também terminou e escreveu 66 rows oficiais. O processo retornou CRASH somente depois disso, quando o renderer tentou converter um valor JSON `null` com `float(None)`.

O renderer foi corrigido para preservar nulls nas tabelas e omiti-los somente dos cálculos visuais. Todos os outputs foram regenerados diretamente de `official_rows.json` SHA-256 `d0b7f2611528ffbeb6b150e6f98f4d8c68818fd2582c51a81f30bcd755d69902`, sem treino ou scoring adicional. Os quatro gráficos em barras foram inspecionados visualmente.

A release é fail-closed, não uma execução integralmente válida: 20 rows ID são VALID, 20 shift e 20 OOD são INVALID, e os seis contrasts são INVALID. Portanto, os gráficos mostram os valores elegíveis de ID e deixam shift/OOD vazios. Não há base para declarar vencedor global ou resgatar os gates inválidos.

- Training: complete, 20 checkpoints
- Official scalar rows: 66
- VALID: 20
- INVALID: 46
- Official rows SHA-256: `d0b7f2611528ffbeb6b150e6f98f4d8c68818fd2582c51a81f30bcd755d69902`
- Result receipt: `docs/review/attr_rtg_rcmz_v1/official_result_receipt_v1.json` SHA-256 `2a2afec143a3d78b00281a2611071799fa9a9f3c2dd747528c7232cba15d9c89`
- Postprocess status: `artifacts/attr_rtg_rcmz/official/postprocess_status.json` SHA-256 `f1dca963e1a66cc016b497599ce1766e234461e546a41e6b0e39d9c295c118c0`
- Postprocessing manifest V7: `4c593e9342268166b243f00c19617a7505a6060b288a0fe1c2b30734a3216585`
- Postprocessing lock SHA-256: `2af28e2ccf1c8c707e7bd71a9f765efc01232ba4644828466054f0bd116f317d`

Generated outputs include `summary.png/svg/html`, manifest JSON/CSV and `architecture_quality`, `governance`, `g_vs_c`, `seed_differences` in PNG/SVG. No model was retrained and no scalar row was changed.

## Modified files

- [artifacts/attr_rtg_rcmz/official/architecture_quality.png](../../artifacts/attr_rtg_rcmz/official/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/official/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/official/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/official/g_vs_c.png](../../artifacts/attr_rtg_rcmz/official/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/official/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/official/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/official/governance.png](../../artifacts/attr_rtg_rcmz/official/governance.png)
- [artifacts/attr_rtg_rcmz/official/governance.svg](../../artifacts/attr_rtg_rcmz/official/governance.svg)
- [artifacts/attr_rtg_rcmz/official/manifest.csv](../../artifacts/attr_rtg_rcmz/official/manifest.csv)
- [artifacts/attr_rtg_rcmz/official/manifest.json](../../artifacts/attr_rtg_rcmz/official/manifest.json)
- [artifacts/attr_rtg_rcmz/official/official_rows.json](../../artifacts/attr_rtg_rcmz/official/official_rows.json)
- [artifacts/attr_rtg_rcmz/official/postprocess_status.json](../../artifacts/attr_rtg_rcmz/official/postprocess_status.json)
- [artifacts/attr_rtg_rcmz/official/postprocess_status.sha256](../../artifacts/attr_rtg_rcmz/official/postprocess_status.sha256)
- [artifacts/attr_rtg_rcmz/official/seed_differences.png](../../artifacts/attr_rtg_rcmz/official/seed_differences.png)
- [artifacts/attr_rtg_rcmz/official/seed_differences.svg](../../artifacts/attr_rtg_rcmz/official/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/official/summary.html](../../artifacts/attr_rtg_rcmz/official/summary.html)
- [artifacts/attr_rtg_rcmz/official/summary.png](../../artifacts/attr_rtg_rcmz/official/summary.png)
- [artifacts/attr_rtg_rcmz/official/summary.svg](../../artifacts/attr_rtg_rcmz/official/summary.svg)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v7.json](../review/attr_rtg_rcmz_v1/local_lock_manifest_v7.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v7.sha256](../review/attr_rtg_rcmz_v1/local_lock_manifest_v7.sha256)
- [docs/review/attr_rtg_rcmz_v1/official_result_receipt_v1.json](../review/attr_rtg_rcmz_v1/official_result_receipt_v1.json)
- [docs/review/attr_rtg_rcmz_v1/official_result_receipt_v1.sha256](../review/attr_rtg_rcmz_v1/official_result_receipt_v1.sha256)
- [docs/review/attr_rtg_rcmz_v1/operational_rendering_hotfix_v1.md](../review/attr_rtg_rcmz_v1/operational_rendering_hotfix_v1.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_operational_rendering_v5.md](../review/attr_rtg_rcmz_v1/user_authorization_operational_rendering_v5.md)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256)
- [locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v6_3d640d25beba627c.json](../../locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v6_3d640d25beba627c.json)
- [locks/attr_rtg_rcmz_v1/manifest.json](../../locks/attr_rtg_rcmz_v1/manifest.json)
- [locks/attr_rtg_rcmz_v1/manifest.sha256](../../locks/attr_rtg_rcmz_v1/manifest.sha256)
- [locks/attr_rtg_rcmz_v1/status.json](../../locks/attr_rtg_rcmz_v1/status.json)
- [locks/attr_rtg_rcmz_v1/status.sha256](../../locks/attr_rtg_rcmz_v1/status.sha256)
- [src/attr_rtg_rcmz/lock_anchor.py](../../src/attr_rtg_rcmz/lock_anchor.py)
- [src/attr_rtg_rcmz/rendering/artifacts.py](../../src/attr_rtg_rcmz/rendering/artifacts.py)
- [src/attr_rtg_rcmz/rendering/charts.py](../../src/attr_rtg_rcmz/rendering/charts.py)
- [tests/test_attr_rtg_rcmz_cli.py](../../tests/test_attr_rtg_rcmz_cli.py)
- [docs/report/0061_finalizacao-rendering-official-rcmz_2026-09-03.md](0061_finalizacao-rendering-official-rcmz_2026-09-03.md)

## Changes

- Confirmado treino completo de 20 arm/seed checkpoints e 66 rows oficiais.
- Corrigido renderer null-safe sem alterar scalar rows.
- Regenerados os quatro gráficos RTG-style em barras e dashboard.
- Criados result receipt, postprocess status, manifest V7 e lock pós-processamento.

## Validation

- 58 testes ATTR-RTG-RCMZ — passed
- Ruff — passed
- compileall — passed
- git diff --check — passed
- SOLID audit — zero RCMZ exceptions/violations
- verify_candidate_manifest — 116 artifacts passed
- verify_canonical_lock — passed
- V7/result/lock sidecars — passed
- Four official PNG charts — visual inspection passed
