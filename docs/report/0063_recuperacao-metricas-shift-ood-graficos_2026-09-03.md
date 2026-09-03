# Request Report

- Status: completed
- Date: 2026-09-03

## User request

Corrigir a perda de métricas shift/OOD, recuperar as métricas dos 20 checkpoints sem treino e gerar os gráficos oficiais.

## Summary

## Resultado

A retenção independente por métrica foi corrigida e a recuperação oficial score-only terminou com exit code 0 em 571.8 segundos. Os 20 checkpoints registrados foram validados e reutilizados. Nenhum treino, optimizer step, escrita ou substituição de checkpoint ocorreu.

As 66 rows continuam fail-closed: 20 ID são VALID; 20 shift, 20 OOD e seis contrasts continuam INVALID porque `safe_service` não tem denominador em pelo menos um fold. A recuperação não imputou nem relaxou esse endpoint. Em vez disso, preservou as outras métricas finitas.

| Regime | H8 NLL | ECE-15 | Unsafe selection | Safe service | Coverage | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| ID | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| shift | 20/20 | 20/20 | 20/20 | 0/20 | 20/20 | 20/20 |
| OOD | 20/20 | 20/20 | 20/20 | 0/20 | 20/20 | 20/20 |

Os scalars ID permaneceram idênticos à release anterior. O SHA-256 do novo `official_rows.json` é `50eac02110e7e0af3e12a1bb0d7e9a7da7d1ea7b3b94570e1f2588f5f3dfb3f0`. O receipt é `docs/review/attr_rtg_rcmz_v1/official_metric_recovery_receipt_v2.json` SHA-256 `3b66518100423bddc833576b28db5bcd6dd6b4637138c18908cdc4e5e9230eb9`.

## Gráficos

Foram regenerados e promovidos ao diretório canônico `artifacts/attr_rtg_rcmz/official/`:

- `architecture_quality.png/svg`: H8 NLL e ECE-15 agora mostram ID, shift e OOD;
- `governance.png/svg`: unsafe selection e coverage mostram os três regimes; safe service permanece vazio em shift/OOD por denominador ausente;
- `g_vs_c.png/svg`: equivalente de decisão no filename legado, com os mesmos limites fail-closed;
- `seed_differences.png/svg`: deltas por seed recalculados;
- `summary.png/svg/html` e manifests JSON/CSV.

Os quatro PNG foram inspecionados visualmente. As barras shift/OOD estão presentes para todas as métricas elegíveis. Safe service permanece corretamente vazio fora de ID. Os seis gates agregados permanecem fail-closed; nenhum vencedor global é declarado.

## Integridade

- Recovery lock V8: `611282851b42fad136986ce4f8d50d3541d0540358862a9ed69196f4e55c816d`.
- Checkpoints originais: todos os 20 hashes permanecem iguais.
- Diretório score-only não contém diretório de checkpoints.
- ID metrics: idênticas à release anterior.
- Release anterior: preservada em `artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/`.

## Modified files

- [artifacts/attr_rtg_rcmz/official/architecture_quality.png](../../artifacts/attr_rtg_rcmz/official/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/official/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/official/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/official/g_vs_c.png](../../artifacts/attr_rtg_rcmz/official/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/official/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/official/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/official/governance.png](../../artifacts/attr_rtg_rcmz/official/governance.png)
- [artifacts/attr_rtg_rcmz/official/governance.svg](../../artifacts/attr_rtg_rcmz/official/governance.svg)
- [artifacts/attr_rtg_rcmz/official/manifest.csv](../../artifacts/attr_rtg_rcmz/official/manifest.csv)
- [artifacts/attr_rtg_rcmz/official/manifest.json](../../artifacts/attr_rtg_rcmz/official/manifest.json)
- [artifacts/attr_rtg_rcmz/official/metric_recovery_run_v2.log](../../artifacts/attr_rtg_rcmz/official/metric_recovery_run_v2.log)
- [artifacts/attr_rtg_rcmz/official/metric_recovery_runtime_status_v2.json](../../artifacts/attr_rtg_rcmz/official/metric_recovery_runtime_status_v2.json)
- [artifacts/attr_rtg_rcmz/official/metric_recovery_status_v2.json](../../artifacts/attr_rtg_rcmz/official/metric_recovery_status_v2.json)
- [artifacts/attr_rtg_rcmz/official/metric_recovery_status_v2.sha256](../../artifacts/attr_rtg_rcmz/official/metric_recovery_status_v2.sha256)
- [artifacts/attr_rtg_rcmz/official/official_rows.json](../../artifacts/attr_rtg_rcmz/official/official_rows.json)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/architecture_quality.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/g_vs_c.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/governance.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/governance.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/governance.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/governance.svg)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/manifest.csv](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/manifest.csv)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/manifest.json](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/manifest.json)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/official_rows.json](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/official_rows.json)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/postprocess_status.json](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/postprocess_status.json)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/postprocess_status.sha256](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/postprocess_status.sha256)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/recovery_manifest.json](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/recovery_manifest.json)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/run.log](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/run.log)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/seed_differences.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/seed_differences.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/seed_differences.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/status.json](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/status.json)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/summary.html](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/summary.html)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/summary.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/summary.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/summary.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/summary.svg)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/supervisor_recovery_v1.json](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-partial-metrics/supervisor_recovery_v1.json)
- [artifacts/attr_rtg_rcmz/official/recovery_input_all20_v2.json](../../artifacts/attr_rtg_rcmz/official/recovery_input_all20_v2.json)
- [artifacts/attr_rtg_rcmz/official/seed_differences.png](../../artifacts/attr_rtg_rcmz/official/seed_differences.png)
- [artifacts/attr_rtg_rcmz/official/seed_differences.svg](../../artifacts/attr_rtg_rcmz/official/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/official/summary.html](../../artifacts/attr_rtg_rcmz/official/summary.html)
- [artifacts/attr_rtg_rcmz/official/summary.png](../../artifacts/attr_rtg_rcmz/official/summary.png)
- [artifacts/attr_rtg_rcmz/official/summary.svg](../../artifacts/attr_rtg_rcmz/official/summary.svg)
- [artifacts/attr_rtg_rcmz/official/supervisor_metric_recovery_v2.json](../../artifacts/attr_rtg_rcmz/official/supervisor_metric_recovery_v2.json)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/architecture_quality.png](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/g_vs_c.png](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/governance.png](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/governance.png)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/governance.svg](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/governance.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/manifest.csv](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/manifest.csv)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/manifest.json](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/manifest.json)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/official_rows.json](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/official_rows.json)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/recovery_manifest.json](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/recovery_manifest.json)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/run.log](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/run.log)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/seed_differences.png](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/seed_differences.png)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/seed_differences.svg](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/status.json](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/status.json)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/summary.html](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/summary.html)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/summary.png](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/summary.png)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/summary.svg](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/summary.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_all20_v2/supervisor.json](../../artifacts/attr_rtg_rcmz/official_score_only_all20_v2/supervisor.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v8.json](../review/attr_rtg_rcmz_v1/local_lock_manifest_v8.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v8.sha256](../review/attr_rtg_rcmz_v1/local_lock_manifest_v8.sha256)
- [docs/review/attr_rtg_rcmz_v1/official_metric_recovery_receipt_v2.json](../review/attr_rtg_rcmz_v1/official_metric_recovery_receipt_v2.json)
- [docs/review/attr_rtg_rcmz_v1/official_metric_recovery_receipt_v2.sha256](../review/attr_rtg_rcmz_v1/official_metric_recovery_receipt_v2.sha256)
- [docs/review/attr_rtg_rcmz_v1/operational_metric_retention_recovery_v1.md](../review/attr_rtg_rcmz_v1/operational_metric_retention_recovery_v1.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_operational_metric_recovery_v6.md](../review/attr_rtg_rcmz_v1/user_authorization_operational_metric_recovery_v6.md)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256)
- [locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v7.json](../../locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v7.json)
- [locks/attr_rtg_rcmz_v1/manifest.json](../../locks/attr_rtg_rcmz_v1/manifest.json)
- [locks/attr_rtg_rcmz_v1/manifest.sha256](../../locks/attr_rtg_rcmz_v1/manifest.sha256)
- [locks/attr_rtg_rcmz_v1/status.json](../../locks/attr_rtg_rcmz_v1/status.json)
- [locks/attr_rtg_rcmz_v1/status.sha256](../../locks/attr_rtg_rcmz_v1/status.sha256)
- [src/attr_rtg_rcmz/lock_anchor.py](../../src/attr_rtg_rcmz/lock_anchor.py)
- [src/attr_rtg_rcmz/official_contrasts.py](../../src/attr_rtg_rcmz/official_contrasts.py)
- [src/attr_rtg_rcmz/official_stats.py](../../src/attr_rtg_rcmz/official_stats.py)
- [tests/test_attr_rtg_rcmz_official_statistics.py](../../tests/test_attr_rtg_rcmz_official_statistics.py)
- [tests/test_attr_rtg_rcmz_recovery.py](../../tests/test_attr_rtg_rcmz_recovery.py)
- [docs/report/0063_recuperacao-metricas-shift-ood-graficos_2026-09-03.md](0063_recuperacao-metricas-shift-ood-graficos_2026-09-03.md)

## Changes

- Tornou a sumarização independente por métrica, preservando valores finitos quando safe_service não possui denominador.
- Tornou contrasts endpoint-wise, mantendo endpoints ausentes e gates fail-closed.
- Executou score-only nos 20 checkpoints, sem treino ou checkpoint write.
- Promoveu rows, dashboards, manifests e quatro famílias de gráficos ao diretório oficial.
- Criou receipts, autorização, amendment e lock V8.

## Validation

- Full ATTR-RTG-RCMZ suite — passed
- Score-only supervisor — exit code 0, COMPLETED
- 66 rows — 20 VALID and 46 INVALID
- Shift/OOD — 20/20 finite for five eligible metrics; safe_service 0/20
- 20 checkpoint SHA-256 values — unchanged
- ID scalars — unchanged
- Four PNG chart families — visual inspection passed
- Ruff — passed
- compileall — passed
- git diff --check — passed
- SOLID audit — zero ATTR-RTG-RCMZ exceptions/violations
- V8 candidate manifest/lock guard — 116 artifacts passed
