# Request Report

- Status: completed
- Date: 2026-09-03

## User request

Refazer a avaliação corrigida para obter resultados numéricos de safe service em shift/OOD e gerar governance.png com essas barras.

## Summary

## Safe service recuperado e gráficos gerados

A avaliação score-only corrigida terminou com exit code 0 em 574,3 segundos. Ela reutilizou e validou os 20 checkpoints; não executou treino, optimizer step ou escrita de checkpoint.

Como o endpoint oficial `safe_service` é indefinido quando algum fold tem denominador zero, foi adicionada uma métrica separada e explicitamente diagnóstica: `safe_service_eligible_only`. Ela faz média igual por origin nos episódios elegíveis, média igual dos episódios elegíveis dentro do world e média igual dos worlds representados. Ela não entra nos contrasts ou gates oficiais.

### Resultado — média entre 5 seeds ± IC95%

| Regime | ASM-R | ASM-CM | ASM-Z | Transformer |
|---|---:|---:|---:|---:|
| ID | 0,9676 ± 0,0026 | 0,9391 ± 0,0157 | 0,9516 ± 0,0004 | 0,9737 ± 0,0087 |
| shift | 0,9672 ± 0,0014 | 0,7057 ± 0,1201 | 0,8947 ± 0,0525 | 0,9264 ± 0,0241 |
| OOD | 0,9740 ± 0,0000 | 0,5561 ± 0,1893 | 0,7292 ± 0,1370 | 0,9255 ± 0,0265 |

Higher is better. Em estimativa pontual, Transformer é maior em ID; ASM-R é maior em shift e OOD. Isso é descrição dos pontos diagnósticos, não declaração de vencedor estatístico.

Elegibilidade:

- ID: 128/128 folds;
- shift: 103/128 folds;
- OOD: 71/128 folds.

Os arquivos `governance.png/svg` e `g_vs_c.png/svg` agora mostram barras numéricas nos três regimes. O eixo central está rotulado `Safe service (eligible folds; diagnostic)`. Os PNG foram inspecionados visualmente.

O `safe_service` preregistrado continua `null` para shift/OOD, as cells continuam INVALID e os gates permanecem fail-closed. Nenhum valor diagnóstico foi usado para resgatar gates.

- Rows SHA-256: `e341ee5bb7113d29dab90ecd07f52fbad2d504dd28036ea9730fe36c5c8390bc`
- Receipt SHA-256: `a1d00de2da351a71321440982cd8c134fc5b8b1827eb8b868cf8b6d9f70013e7`
- Lock V9: `3c48351f130b2134614c925a1c36bba2a90bdf377355d80cead3874cec434d42`
- Outputs canônicos: `artifacts/attr_rtg_rcmz/official/`

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
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/architecture_quality.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/g_vs_c.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/governance.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/governance.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/governance.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/governance.svg)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/manifest.csv](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/manifest.csv)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/manifest.json](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/manifest.json)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/metric_recovery_status_v2.json](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/metric_recovery_status_v2.json)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/metric_recovery_status_v2.sha256](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/metric_recovery_status_v2.sha256)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/official_rows.json](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/official_rows.json)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/seed_differences.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/seed_differences.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/seed_differences.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/summary.html](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/summary.html)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/summary.png](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/summary.png)
- [artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/summary.svg](../../artifacts/attr_rtg_rcmz/official/recovery_archive/complete-before-safe-service-diagnostic/summary.svg)
- [artifacts/attr_rtg_rcmz/official/safe_service_diagnostic_run_v3.log](../../artifacts/attr_rtg_rcmz/official/safe_service_diagnostic_run_v3.log)
- [artifacts/attr_rtg_rcmz/official/safe_service_diagnostic_runtime_status_v3.json](../../artifacts/attr_rtg_rcmz/official/safe_service_diagnostic_runtime_status_v3.json)
- [artifacts/attr_rtg_rcmz/official/safe_service_diagnostic_status_v3.json](../../artifacts/attr_rtg_rcmz/official/safe_service_diagnostic_status_v3.json)
- [artifacts/attr_rtg_rcmz/official/safe_service_diagnostic_status_v3.sha256](../../artifacts/attr_rtg_rcmz/official/safe_service_diagnostic_status_v3.sha256)
- [artifacts/attr_rtg_rcmz/official/seed_differences.png](../../artifacts/attr_rtg_rcmz/official/seed_differences.png)
- [artifacts/attr_rtg_rcmz/official/seed_differences.svg](../../artifacts/attr_rtg_rcmz/official/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/official/summary.html](../../artifacts/attr_rtg_rcmz/official/summary.html)
- [artifacts/attr_rtg_rcmz/official/summary.png](../../artifacts/attr_rtg_rcmz/official/summary.png)
- [artifacts/attr_rtg_rcmz/official/summary.svg](../../artifacts/attr_rtg_rcmz/official/summary.svg)
- [artifacts/attr_rtg_rcmz/official/supervisor_safe_service_diagnostic_v3.json](../../artifacts/attr_rtg_rcmz/official/supervisor_safe_service_diagnostic_v3.json)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/architecture_quality.png](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/g_vs_c.png](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/governance.png](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/governance.png)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/governance.svg](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/governance.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/manifest.csv](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/manifest.csv)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/manifest.json](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/manifest.json)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/official_rows.json](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/official_rows.json)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/recovery_manifest.json](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/recovery_manifest.json)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/run.log](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/run.log)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/seed_differences.png](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/seed_differences.png)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/seed_differences.svg](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/status.json](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/status.json)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/summary.html](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/summary.html)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/summary.png](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/summary.png)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/summary.svg](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/summary.svg)
- [artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/supervisor.json](../../artifacts/attr_rtg_rcmz/official_score_only_safe_service_v3/supervisor.json)
- [docs/review/attr_rtg_rcmz_v1/diagnostic_safe_service_eligible_only_v1.md](../review/attr_rtg_rcmz_v1/diagnostic_safe_service_eligible_only_v1.md)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v9.json](../review/attr_rtg_rcmz_v1/local_lock_manifest_v9.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v9.sha256](../review/attr_rtg_rcmz_v1/local_lock_manifest_v9.sha256)
- [docs/review/attr_rtg_rcmz_v1/official_safe_service_diagnostic_receipt_v3.json](../review/attr_rtg_rcmz_v1/official_safe_service_diagnostic_receipt_v3.json)
- [docs/review/attr_rtg_rcmz_v1/official_safe_service_diagnostic_receipt_v3.sha256](../review/attr_rtg_rcmz_v1/official_safe_service_diagnostic_receipt_v3.sha256)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_safe_service_diagnostic_v7.md](../review/attr_rtg_rcmz_v1/user_authorization_safe_service_diagnostic_v7.md)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256)
- [locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v8.json](../../locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v8.json)
- [locks/attr_rtg_rcmz_v1/manifest.json](../../locks/attr_rtg_rcmz_v1/manifest.json)
- [locks/attr_rtg_rcmz_v1/manifest.sha256](../../locks/attr_rtg_rcmz_v1/manifest.sha256)
- [locks/attr_rtg_rcmz_v1/status.json](../../locks/attr_rtg_rcmz_v1/status.json)
- [locks/attr_rtg_rcmz_v1/status.sha256](../../locks/attr_rtg_rcmz_v1/status.sha256)
- [src/attr_rtg_rcmz/lock_anchor.py](../../src/attr_rtg_rcmz/lock_anchor.py)
- [src/attr_rtg_rcmz/official_stats.py](../../src/attr_rtg_rcmz/official_stats.py)
- [src/attr_rtg_rcmz/rendering/charts.py](../../src/attr_rtg_rcmz/rendering/charts.py)
- [tests/test_attr_rtg_rcmz_official_statistics.py](../../tests/test_attr_rtg_rcmz_official_statistics.py)
- [docs/report/0065_safe-service-shift-ood-governance_2026-09-03.md](0065_safe-service-shift-ood-governance_2026-09-03.md)

## Changes

- Adicionada métrica diagnóstica safe_service_eligible_only com contagem explícita de folds elegíveis.
- Reexecutado scoring dos 20 checkpoints sem treino.
- Regenerados e promovidos governance.png/svg e g_vs_c.png/svg com barras ID/shift/OOD.
- Preservada a semântica oficial fail-closed e excluído o diagnóstico dos gates.
- Criados receipt e lock operacional V9.

## Validation

- Score-only supervisor — exit code 0, COMPLETED in 574.3 s
- Safe-service diagnostic — 20/20 finite values in ID, shift and OOD
- Fold eligibility — ID 128/128, shift 103/128, OOD 71/128
- Official core metrics — unchanged from prior recovery
- 20 checkpoint SHA-256 values — unchanged; no checkpoint output directory created
- governance.png and g_vs_c.png — visual inspection passed
- 63 ATTR-RTG-RCMZ tests — passed
- Ruff — passed
- compileall — passed
- git diff --check — passed
- SOLID audit — zero ATTR-RTG-RCMZ exceptions/violations
- V9 lock guard — 116 artifacts passed
