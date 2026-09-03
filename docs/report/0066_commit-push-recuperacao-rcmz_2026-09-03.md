# Request Report

- Status: completed
- Date: 2026-09-03

## User request

Commitar e fazer push de tudo, exceto arquivos grandes.

## Summary

Todo o workspace pendente foi revisado para publicação. Nenhum arquivo pendente excede 1 MiB; o maior é um dashboard HTML de aproximadamente 386 KiB. Checkpoints, archives de runtime e demais arquivos grandes já ignorados pelo Git permanecem fora do commit.

O commit inclui o hotfix do scorer, recovery all-20 sem retreino, retenção independente de métricas, safe-service eligible-only diagnóstico, gráficos PNG/SVG/HTML, receipts operacionais, manifests/locks V6–V9, testes e relatórios. Os dois diretórios score-only publicados somam aproximadamente 2,5 MiB em arquivos pequenos.

Antes do commit: 63 testes ATTR-RTG-RCMZ passaram; Ruff, compileall, git diff --check, lock guard V9 e auditoria SOLID passaram. O branch local e `origin/main` estavam sincronizados em `89fefe4` antes da publicação. Nenhum padrão comum de credential/private key foi encontrado nos arquivos alterados.

## Modified files

- [.gitignore](../../.gitignore)
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
- [docs/report/0059_estimativa-tempo-treino-oficial-rcmz_2026-09-02.md](0059_estimativa-tempo-treino-oficial-rcmz_2026-09-02.md)
- [docs/report/0060_hotfix-scorer-recovery-seed29-sem-retreino_2026-09-02.md](0060_hotfix-scorer-recovery-seed29-sem-retreino_2026-09-02.md)
- [docs/report/0061_finalizacao-rendering-official-rcmz_2026-09-03.md](0061_finalizacao-rendering-official-rcmz_2026-09-03.md)
- [docs/report/0062_explicacao-shift-ood-invalid-rcmz_2026-09-03.md](0062_explicacao-shift-ood-invalid-rcmz_2026-09-03.md)
- [docs/report/0063_recuperacao-metricas-shift-ood-graficos_2026-09-03.md](0063_recuperacao-metricas-shift-ood-graficos_2026-09-03.md)
- [docs/report/0064_esclarecimento-safe-service-shift-ood_2026-09-03.md](0064_esclarecimento-safe-service-shift-ood_2026-09-03.md)
- [docs/report/0065_safe-service-shift-ood-governance_2026-09-03.md](0065_safe-service-shift-ood-governance_2026-09-03.md)
- [docs/review/attr_rtg_rcmz_v1/diagnostic_safe_service_eligible_only_v1.md](../review/attr_rtg_rcmz_v1/diagnostic_safe_service_eligible_only_v1.md)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v6.json](../review/attr_rtg_rcmz_v1/local_lock_manifest_v6.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v6.sha256](../review/attr_rtg_rcmz_v1/local_lock_manifest_v6.sha256)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v7.json](../review/attr_rtg_rcmz_v1/local_lock_manifest_v7.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v7.sha256](../review/attr_rtg_rcmz_v1/local_lock_manifest_v7.sha256)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v8.json](../review/attr_rtg_rcmz_v1/local_lock_manifest_v8.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v8.sha256](../review/attr_rtg_rcmz_v1/local_lock_manifest_v8.sha256)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v9.json](../review/attr_rtg_rcmz_v1/local_lock_manifest_v9.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v9.sha256](../review/attr_rtg_rcmz_v1/local_lock_manifest_v9.sha256)
- [docs/review/attr_rtg_rcmz_v1/official_failure_receipt_v1.json](../review/attr_rtg_rcmz_v1/official_failure_receipt_v1.json)
- [docs/review/attr_rtg_rcmz_v1/official_failure_receipt_v1.sha256](../review/attr_rtg_rcmz_v1/official_failure_receipt_v1.sha256)
- [docs/review/attr_rtg_rcmz_v1/official_metric_recovery_receipt_v2.json](../review/attr_rtg_rcmz_v1/official_metric_recovery_receipt_v2.json)
- [docs/review/attr_rtg_rcmz_v1/official_metric_recovery_receipt_v2.sha256](../review/attr_rtg_rcmz_v1/official_metric_recovery_receipt_v2.sha256)
- [docs/review/attr_rtg_rcmz_v1/official_result_receipt_v1.json](../review/attr_rtg_rcmz_v1/official_result_receipt_v1.json)
- [docs/review/attr_rtg_rcmz_v1/official_result_receipt_v1.sha256](../review/attr_rtg_rcmz_v1/official_result_receipt_v1.sha256)
- [docs/review/attr_rtg_rcmz_v1/official_safe_service_diagnostic_receipt_v3.json](../review/attr_rtg_rcmz_v1/official_safe_service_diagnostic_receipt_v3.json)
- [docs/review/attr_rtg_rcmz_v1/official_safe_service_diagnostic_receipt_v3.sha256](../review/attr_rtg_rcmz_v1/official_safe_service_diagnostic_receipt_v3.sha256)
- [docs/review/attr_rtg_rcmz_v1/operational_hotfix_amendment_v1.md](../review/attr_rtg_rcmz_v1/operational_hotfix_amendment_v1.md)
- [docs/review/attr_rtg_rcmz_v1/operational_metric_retention_recovery_v1.md](../review/attr_rtg_rcmz_v1/operational_metric_retention_recovery_v1.md)
- [docs/review/attr_rtg_rcmz_v1/operational_rendering_hotfix_v1.md](../review/attr_rtg_rcmz_v1/operational_rendering_hotfix_v1.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_operational_metric_recovery_v6.md](../review/attr_rtg_rcmz_v1/user_authorization_operational_metric_recovery_v6.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_operational_recovery_v4.md](../review/attr_rtg_rcmz_v1/user_authorization_operational_recovery_v4.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_operational_rendering_v5.md](../review/attr_rtg_rcmz_v1/user_authorization_operational_rendering_v5.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_safe_service_diagnostic_v7.md](../review/attr_rtg_rcmz_v1/user_authorization_safe_service_diagnostic_v7.md)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256)
- [locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v5_6ed6bd2b39460695.json](../../locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v5_6ed6bd2b39460695.json)
- [locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v6_3d640d25beba627c.json](../../locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v6_3d640d25beba627c.json)
- [locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v7.json](../../locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v7.json)
- [locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v8.json](../../locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v8.json)
- [locks/attr_rtg_rcmz_v1/manifest.json](../../locks/attr_rtg_rcmz_v1/manifest.json)
- [locks/attr_rtg_rcmz_v1/manifest.sha256](../../locks/attr_rtg_rcmz_v1/manifest.sha256)
- [locks/attr_rtg_rcmz_v1/status.json](../../locks/attr_rtg_rcmz_v1/status.json)
- [locks/attr_rtg_rcmz_v1/status.sha256](../../locks/attr_rtg_rcmz_v1/status.sha256)
- [src/attr_rtg_rcmz/cli.py](../../src/attr_rtg_rcmz/cli.py)
- [src/attr_rtg_rcmz/constants.py](../../src/attr_rtg_rcmz/constants.py)
- [src/attr_rtg_rcmz/lock_anchor.py](../../src/attr_rtg_rcmz/lock_anchor.py)
- [src/attr_rtg_rcmz/official.py](../../src/attr_rtg_rcmz/official.py)
- [src/attr_rtg_rcmz/official_contrasts.py](../../src/attr_rtg_rcmz/official_contrasts.py)
- [src/attr_rtg_rcmz/official_isolated.py](../../src/attr_rtg_rcmz/official_isolated.py)
- [src/attr_rtg_rcmz/official_stats.py](../../src/attr_rtg_rcmz/official_stats.py)
- [src/attr_rtg_rcmz/official_supervisor.py](../../src/attr_rtg_rcmz/official_supervisor.py)
- [src/attr_rtg_rcmz/official_training.py](../../src/attr_rtg_rcmz/official_training.py)
- [src/attr_rtg_rcmz/recovery.py](../../src/attr_rtg_rcmz/recovery.py)
- [src/attr_rtg_rcmz/rendering/artifacts.py](../../src/attr_rtg_rcmz/rendering/artifacts.py)
- [src/attr_rtg_rcmz/rendering/charts.py](../../src/attr_rtg_rcmz/rendering/charts.py)
- [src/attr_rtg_rcmz/scorer.py](../../src/attr_rtg_rcmz/scorer.py)
- [tests/goldens/attr_rtg_rcmz_v1_synthetic.json](../../tests/goldens/attr_rtg_rcmz_v1_synthetic.json)
- [tests/test_attr_rtg_rcmz_cli.py](../../tests/test_attr_rtg_rcmz_cli.py)
- [tests/test_attr_rtg_rcmz_official_statistics.py](../../tests/test_attr_rtg_rcmz_official_statistics.py)
- [tests/test_attr_rtg_rcmz_recovery.py](../../tests/test_attr_rtg_rcmz_recovery.py)
- [tests/test_attr_rtg_rcmz_scorer_isolation.py](../../tests/test_attr_rtg_rcmz_scorer_isolation.py)
- [tests/test_attr_rtg_rcmz_supervisor.py](../../tests/test_attr_rtg_rcmz_supervisor.py)
- [docs/report/0066_commit-push-recuperacao-rcmz_2026-09-03.md](0066_commit-push-recuperacao-rcmz_2026-09-03.md)

## Changes

- Incluídas todas as mudanças pendentes e artifacts pequenos.
- Mantidos fora do Git checkpoints e runtime grande já ignorados.
- Preparado commit único para origin/main.

## Validation

- 63 ATTR-RTG-RCMZ tests — passed
- Ruff — passed
- compileall — passed
- git diff --check — passed
- SOLID audit — zero ATTR-RTG-RCMZ exceptions/violations
- V9 lock guard — 116 artifacts passed
- Changed-file size audit — largest 386149 bytes; no pending file >1 MiB
- Credential pattern scan — no hits
- git fetch and divergence check — local synchronized with origin/main
