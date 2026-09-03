# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Corrigir o erro isolated scorer TypeError float(list) sem treinar novamente o seed 29; preservar seus quatro checkpoints e preparar retomada explícita para o usuário executar.

## Summary

## Correção

A falha foi isolada no serializer do scorer: logits eram rank 2, mas `common24` e `native_state` são rank 3. A função antiga tentava executar `float()` sobre listas internas. Ela foi substituída por conversão recursiva imutável que preserva todas as dimensões e converte apenas folhas escalares em `float`.

A correção foi reproduzida com os quatro checkpoints oficiais reais do seed 29, sem treino: R, CM, Z e T retornaram logits `[2,6]`, common24 `[2,6,24]` e native state `[2,6,28]` com sucesso.

## Recuperação sem retreino

Os quatro checkpoints seed-29/update-2000 foram copiados para um arquivo imutável e verificados byte-for-byte. A retomada aceita apenas o grupo completo R/CM/Z/T após validar SHA-256, seed, arm, update, config, chaves/tensor shape/dtype do modelo e optimizer mapping. O seed 29 terá somente scoring/calibration/test; nenhuma de suas quatro arms será treinada novamente. O próximo treino começa no seed 43 e continua 71, 89 e 107.

Hashes preservados:

- R `f23c03a275441b6bb6bf5bd5452ef562edd483c313c3e9e208385c72aba20826`
- CM `a42041302e65e4ff81191760962c939211018735f5a6dd172dab781ab96e46d6`
- Z `da591bfe492a589194b582fd22f119c647427a167819ed8ac62ac8ff1eac7110`
- T `ed20db56ca196abef95bfd108095ec561adef27515fd1eb82d6dd95a7105e0c3`

O failed attempt foi preservado em `artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer` e registrado por `docs/review/attr_rtg_rcmz_v1/official_failure_receipt_v1.json` SHA-256 `458e0870550211d2a1457d3adecd3faa853c62a418feb37cd198a4df5b5f989a`. Nenhum resultado científico foi liberado.

## Reseal operacional

O protocolo científico permaneceu com SHA-256 `c37fa09bdad9715d82d5cb6b6108ce5d2147462c79738674abb75ca50dbc0f84`. Foi criado o manifest V6 `docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v6.json` com 114 artifacts, SHA-256 `217ddbff40f9719a01010ec67d2634c1f613d09e355af5d33518a5f52c7cfd93` e content SHA-256 `997b9032b3f8728460c9b3885188a295d739a1a09d03c5a833b6192a7ae80287`. O novo lock canônico tem SHA-256 `3d640d25beba627c21b2088b40e9aa4e650c9fae89d9237f9fe4946f1ddcb6b2`. O lock anterior foi arquivado. Não houve revisão científica pesada nem mudança em modelo, dados, treino, H8, estatística, decisão ou gates.

## Comando de retomada para o usuário

```bash
cd /home/gnai-creator/dev/ai/ASM
PYTHONPATH=src:. .venv/bin/python -m attr_rtg_rcmz.official_supervisor \
  --output-dir artifacts/attr_rtg_rcmz/official \
  --lock-file locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json \
  --lock-sha256 3d640d25beba627c21b2088b40e9aa4e650c9fae89d9237f9fe4946f1ddcb6b2 \
  --receipt artifacts/attr_rtg_rcmz/official/supervisor_recovery_v1.json \
  --recovery-manifest artifacts/attr_rtg_rcmz/official/recovery_input_v1.json
```

O agente não executou a retomada. O supervisor arquivará tombstone/status/log anteriores, preservará checkpoints e registrará a nova tentativa.

## Modified files

- [.gitignore](../../.gitignore)
- [artifacts/attr_rtg_rcmz/official/recovery_input_v1.json](../../artifacts/attr_rtg_rcmz/official/recovery_input_v1.json)
- [artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/TOMBSTONE.json](../../artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/TOMBSTONE.json)
- [artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/checkpoints/seed-29_CM_update-2000.ckpt](../../artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/checkpoints/seed-29_CM_update-2000.ckpt)
- [artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/checkpoints/seed-29_R_update-2000.ckpt](../../artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/checkpoints/seed-29_R_update-2000.ckpt)
- [artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/checkpoints/seed-29_T_update-2000.ckpt](../../artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/checkpoints/seed-29_T_update-2000.ckpt)
- [artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/checkpoints/seed-29_Z_update-2000.ckpt](../../artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/checkpoints/seed-29_Z_update-2000.ckpt)
- [artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/run.log](../../artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/run.log)
- [artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/status.json](../../artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/status.json)
- [artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/supervisor.json](../../artifacts/attr_rtg_rcmz/official_failed_2026-09-03_s29_scorer/supervisor.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v6.json](../review/attr_rtg_rcmz_v1/local_lock_manifest_v6.json)
- [docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v6.sha256](../review/attr_rtg_rcmz_v1/local_lock_manifest_v6.sha256)
- [docs/review/attr_rtg_rcmz_v1/official_failure_receipt_v1.json](../review/attr_rtg_rcmz_v1/official_failure_receipt_v1.json)
- [docs/review/attr_rtg_rcmz_v1/official_failure_receipt_v1.sha256](../review/attr_rtg_rcmz_v1/official_failure_receipt_v1.sha256)
- [docs/review/attr_rtg_rcmz_v1/operational_hotfix_amendment_v1.md](../review/attr_rtg_rcmz_v1/operational_hotfix_amendment_v1.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_operational_recovery_v4.md](../review/attr_rtg_rcmz_v1/user_authorization_operational_recovery_v4.md)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.sha256)
- [locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v5_6ed6bd2b39460695.json](../../locks/attr_rtg_rcmz_v1/archive/LOCAL_PROTOCOL_LOCK_v5_6ed6bd2b39460695.json)
- [locks/attr_rtg_rcmz_v1/manifest.json](../../locks/attr_rtg_rcmz_v1/manifest.json)
- [locks/attr_rtg_rcmz_v1/manifest.sha256](../../locks/attr_rtg_rcmz_v1/manifest.sha256)
- [locks/attr_rtg_rcmz_v1/status.json](../../locks/attr_rtg_rcmz_v1/status.json)
- [locks/attr_rtg_rcmz_v1/status.sha256](../../locks/attr_rtg_rcmz_v1/status.sha256)
- [src/attr_rtg_rcmz/cli.py](../../src/attr_rtg_rcmz/cli.py)
- [src/attr_rtg_rcmz/constants.py](../../src/attr_rtg_rcmz/constants.py)
- [src/attr_rtg_rcmz/lock_anchor.py](../../src/attr_rtg_rcmz/lock_anchor.py)
- [src/attr_rtg_rcmz/official.py](../../src/attr_rtg_rcmz/official.py)
- [src/attr_rtg_rcmz/official_isolated.py](../../src/attr_rtg_rcmz/official_isolated.py)
- [src/attr_rtg_rcmz/official_supervisor.py](../../src/attr_rtg_rcmz/official_supervisor.py)
- [src/attr_rtg_rcmz/official_training.py](../../src/attr_rtg_rcmz/official_training.py)
- [src/attr_rtg_rcmz/recovery.py](../../src/attr_rtg_rcmz/recovery.py)
- [src/attr_rtg_rcmz/scorer.py](../../src/attr_rtg_rcmz/scorer.py)
- [tests/goldens/attr_rtg_rcmz_v1_synthetic.json](../../tests/goldens/attr_rtg_rcmz_v1_synthetic.json)
- [tests/test_attr_rtg_rcmz_recovery.py](../../tests/test_attr_rtg_rcmz_recovery.py)
- [tests/test_attr_rtg_rcmz_scorer_isolation.py](../../tests/test_attr_rtg_rcmz_scorer_isolation.py)
- [tests/test_attr_rtg_rcmz_supervisor.py](../../tests/test_attr_rtg_rcmz_supervisor.py)
- [docs/report/0060_hotfix-scorer-recovery-seed29-sem-retreino_2026-09-02.md](0060_hotfix-scorer-recovery-seed29-sem-retreino_2026-09-02.md)

## Changes

- Corrigido serializer rank-3 do scorer.
- Implementada recuperação fail-closed por checkpoint completo, sem retreino do seed 29.
- Arquivado e registrado o failed attempt e os quatro checkpoints.
- Criados amendment, autorização operacional, manifest V6 e novo lock.
- Preparado comando supervisionado de retomada; nenhuma execução oficial iniciada.

## Validation

- 57 testes ATTR-RTG-RCMZ — passed
- All four actual seed-29 checkpoints scorer reproduction — passed
- Recovery manifest validation against four actual checkpoints — passed
- Live/archive checkpoint byte comparison — passed
- Ruff — passed
- compileall — passed
- git diff --check — passed
- SOLID audit — zero RCMZ exceptions/violations
- verify_candidate_manifest — 114 artifacts passed
- verify_canonical_lock — passed
- SHA-256 sidecars — passed
