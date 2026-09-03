# ATTR-RTG-RCMZ-V1 operational hotfix amendment V1

Status: **AUTHORIZED RECOVERY — NOT YET RESEALED**

The first official attempt failed after all four seed-29 arms reached update 2,000. The isolated scorer assumed every output was rank 2 and called `float(list)` on post-candidate `common24 [B,6,24]`; `native_state [B,6,28]` had the same latent defect. No scientific result was released.

This hotfix changes only scorer output serialization and explicit recovery orchestration. Recursive tuple conversion now preserves tensor rank and float leaves. Recovery accepts only a complete registered R/CM/Z/T seed group at update 2,000 after exact file SHA-256, checkpoint config/update, model key, tensor shape/dtype and optimizer validation.

The four seed-29 checkpoints are reused byte-for-byte and must not be retrained, moved or overwritten:

- R `f23c03a275441b6bb6bf5bd5452ef562edd483c313c3e9e208385c72aba20826`
- CM `a42041302e65e4ff81191760962c939211018735f5a6dd172dab781ab96e46d6`
- Z `da591bfe492a589194b582fd22f119c647427a167819ed8ac62ac8ff1eac7110`
- T `ed20db56ca196abef95bfd108095ec561adef27515fd1eb82d6dd95a7105e0c3`

Recovery first scores seed 29 from these checkpoints, then begins new training at seed 43. Seeds 43, 71, 89 and 107 remain pending. Protocol, model architectures, configs, data, training updates/order, H8, calibration, metrics, bootstrap, contrasts, decisions and gates are unchanged. No scientific re-review or new claim is made.

Failure receipt: `docs/review/attr_rtg_rcmz_v1/official_failure_receipt_v1.json` SHA-256 `458e0870550211d2a1457d3adecd3faa853c62a418feb37cd198a4df5b5f989a`.
Predecessor lock SHA-256: `6ed6bd2b39460695a891c27c51cc417453e59eb10ad519660219f84a8cb8950e`.
Changed files:
- `src/attr_rtg_rcmz/scorer.py`
- `src/attr_rtg_rcmz/recovery.py`
- `src/attr_rtg_rcmz/cli.py`
- `src/attr_rtg_rcmz/official.py`
- `src/attr_rtg_rcmz/official_isolated.py`
- `src/attr_rtg_rcmz/official_training.py`
- `src/attr_rtg_rcmz/official_supervisor.py`
- `src/attr_rtg_rcmz/constants.py`
- `tests/test_attr_rtg_rcmz_scorer_isolation.py`
- `tests/test_attr_rtg_rcmz_recovery.py`
- `tests/test_attr_rtg_rcmz_supervisor.py`
- `tests/goldens/attr_rtg_rcmz_v1_synthetic.json`
- `.gitignore`
