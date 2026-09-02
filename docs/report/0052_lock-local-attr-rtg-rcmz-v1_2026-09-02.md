# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Finalizar rapidamente o ATTR-RTG-RCMZ-V1, explicitar o pareamento e os 50.000 parâmetros de cada modelo, corrigir todos os blockers com subagentes, obter revisão final 4/4, criar o LOCAL PROTOCOL LOCK e entregar o comando para o usuário executar o treino em outro terminal, sem o agente iniciar o treino.

## Summary

## Resultado

ATTR-RTG-RCMZ-V1 foi concluído e recebeu quatro pareceres `READY TO LOCAL LOCK: YES`. O `LOCAL PROTOCOL LOCK` canônico foi criado e verificado. Nenhum dado oficial, treino, calibration ou test opening foi executado pelo agente; o usuário executará o treino em outro terminal.

## Pareamento

| Modelo | Trainable | Graph-active | Mismatch |
|---|---:|---:|---:|
| ASM-R | 50.000 | 50.000 | 0,000% |
| ASM-CM | 50.000 | 50.000 | 0,000% |
| ASM-Z | 50.000 | 50.000 | 0,000% |
| Transformer | 50.000 | 50.000 | 0,000% |

São 20 configurações: quatro braços por cinco seeds. Não existem parâmetros de padding nem catch-all activity links. Cada candidato é consumido por um fork pós-history da arquitetura registrada; ASM-Z faz exatamente um solve/update por frame de quatro bytes.

## Integridade e execução

- Protocol SHA-256: `c81b32e54a6a5213a5b571fe041bb5b1caa7317fd65f9d515ba19ba16e1c71ab`
- Candidate manifest SHA-256: `282a7cc6fd545948a42934125d2b4d5b289414239c14bfab6f2f4f06d9e654c9`
- Candidate content SHA-256: `cb1b7a2196783a6ed2d544e1d526d9880487ff73f1a82a217101b0d6fe8f1365`
- Lock receipt SHA-256: `f3d772784742307fe9939000f29e5b0dd09e7f0d82861df4897d0dd3bbb5c661`
- Lock path: `locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json`
- CUDA evidence payload: `c729f6249cb7bdd9ceb6b7f5dcd5bf1de0063d9cdcd077e148b8f73510d9741e` em duas runs idênticas
- Peak observado com bound: `3395329792` bytes, abaixo de `20*2^30`
- Scorer isolado em subprocesso; truth permanece no broker e é juntada somente após congelar os quatro braços.
- Supervisor real impõe deadline monotônico de 20 horas, transmite feedback e registra completion/crash/timeout.
- Os quatro gráficos usam barras agrupadas no estilo RTG e são entregues em PNG/SVG e HTML.

## Comando entregue ao usuário

```bash
cd /home/gnai-creator/dev/ai/ASM
PYTHONPATH=src:. .venv/bin/python -m attr_rtg_rcmz.official_supervisor \
  --output-dir artifacts/attr_rtg_rcmz/official \
  --lock-file locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json \
  --lock-sha256 f3d772784742307fe9939000f29e5b0dd09e7f0d82861df4897d0dd3bbb5c661 \
  --receipt artifacts/attr_rtg_rcmz/official/supervisor.json
```

O CLI interno publica heartbeat em até 10 segundos, ETA, fase, seed, braço, update, VRAM, `status.json` e `run.log`.

## Modified files

- [artifacts/attr_rtg_rcmz/dry_run/manifest.csv](../../artifacts/attr_rtg_rcmz/dry_run/manifest.csv)
- [artifacts/attr_rtg_rcmz/dry_run/manifest.json](../../artifacts/attr_rtg_rcmz/dry_run/manifest.json)
- [artifacts/attr_rtg_rcmz/dry_run/run.log](../../artifacts/attr_rtg_rcmz/dry_run/run.log)
- [artifacts/attr_rtg_rcmz/dry_run/status.json](../../artifacts/attr_rtg_rcmz/dry_run/status.json)
- [artifacts/attr_rtg_rcmz/dry_run/summary.html](../../artifacts/attr_rtg_rcmz/dry_run/summary.html)
- [artifacts/attr_rtg_rcmz/dry_run/summary.png](../../artifacts/attr_rtg_rcmz/dry_run/summary.png)
- [artifacts/attr_rtg_rcmz/dry_run/summary.svg](../../artifacts/attr_rtg_rcmz/dry_run/summary.svg)
- [artifacts/attr_rtg_rcmz/dry_run_preview/manifest.csv](../../artifacts/attr_rtg_rcmz/dry_run_preview/manifest.csv)
- [artifacts/attr_rtg_rcmz/dry_run_preview/manifest.json](../../artifacts/attr_rtg_rcmz/dry_run_preview/manifest.json)
- [artifacts/attr_rtg_rcmz/dry_run_preview/run.log](../../artifacts/attr_rtg_rcmz/dry_run_preview/run.log)
- [artifacts/attr_rtg_rcmz/dry_run_preview/status.json](../../artifacts/attr_rtg_rcmz/dry_run_preview/status.json)
- [artifacts/attr_rtg_rcmz/dry_run_preview/summary.html](../../artifacts/attr_rtg_rcmz/dry_run_preview/summary.html)
- [artifacts/attr_rtg_rcmz/dry_run_preview/summary.png](../../artifacts/attr_rtg_rcmz/dry_run_preview/summary.png)
- [artifacts/attr_rtg_rcmz/dry_run_preview/summary.svg](../../artifacts/attr_rtg_rcmz/dry_run_preview/summary.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/architecture_quality.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/checkpoints/seed-29_R_update-2.ckpt](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/checkpoints/seed-29_R_update-2.ckpt)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/g_vs_c.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/governance.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/governance.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/governance.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/governance.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/manifest.csv](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/manifest.csv)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/manifest.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/manifest.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/run.log](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/run.log)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/seed_differences.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/seed_differences.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/seed_differences.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/smoke_official_rows.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/smoke_official_rows.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/status.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/status.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/summary.html](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/summary.html)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/summary.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/summary.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_bars/summary.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_bars/summary.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/architecture_quality.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/checkpoints/seed-29_R_update-2.ckpt](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/checkpoints/seed-29_R_update-2.ckpt)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/g_vs_c.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/governance.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/governance.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/governance.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/governance.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/manifest.csv](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/manifest.csv)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/manifest.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/manifest.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/run.log](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/run.log)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/seed_differences.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/seed_differences.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/seed_differences.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/smoke_official_rows.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/smoke_official_rows.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/status.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/status.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/summary.html](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/summary.html)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/summary.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/summary.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/summary.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_candidate_v2/summary.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/architecture_quality.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/checkpoints/seed-29_R_update-2.ckpt](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/checkpoints/seed-29_R_update-2.ckpt)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/g_vs_c.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/governance.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/governance.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/governance.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/governance.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/manifest.csv](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/manifest.csv)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/manifest.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/manifest.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/run.log](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/run.log)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/seed_differences.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/seed_differences.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/seed_differences.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/smoke_official_rows.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/smoke_official_rows.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/status.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/status.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/summary.html](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/summary.html)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/summary.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/summary.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final/summary.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final/summary.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/architecture_quality.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/checkpoints/seed-29_R_update-2.ckpt](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/checkpoints/seed-29_R_update-2.ckpt)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/g_vs_c.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/governance.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/governance.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/governance.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/governance.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/manifest.csv](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/manifest.csv)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/manifest.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/manifest.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/run.log](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/run.log)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/seed_differences.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/seed_differences.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/seed_differences.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/smoke_official_rows.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/smoke_official_rows.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/status.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/status.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/summary.html](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/summary.html)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/summary.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/summary.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/summary.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v2/summary.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/architecture_quality.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/checkpoints/seed-29_R_update-2.ckpt](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/checkpoints/seed-29_R_update-2.ckpt)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/g_vs_c.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/governance.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/governance.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/governance.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/governance.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/manifest.csv](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/manifest.csv)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/manifest.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/manifest.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/run.log](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/run.log)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/seed_differences.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/seed_differences.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/seed_differences.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/smoke_official_rows.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/smoke_official_rows.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/status.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/status.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/summary.html](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/summary.html)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/summary.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/summary.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/summary.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_final_v4/summary.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/architecture_quality.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/architecture_quality.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/architecture_quality.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/architecture_quality.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/checkpoints/seed-29_R_update-2.ckpt](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/checkpoints/seed-29_R_update-2.ckpt)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/g_vs_c.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/g_vs_c.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/g_vs_c.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/g_vs_c.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/governance.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/governance.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/governance.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/governance.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/manifest.csv](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/manifest.csv)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/manifest.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/manifest.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/run.log](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/run.log)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/seed_differences.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/seed_differences.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/seed_differences.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/seed_differences.svg)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/smoke_official_rows.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/smoke_official_rows.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/status.json](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/status.json)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/summary.html](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/summary.html)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/summary.png](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/summary.png)
- [artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/summary.svg](../../artifacts/attr_rtg_rcmz/prelock_smoke_locked_candidate/summary.svg)
- [artifacts/attr_rtg_rcmz/smoke_official_final/checkpoints/seed-29_R_update-2.ckpt](../../artifacts/attr_rtg_rcmz/smoke_official_final/checkpoints/seed-29_R_update-2.ckpt)
- [artifacts/attr_rtg_rcmz/smoke_official_final/manifest.csv](../../artifacts/attr_rtg_rcmz/smoke_official_final/manifest.csv)
- [artifacts/attr_rtg_rcmz/smoke_official_final/manifest.json](../../artifacts/attr_rtg_rcmz/smoke_official_final/manifest.json)
- [artifacts/attr_rtg_rcmz/smoke_official_final/run.log](../../artifacts/attr_rtg_rcmz/smoke_official_final/run.log)
- [artifacts/attr_rtg_rcmz/smoke_official_final/smoke_official_rows.json](../../artifacts/attr_rtg_rcmz/smoke_official_final/smoke_official_rows.json)
- [artifacts/attr_rtg_rcmz/smoke_official_final/status.json](../../artifacts/attr_rtg_rcmz/smoke_official_final/status.json)
- [artifacts/attr_rtg_rcmz/smoke_official_final/summary.html](../../artifacts/attr_rtg_rcmz/smoke_official_final/summary.html)
- [artifacts/attr_rtg_rcmz/smoke_official_final/summary.png](../../artifacts/attr_rtg_rcmz/smoke_official_final/summary.png)
- [artifacts/attr_rtg_rcmz/smoke_official_final/summary.svg](../../artifacts/attr_rtg_rcmz/smoke_official_final/summary.svg)
- [configs/attr_rtg_rcmz_v1/cm_seed107.yaml](../../configs/attr_rtg_rcmz_v1/cm_seed107.yaml)
- [configs/attr_rtg_rcmz_v1/cm_seed29.yaml](../../configs/attr_rtg_rcmz_v1/cm_seed29.yaml)
- [configs/attr_rtg_rcmz_v1/cm_seed43.yaml](../../configs/attr_rtg_rcmz_v1/cm_seed43.yaml)
- [configs/attr_rtg_rcmz_v1/cm_seed71.yaml](../../configs/attr_rtg_rcmz_v1/cm_seed71.yaml)
- [configs/attr_rtg_rcmz_v1/cm_seed89.yaml](../../configs/attr_rtg_rcmz_v1/cm_seed89.yaml)
- [configs/attr_rtg_rcmz_v1/r_seed107.yaml](../../configs/attr_rtg_rcmz_v1/r_seed107.yaml)
- [configs/attr_rtg_rcmz_v1/r_seed29.yaml](../../configs/attr_rtg_rcmz_v1/r_seed29.yaml)
- [configs/attr_rtg_rcmz_v1/r_seed43.yaml](../../configs/attr_rtg_rcmz_v1/r_seed43.yaml)
- [configs/attr_rtg_rcmz_v1/r_seed71.yaml](../../configs/attr_rtg_rcmz_v1/r_seed71.yaml)
- [configs/attr_rtg_rcmz_v1/r_seed89.yaml](../../configs/attr_rtg_rcmz_v1/r_seed89.yaml)
- [configs/attr_rtg_rcmz_v1/t_seed107.yaml](../../configs/attr_rtg_rcmz_v1/t_seed107.yaml)
- [configs/attr_rtg_rcmz_v1/t_seed29.yaml](../../configs/attr_rtg_rcmz_v1/t_seed29.yaml)
- [configs/attr_rtg_rcmz_v1/t_seed43.yaml](../../configs/attr_rtg_rcmz_v1/t_seed43.yaml)
- [configs/attr_rtg_rcmz_v1/t_seed71.yaml](../../configs/attr_rtg_rcmz_v1/t_seed71.yaml)
- [configs/attr_rtg_rcmz_v1/t_seed89.yaml](../../configs/attr_rtg_rcmz_v1/t_seed89.yaml)
- [configs/attr_rtg_rcmz_v1/z_seed107.yaml](../../configs/attr_rtg_rcmz_v1/z_seed107.yaml)
- [configs/attr_rtg_rcmz_v1/z_seed29.yaml](../../configs/attr_rtg_rcmz_v1/z_seed29.yaml)
- [configs/attr_rtg_rcmz_v1/z_seed43.yaml](../../configs/attr_rtg_rcmz_v1/z_seed43.yaml)
- [configs/attr_rtg_rcmz_v1/z_seed71.yaml](../../configs/attr_rtg_rcmz_v1/z_seed71.yaml)
- [configs/attr_rtg_rcmz_v1/z_seed89.yaml](../../configs/attr_rtg_rcmz_v1/z_seed89.yaml)
- [docs/ATTR_RTG_RCMZ_PREREGISTRATION.md](../ATTR_RTG_RCMZ_PREREGISTRATION.md)
- [docs/review/attr_rtg_rcmz_v1/draft_v1_r2_d05f72f665606511.md](../review/attr_rtg_rcmz_v1/draft_v1_r2_d05f72f665606511.md)
- [docs/review/attr_rtg_rcmz_v1/draft_v1_r3_0bb486cca2069580.md](../review/attr_rtg_rcmz_v1/draft_v1_r3_0bb486cca2069580.md)
- [docs/review/attr_rtg_rcmz_v1/draft_v1_r4_50e8a15facf54bfc.md](../review/attr_rtg_rcmz_v1/draft_v1_r4_50e8a15facf54bfc.md)
- [docs/review/attr_rtg_rcmz_v1/draft_v1_r5_c81b32e54a6a5213.md](../review/attr_rtg_rcmz_v1/draft_v1_r5_c81b32e54a6a5213.md)
- [docs/review/attr_rtg_rcmz_v1/final_review_verdicts_v4.json](../review/attr_rtg_rcmz_v1/final_review_verdicts_v4.json)
- [docs/review/attr_rtg_rcmz_v1/h8_decision_cuda_amendment_v2.md](../review/attr_rtg_rcmz_v1/h8_decision_cuda_amendment_v2.md)
- [docs/review/attr_rtg_rcmz_v1/h8_decision_rule_amendment_v1.md](../review/attr_rtg_rcmz_v1/h8_decision_rule_amendment_v1.md)
- [docs/review/attr_rtg_rcmz_v1/local_protocol_lock_ceremony_v4.json](../review/attr_rtg_rcmz_v1/local_protocol_lock_ceremony_v4.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v1.json](../review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v1.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v2.json](../review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v2.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v3.json](../review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v3.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v4.json](../review/attr_rtg_rcmz_v1/prelock_candidate_manifest_v4.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/architecture_quality.png](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/architecture_quality.png)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/architecture_quality.svg](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/architecture_quality.svg)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/checkpoints/seed-29_R_update-2.ckpt](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/checkpoints/seed-29_R_update-2.ckpt)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/g_vs_c.png](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/g_vs_c.png)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/g_vs_c.svg](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/g_vs_c.svg)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/governance.png](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/governance.png)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/governance.svg](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/governance.svg)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/manifest.csv](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/manifest.csv)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/manifest.json](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/manifest.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/run.log](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/run.log)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/seed_differences.png](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/seed_differences.png)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/seed_differences.svg](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/seed_differences.svg)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/smoke_official_rows.json](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/smoke_official_rows.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/status.json](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/status.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/summary.html](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/summary.html)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/summary.png](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/summary.png)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/summary.svg](../review/attr_rtg_rcmz_v1/prelock_evidence/supervised_smoke_output/summary.svg)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run1.json](../review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run1.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run2.json](../review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run2.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run2.supervisor.json](../review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run2.supervisor.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run3.json](../review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run3.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run4.json](../review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run4.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run4.supervisor.json](../review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run4.supervisor.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run5.json](../review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run5.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run6.json](../review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run6.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run6.supervisor.json](../review/attr_rtg_rcmz_v1/prelock_evidence/synthetic_cuda_run6.supervisor.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_evidence/timeout_smoke.supervisor.json](../review/attr_rtg_rcmz_v1/prelock_evidence/timeout_smoke.supervisor.json)
- [docs/review/attr_rtg_rcmz_v1/prelock_readiness_v1.md](../review/attr_rtg_rcmz_v1/prelock_readiness_v1.md)
- [docs/review/attr_rtg_rcmz_v1/prelock_readiness_v3.md](../review/attr_rtg_rcmz_v1/prelock_readiness_v3.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_conditional_local_lock.md](../review/attr_rtg_rcmz_v1/user_authorization_conditional_local_lock.md)
- [docs/review/attr_rtg_rcmz_v1/user_authorization_conditional_local_lock_v2.md](../review/attr_rtg_rcmz_v1/user_authorization_conditional_local_lock_v2.md)
- [docs/review/attr_rtg_rcmz_v1/user_evaluation_v1.md](../review/attr_rtg_rcmz_v1/user_evaluation_v1.md)
- [locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json](../../locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json)
- [pyproject.toml](../../pyproject.toml)
- [scripts/run_attr_rtg_rcmz.py](../../scripts/run_attr_rtg_rcmz.py)
- [src/attr_rtg_rcmz/__init__.py](../../src/attr_rtg_rcmz/__init__.py)
- [src/attr_rtg_rcmz/adapter.py](../../src/attr_rtg_rcmz/adapter.py)
- [src/attr_rtg_rcmz/adapters.py](../../src/attr_rtg_rcmz/adapters.py)
- [src/attr_rtg_rcmz/backbones.py](../../src/attr_rtg_rcmz/backbones.py)
- [src/attr_rtg_rcmz/bootstrap.py](../../src/attr_rtg_rcmz/bootstrap.py)
- [src/attr_rtg_rcmz/calibration.py](../../src/attr_rtg_rcmz/calibration.py)
- [src/attr_rtg_rcmz/checkpoint.py](../../src/attr_rtg_rcmz/checkpoint.py)
- [src/attr_rtg_rcmz/cli.py](../../src/attr_rtg_rcmz/cli.py)
- [src/attr_rtg_rcmz/cm.py](../../src/attr_rtg_rcmz/cm.py)
- [src/attr_rtg_rcmz/config.py](../../src/attr_rtg_rcmz/config.py)
- [src/attr_rtg_rcmz/constants.py](../../src/attr_rtg_rcmz/constants.py)
- [src/attr_rtg_rcmz/contracts.py](../../src/attr_rtg_rcmz/contracts.py)
- [src/attr_rtg_rcmz/contrasts.py](../../src/attr_rtg_rcmz/contrasts.py)
- [src/attr_rtg_rcmz/data_contracts.py](../../src/attr_rtg_rcmz/data_contracts.py)
- [src/attr_rtg_rcmz/decision.py](../../src/attr_rtg_rcmz/decision.py)
- [src/attr_rtg_rcmz/ece.py](../../src/attr_rtg_rcmz/ece.py)
- [src/attr_rtg_rcmz/engine.py](../../src/attr_rtg_rcmz/engine.py)
- [src/attr_rtg_rcmz/evaluation_broker.py](../../src/attr_rtg_rcmz/evaluation_broker.py)
- [src/attr_rtg_rcmz/folds.py](../../src/attr_rtg_rcmz/folds.py)
- [src/attr_rtg_rcmz/gates.py](../../src/attr_rtg_rcmz/gates.py)
- [src/attr_rtg_rcmz/h8.py](../../src/attr_rtg_rcmz/h8.py)
- [src/attr_rtg_rcmz/interfaces.py](../../src/attr_rtg_rcmz/interfaces.py)
- [src/attr_rtg_rcmz/lock_anchor.py](../../src/attr_rtg_rcmz/lock_anchor.py)
- [src/attr_rtg_rcmz/lock_guard.py](../../src/attr_rtg_rcmz/lock_guard.py)
- [src/attr_rtg_rcmz/manifests.py](../../src/attr_rtg_rcmz/manifests.py)
- [src/attr_rtg_rcmz/metrics.py](../../src/attr_rtg_rcmz/metrics.py)
- [src/attr_rtg_rcmz/models.py](../../src/attr_rtg_rcmz/models.py)
- [src/attr_rtg_rcmz/monitoring.py](../../src/attr_rtg_rcmz/monitoring.py)
- [src/attr_rtg_rcmz/nll.py](../../src/attr_rtg_rcmz/nll.py)
- [src/attr_rtg_rcmz/official.py](../../src/attr_rtg_rcmz/official.py)
- [src/attr_rtg_rcmz/official_contrasts.py](../../src/attr_rtg_rcmz/official_contrasts.py)
- [src/attr_rtg_rcmz/official_data.py](../../src/attr_rtg_rcmz/official_data.py)
- [src/attr_rtg_rcmz/official_isolated.py](../../src/attr_rtg_rcmz/official_isolated.py)
- [src/attr_rtg_rcmz/official_stats.py](../../src/attr_rtg_rcmz/official_stats.py)
- [src/attr_rtg_rcmz/official_supervisor.py](../../src/attr_rtg_rcmz/official_supervisor.py)
- [src/attr_rtg_rcmz/official_training.py](../../src/attr_rtg_rcmz/official_training.py)
- [src/attr_rtg_rcmz/policy.py](../../src/attr_rtg_rcmz/policy.py)
- [src/attr_rtg_rcmz/prelock_evidence.py](../../src/attr_rtg_rcmz/prelock_evidence.py)
- [src/attr_rtg_rcmz/progress.py](../../src/attr_rtg_rcmz/progress.py)
- [src/attr_rtg_rcmz/quantiles.py](../../src/attr_rtg_rcmz/quantiles.py)
- [src/attr_rtg_rcmz/r.py](../../src/attr_rtg_rcmz/r.py)
- [src/attr_rtg_rcmz/rendering/__init__.py](../../src/attr_rtg_rcmz/rendering/__init__.py)
- [src/attr_rtg_rcmz/rendering/artifacts.py](../../src/attr_rtg_rcmz/rendering/artifacts.py)
- [src/attr_rtg_rcmz/rendering/charts.py](../../src/attr_rtg_rcmz/rendering/charts.py)
- [src/attr_rtg_rcmz/rendering/png.py](../../src/attr_rtg_rcmz/rendering/png.py)
- [src/attr_rtg_rcmz/scorer.py](../../src/attr_rtg_rcmz/scorer.py)
- [src/attr_rtg_rcmz/t.py](../../src/attr_rtg_rcmz/t.py)
- [src/attr_rtg_rcmz/validation.py](../../src/attr_rtg_rcmz/validation.py)
- [src/attr_rtg_rcmz/z.py](../../src/attr_rtg_rcmz/z.py)
- [tests/goldens/attr_rtg_rcmz_v1_synthetic.json](../../tests/goldens/attr_rtg_rcmz_v1_synthetic.json)
- [tests/test_attr_rtg_rcmz_cli.py](../../tests/test_attr_rtg_rcmz_cli.py)
- [tests/test_attr_rtg_rcmz_data.py](../../tests/test_attr_rtg_rcmz_data.py)
- [tests/test_attr_rtg_rcmz_engine.py](../../tests/test_attr_rtg_rcmz_engine.py)
- [tests/test_attr_rtg_rcmz_models.py](../../tests/test_attr_rtg_rcmz_models.py)
- [tests/test_attr_rtg_rcmz_official.py](../../tests/test_attr_rtg_rcmz_official.py)
- [tests/test_attr_rtg_rcmz_official_statistics.py](../../tests/test_attr_rtg_rcmz_official_statistics.py)
- [tests/test_attr_rtg_rcmz_scorer_isolation.py](../../tests/test_attr_rtg_rcmz_scorer_isolation.py)
- [tests/test_attr_rtg_rcmz_statistics.py](../../tests/test_attr_rtg_rcmz_statistics.py)
- [tests/test_attr_rtg_rcmz_supervisor.py](../../tests/test_attr_rtg_rcmz_supervisor.py)
- [docs/report/0052_lock-local-attr-rtg-rcmz-v1_2026-09-02.md](0052_lock-local-attr-rtg-rcmz-v1_2026-09-02.md)

## Changes

- Implementados forks reais de candidatos e pareamento natural exato de 50.000 parâmetros.
- Adicionados isolamento do scorer, INVALID rows, RNG normativo, guard de lock ancorado e supervisor real.
- Convertidos os quatro gráficos para barras agrupadas no estilo RTG.
- Criado e verificado o LOCAL PROTOCOL LOCK após aprovação 4/4.

## Validation

- 46 testes ATTR-RTG-RCMZ — passed
- Ruff — passed
- compileall — passed
- git diff --check — passed
- SOLID source audit — zero RCMZ exceptions/violations
- Two structural CUDA runs — identical payload hash
- Canonical lock verification — passed
- Official supervisor --help — passed
