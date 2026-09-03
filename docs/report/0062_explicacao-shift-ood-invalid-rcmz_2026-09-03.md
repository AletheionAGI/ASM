# Request Report

- Status: completed
- Date: 2026-09-03

## User request

Explicar onde estão os resultados shift e OOD após a execução oficial.

## Summary

Os regimes shift e OOD não foram omitidos do dataset. Eles aparecem na release como `INVALID` e, por isso, o renderer corretamente deixa suas barras vazias.

Causa exata, igual para todos os cinco seeds e quatro arms:

- shift: `metrics: empty origin fold for safe_service at 8/0`;
- OOD: `metrics: empty origin fold for safe_service at 0/0`.

Nesses episódios não existiu origin elegível para o denominador de `safe_service`. O protocolo fail-closed exige que a célula arm/seed/regime seja INVALID quando qualquer fold não possui denominador.

Há também um defeito de retenção: quando `safe_service` falhou, `summarize()` abortou a célula e o orchestrator gravou todas as seis métricas como `null`, embora H8 NLL, ECE, unsafe-selection, coverage e abstention já pudessem ser calculadas. Isso explica por que nenhuma dessas barras aparece para shift/OOD. Os 20 checkpoints e os worlds permanecem disponíveis.

A correção possível não exige treino: tornar a sumarização independente por métrica, manter a célula fail-closed/INVALID, preservar `safe_service=null`, recalcular scoring a partir dos 20 checkpoints e publicar as outras métricas finitas. O status INVALID não deve ser removido e os gates dependentes continuam falhando. O scoring observado no fim da execução sugere poucos minutos, mas deve ser executado como uma recuperação explícita, não silenciosamente.

## Modified files

- [docs/report/0062_explicacao-shift-ood-invalid-rcmz_2026-09-03.md](0062_explicacao-shift-ood-invalid-rcmz_2026-09-03.md)

## Changes

- Diagnosticada a ausência visual de shift/OOD como fail-closed por denominador vazio de safe_service e perda indevida das métricas parciais.

## Validation

- official_rows.json inspected — 20 VALID ID, 20 INVALID shift, 20 INVALID OOD, 6 INVALID contrasts
- Invalid reasons counted — shift world 8/episode 0; OOD world 0/episode 0
