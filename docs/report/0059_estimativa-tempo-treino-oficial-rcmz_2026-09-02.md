# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Estimar em quanto tempo o treino oficial ATTR-RTG-RCMZ deve terminar.

## Summary

Às 2026-09-02 17:26:46 -03, a execução oficial estava em `seed=29`, braço `CM`, update `500/2000`, com aproximadamente 16,7 minutos de tempo total. O braço R terminou 2.000 updates em cerca de 584,5 segundos (9,7 min). O CM levou cerca de 300 segundos para os primeiros 500 updates, projetando aproximadamente 20 minutos para esse braço.

O `eta_seconds` atual do `status.json` não representa o benchmark inteiro: ele usa o progresso do braço corrente e o elapsed global, portanto superestima o restante desse braço e não inclui corretamente os outros 18 braços ainda pendentes.

Estimativa operacional inicial para toda a execução: aproximadamente 4,5 a 6 horas desde o início, com melhor ponto central perto de 5 horas. Como a run começou por volta de 17:10 BRT, a conclusão mais provável fica perto de 22:10 BRT, com faixa prudente entre 21:30 e 23:15. Terminar até 22h é possível, mas ainda não garantido. A estimativa deve ser recalculada quando o primeiro seed completar R, CM, Z e T, pois então haverá tempo real de todos os quatro braços.

A GPU estava ativa em 37%, usando aproximadamente 4.002 MiB, e os processos supervisor/CLI continuavam em execução. Nenhuma intervenção foi feita.

## Modified files

- [docs/report/0059_estimativa-tempo-treino-oficial-rcmz_2026-09-02.md](0059_estimativa-tempo-treino-oficial-rcmz_2026-09-02.md)

## Changes

- Registrada estimativa operacional baseada no progresso real, sem interferir na execução.

## Validation

- status.json read — seed 29, CM, update 500/2000
- run.log read — R completed in 584.5s
- nvidia-smi read — GPU active, 4002 MiB
