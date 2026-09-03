# Request Report

- Status: blocked
- Date: 2026-09-03

## User request

Explicar e corrigir a ausência de safe service em shift e OOD sem violar o protocolo congelado.

## Summary

Você está correto que não há barras de `safe_service` para shift e OOD. Isso não é mais perda de dados pelo renderer: o valor oficial é matematicamente indefinido nesses regimes sob a definição congelada.

`safe_service` é condicionado à existência de pelo menos um candidate seguro. Em shift, world 8/episode 0 não possui origin elegível; em OOD, world 0/episode 0 também não possui. Assim, o denominador é zero. O protocolo estabelece que denominador ausente em qualquer fold torna a célula INVALID e `safe_service=null`; não é permitido imputar zero, um ou uma média dos outros folds.

Há três opções:

1. manter a release oficial correta e fail-closed, com barras vazias;
2. adicionar uma métrica separada, `safe_service_eligible_only`, calculada apenas nos folds elegíveis e rotulada como diagnóstico não oficial; isso exige outro scoring-only, sem treino;
3. redefinir a métrica oficial para ignorar folds sem denominador, o que altera o protocolo congelado e exige novo preregistro/lock. Esta opção não é recomendada para a release atual.

A opção segura para disponibilizar informação visual é a opção 2, mantendo `safe_service` oficial nulo e os gates fail-closed.

## Modified files

- [docs/report/0064_esclarecimento-safe-service-shift-ood_2026-09-03.md](0064_esclarecimento-safe-service-shift-ood_2026-09-03.md)

## Changes

- Esclarecido que safe_service é indefinido por denominador zero, não perdido pelo renderer; proposta métrica diagnóstica elegible-only separada.

## Validation

- Recovered official rows inspected — safe_service finite 20/20 ID and null 20/20 shift/OOD
- Frozen fail-closed denominator rule checked against invalid reasons
