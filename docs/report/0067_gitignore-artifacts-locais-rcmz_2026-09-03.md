# Request Report

- Status: completed
- Date: 2026-09-03

## User request

Adicionar ao .gitignore os arquivos locais que não foram enviados.

## Summary

Os checkpoints e runtimes grandes já estavam cobertos pelas regras específicas de `official/` e `official_failed_*/`. Foi adicionada uma regra explícita para todos os `run.log` mutáveis dentro de `artifacts/attr_rtg_rcmz/**`, incluindo os diretórios score-only cujos demais outputs pequenos estão versionados. A verificação confirmou que checkpoints oficiais, archive da tentativa falha e os dois logs score-only são ignorados. `git ls-files --others --exclude-standard` não retorna nenhum arquivo pendente fora do `.gitignore`.

## Modified files

- [.gitignore](../../.gitignore)
- [docs/report/0067_gitignore-artifacts-locais-rcmz_2026-09-03.md](0067_gitignore-artifacts-locais-rcmz_2026-09-03.md)

## Changes

- Adicionada `/artifacts/attr_rtg_rcmz/**/run.log`.
- Confirmadas as regras existentes para official/ e official_failed_*/.

## Validation

- git check-ignore -v — official checkpoints, failed checkpoints and score-only run logs matched expected rules
- git ls-files --others --exclude-standard — no unignored untracked files
