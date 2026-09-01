# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Commitar e enviar para origin/main todas as mudanças atuais do repositório ASM.

## Summary

## Commit e push concluídos

Todas as mudanças atuais do repositório ASM foram incluídas em um commit único de conteúdo e enviadas para `origin/main`.

- commit de conteúdo: `ca39f52602dd19b584a5da33590320e651e9c930`;
- mensagem: `feat: add variable-rank models and transition governance benchmarks`;
- conteúdo: 498 paths Git, 166.495 inserções e 36 exclusões;
- tamanho aproximado auditado: 25,2 MB;
- remote: `git@github.com:AletheionAGI/ASM.git`;
- push: `a4a44b3..ca39f52 main -> main`.

Antes do commit, foram verificados limites de tamanho, ausência de arquivos de segredo/chaves, whitespace do diff e arquivos acima do limite do GitHub. O `git diff --cached --check` passou após limpar whitespace residual em SVGs gerados. O remote foi conferido com `git ls-remote` e apontou para o mesmo SHA do commit de conteúdo.

## Modified files

- [docs/report/0038_commit-e-push-repositorio-completo_2026-09-01.md](0038_commit-e-push-repositorio-completo_2026-09-01.md)

## Changes

- Criei um commit com todas as 498 alterações Git efetivas.
- Enviei a branch main para origin/main.
- Verifiquei o SHA remoto e o estado sincronizado da branch.

## Validation

- git diff --cached --check — passou antes do commit
- varredura de nomes de secrets e private keys — nenhum achado
- arquivos staged acima de 20 MB — nenhum
- git push origin main — passou
- git ls-remote origin refs/heads/main — confirmou ca39f52602dd19b584a5da33590320e651e9c930
