# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Explicar por que o Markdown do protocolo ainda aparece como DRAFT após o lock.

## Summary

O Markdown permanece com o texto `DRAFT` porque ele faz parte dos 106 artifacts selados pelo candidate manifest `282a7cc6fd545948a42934125d2b4d5b289414239c14bfab6f2f4f06d9e654c9` e pelo lock `f3d772784742307fe9939000f29e5b0dd09e7f0d82861df4897d0dd3bbb5c661`. Alterá-lo depois do lock mudaria o SHA-256 do protocolo, invalidaria o candidate manifest, o source anchor e o `LOCAL PROTOCOL LOCK`.

O status pós-lock autoritativo foi registrado separadamente em `locks/attr_rtg_rcmz_v1/status.json` como `LOCAL PROTOCOL LOCK`. Portanto, o texto `DRAFT` no artifact identifica o estado do snapshot submetido à revisão, não o estado operacional atual.

Para mudar também o próprio Markdown com integridade criptográfica, seria necessário criar uma nova versão do protocolo, gerar novo manifest e obter novamente a aprovação 4/4 antes de substituir o lock. Uma alternativa sem invalidar o lock é criar uma cópia pós-lock claramente marcada como `LOCKED`, mantendo o preregistration original imutável.

## Modified files

- [docs/report/0054_explicacao-status-draft-pos-lock_2026-09-02.md](0054_explicacao-status-draft-pos-lock_2026-09-02.md)

## Changes

- Nenhum artifact selado foi alterado; foi documentada a razão de integridade para preservar o status histórico no Markdown.

## Validation

- None recorded.
