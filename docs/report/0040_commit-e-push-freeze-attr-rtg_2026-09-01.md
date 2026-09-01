# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Commitar e enviar para origin/main todas as mudanças atuais do congelamento ATTR-RTG.

## Summary

## Commit e push concluídos

Todas as mudanças pendentes do congelamento ATTR-RTG foram commitadas e enviadas para `origin/main`.

- commit de conteúdo: `c8af4719fb55ead6d8cd5e7c04446ed98f516fdc`;
- mensagem: `docs: freeze ATTR-RTG preregistration`;
- branch: `main`;
- remote: `origin` (`git@github.com:AletheionAGI/ASM.git`);
- alteração: 15 paths, 1.085 inserções e 103 exclusões;
- preregistration hash preservado: `4db4e22029431c6544a3d8c032cd75d45308250b3b601c0fe437ace4c51ee7f0`.

O commit inclui o protocolo em status `FROZEN PREREGISTRATION`, dez YAMLs completos por seed, manifesto/hash do preregistration e o relatório técnico 0039. Antes do commit, o manifesto foi verificado com `sha256sum -c`, `git diff --cached --check` passou e nenhuma chave/arquivo de segredo foi detectado.

## Modified files

- [configs/rtg_asm_30k.yaml](../../configs/rtg_asm_30k.yaml)
- [configs/rtg_asm_30k_seed107.yaml](../../configs/rtg_asm_30k_seed107.yaml)
- [configs/rtg_asm_30k_seed29.yaml](../../configs/rtg_asm_30k_seed29.yaml)
- [configs/rtg_asm_30k_seed43.yaml](../../configs/rtg_asm_30k_seed43.yaml)
- [configs/rtg_asm_30k_seed71.yaml](../../configs/rtg_asm_30k_seed71.yaml)
- [configs/rtg_asm_30k_seed89.yaml](../../configs/rtg_asm_30k_seed89.yaml)
- [docs/ATTR_RTG_PREREGISTRATION.md](../ATTR_RTG_PREREGISTRATION.md)
- [docs/ATTR_RTG_PREREGISTRATION_MANIFEST.json](../ATTR_RTG_PREREGISTRATION_MANIFEST.json)
- [docs/ATTR_RTG_PREREGISTRATION_MANIFEST.sha256](../ATTR_RTG_PREREGISTRATION_MANIFEST.sha256)
- [docs/report/0039_congelamento-preregistration-attr-rtg_2026-09-01.md](0039_congelamento-preregistration-attr-rtg_2026-09-01.md)
- [transformer/rtg_transformer_30k_seed107.yaml](../../transformer/rtg_transformer_30k_seed107.yaml)
- [transformer/rtg_transformer_30k_seed29.yaml](../../transformer/rtg_transformer_30k_seed29.yaml)
- [transformer/rtg_transformer_30k_seed43.yaml](../../transformer/rtg_transformer_30k_seed43.yaml)
- [transformer/rtg_transformer_30k_seed71.yaml](../../transformer/rtg_transformer_30k_seed71.yaml)
- [transformer/rtg_transformer_30k_seed89.yaml](../../transformer/rtg_transformer_30k_seed89.yaml)
- [docs/report/0040_commit-e-push-freeze-attr-rtg_2026-09-01.md](0040_commit-e-push-freeze-attr-rtg_2026-09-01.md)

## Changes

- Criei o commit de conteúdo c8af4719fb55ead6d8cd5e7c04446ed98f516fdc.
- Enviei a branch main para origin/main.
- Preservei e verifiquei o PREREGISTRATION HASH.

## Validation

- sha256sum -c docs/ATTR_RTG_PREREGISTRATION_MANIFEST.sha256 — passou
- git diff --cached --check — passou
- varredura de arquivos/chaves de segredo — nenhum achado
- git push origin main — passou
