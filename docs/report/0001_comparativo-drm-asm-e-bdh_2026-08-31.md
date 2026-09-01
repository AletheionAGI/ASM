# Request Report

- Status: completed
- Date: 2026-08-31

## User request

Estudar o repositório ASM e, em paralelo, docs/paper/drm_v7.tex e o paper disponível em docs/paper/others/2509.26507v1.pdf, e produzir um comparativo entre os dois modelos.

## Summary

Foi produzido um comparativo técnico em português que separa DRM v7 como framework geométrico, BDH/BDH-GPU como arquitetura de linguagem e ASM como implementação parcial derivada do DRM. O estudo cobre arquitetura, estado, memória, atenção, rank, evidência experimental, limitações, sinergias e um protocolo recomendado de comparação direta.

## Modified files

- [docs/comparativo_drm_bdh.md](../comparativo_drm_bdh.md)
- [docs/report/0001_comparativo-drm-asm-e-bdh_2026-08-31.md](0001_comparativo-drm-asm-e-bdh_2026-08-31.md)

## Changes

- Adicionado comparativo técnico DRM/ASM versus BDH com tabela por eixos e veredito por nível de abstração.
- Documentadas as diferenças entre rank geométrico, sparsity e gate activity.
- Contrastados memória por holonomia, estado sináptico BDH e fast weights do ASM-CM.
- Registradas cautelas sobre BDH-GPU vanilla versus BDH-GPU com gating, benchmarks atuais e ausência de comparação direta.
- Proposto experimento controlado e uma ponte de pesquisa em que DRM analisa a dinâmica concreta do BDH.

## Validation

- git diff --check — passou.
- Validação local dos 12 links Markdown — todos os destinos existem.
- Verificação das seções obrigatórias e das afirmações-chave contra os papers e artefatos versionados — passou.
