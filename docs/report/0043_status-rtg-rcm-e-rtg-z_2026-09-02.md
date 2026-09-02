# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Informar como estão o RTG-RCM e o RTG-Z.

## Summary

## ATTR-RTG-RCM-V1

**Estado:** DRAFT V7, `NORMATIVE READY`, mas `NOT FROZEN`.

Os quatro domínios — arquitetura/fairness, estatística, leakage/custódia e GPU/determinismo — aprovaram normativamente o snapshot exato `1be2ab4e08fe35f9a38c9af61b68aa17c3043168f2eafa68bf297afe93c36541`. A matriz cumulativa preserva 89 objeções. Destas, 18 linhas continuam `OPEN FACTUAL`.

O texto e as regras decisórias estão fechados. O freeze ainda depende de evidência concreta: auditoria graph/active budgets, symbols/schemas/exporters, two-table history-once/six-fork, RNG/batch manifests, maps/normalizers/heads toys, identities/keys/HSM/beacon/TSA/ledger, CUDA goldens/paridade, full-shape peak VRAM, constant-work/WCET, attestation, AEAD/verifier, wire noninterference e watchdog/release toys.

**Consequência:** ainda não pode gerar o estudo oficial, treinar, calibrar ou abrir test. O próximo estágio é implementar e validar o pacote factual pre-freeze; depois, re-review factual e freeze.

## ATTR-RTG-Z-V1

**Estado:** DRAFT V2, `NOT READY TO FREEZE` e `NOT FROZEN`.

O snapshot revisado é `e1eb8013e4dc334ea5aec6721d532fe6fb762f6f8d29d2fa496c3c6fa220f400`. O protocolo já fixa somente ASM-Z versus Transformer novo; configs completos candidatos com contexto 256; seeds propostas `31,47,73,97,113`; `17.024 = 17.024` parâmetros ativos; common16; heads G/D/E/C; claims físicos e de governança; bootstrap pareado; FP32 sem autocast; análise FP64; custódia e DAG pré-truth.

O núcleo ASM-Z e o pareamento de backbones estão implementados e testados. Porém o protocolo ainda tem blockers normativos em todos os domínios:

- arquitetura: regra de `detach` precisa distinguir grafo diferenciável de treino e carrier de inferência depois do commitment;
- estatística: folds/marginais, unsafe predicate, persistence distributions, transporte de calibration, sensitivity de cinco runs e losses/update rules dos heads;
- leakage: root-keyed KDF/attempt binding, prova de freshness, lineage equivalence, principals T3/T4 separados e receipt/release automático;
- GPU: backend/kernel/tolerâncias literais, shared_log, schedule B512→B1024, reduction/storage rules, definição de peak e elapsed-time cap.

Também faltam adapters Z/T, common16 e heads implementados, generator/splits, ceremony, CUDA full-shape, goldens, ACL/attestation, AEAD, noninterference e watchdog.

**Consequência:** ATTR-RTG-Z está atrás do RCM. O próximo passo é DRAFT V3 fechando os blockers textuais, seguido da implementação dos artifacts factuais e nova revisão hostil. Nenhum treino confirmatório foi iniciado.

## Resumo direto

| Protocolo | Texto normativo | Evidência factual | Freeze | Execução |
|---|---|---|---|---|
| RTG-RCM V7 | aprovado nos 4 domínios | 18 blockers factuais | não | proibida |
| RTG-Z V2 | ainda tem blockers normativos | implementação RTG também incompleta | não | proibida |

O RCM está mais próximo do freeze. O Z já tem arquitetura e backbones pareados, mas ainda precisa de uma nova rodada de fechamento do protocolo e da infraestrutura RTG própria.

## Modified files

- [docs/review/attr_rtg_rcm_v1/normative_readiness_v7.md](../review/attr_rtg_rcm_v1/normative_readiness_v7.md)
- [docs/report/0042_asm-z-e-attr-rtg-z_2026-09-02.md](0042_asm-z-e-attr-rtg-z_2026-09-02.md)
- [docs/report/0043_status-rtg-rcm-e-rtg-z_2026-09-02.md](0043_status-rtg-rcm-e-rtg-z_2026-09-02.md)

## Changes

- Verificado o status e os hashes atuais dos dois protocolos.
- Corrigida a contagem documental de linhas OPEN FACTUAL do RCM de 20 para 18.
- Consolidado o próximo passo de cada protocolo.

## Validation

- SHA-256 RCM V7 recomputado — 1be2ab4e08fe35f9a38c9af61b68aa17c3043168f2eafa68bf297afe93c36541
- SHA-256 RTG-Z V2 recomputado — e1eb8013e4dc334ea5aec6721d532fe6fb762f6f8d29d2fa496c3c6fa220f400
- Matriz RCM revisada — 89 objeções, 18 linhas OPEN FACTUAL
