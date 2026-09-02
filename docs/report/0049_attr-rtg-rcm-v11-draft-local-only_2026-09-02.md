# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Criar, com autorização explícita, o V11/DRAFT local-only do ATTR-RTG-RCM, sem nenhum serviço externo ou custo, preservando V10, sem lock ou execução, e submetê-lo a revisão normativa exata.

## Summary

## Resultado

O **ATTR-RTG-RCM V11/DRAFT local-only** foi criado e revisado sem usar serviços externos ou gerar custo.

- Protocol R3 SHA-256: `fb838cc886a019e1b336029d2e42a1befe7271f2e5e43884f75dc78c6d18935c`
- Objection Matrix R3 SHA-256: `4f34eb91e0becdc9c30a0f46bf536c435102f36107533cc896d0ae95eec28d65`
- Normative-ready evidence file SHA-256: `4836f97a04ab8f0f33dfccdbdcda68d0267b49e26b46ea93e93149c93d978b00`
- Evidence canonical content SHA-256: `c66a496fe497241f9f922ddc85edac008df407dd2303b0f198fc26c9954e7c7e`

Estado final:

| Domain | Normative | Factual local | Local lock |
|---|---|---|---|
| Architecture/fairness | READY | OPEN | NO |
| Statistics | READY | OPEN | NO |
| Leakage/custody | READY | OPEN | NO |
| GPU/determinism | READY | OPEN | NO |

Portanto o V11 R3 está **NORMATIVE READY — FACTUAL LOCAL OPEN — NOT LOCKED**. O pedido para criar o DRAFT foi concluído; não houve autorização ou criação de lock.

## Threat model e linguagem

V11 é um plano de benchmark fixo, transparente e prospectivo sob um único administrador. Remove HSM, beacon, ledger/TSA, watchdog independente, WORM, remote attestation, principals externos e clean-room de terceiros do escopo. Esses itens não são tratados como localmente satisfeitos: são explicitamente `EXCLUDED — NOT CLAIMED`.

Resultados futuros, se autorizados, devem começar por `LOCAL-ONLY, SINGLE-ADMINISTRATOR, NOT INDEPENDENTLY ATTESTED` e são condicionais à honest-operator/no-preinspection/no-unreported-attempt assumption. V11 não prova chronology, anti-grinding, secret custody, one-shot, trusted time, host-loss publication ou resistência a root/admin.

## Controles locais definidos

- `seed_basis_object_v1` com allowlist exata, deterministic CBOR RFC8949, definite lengths e rejeição de duplicates/non-shortest encodings;
- domain literal `ATTR-RTG-RCM-V11-LOCAL-ONLY`;
- dependency graph acíclico: preseed manifests → basis bytes/B → seeds/commitments → final lock → signatures/audit;
- seeds transparentes e derivadas do futuro local lock;
- roles separados por processos/UIDs/IPC locais, não principals independentes;
- XChaCha receipts, hash-chain, durable registries, local supervisor, fixed-cadence spool e isolated local execution profile com nonclaims explícitos;
- arquitetura, estimands, folds, bootstrap, maps, gates e CUDA/SASS/peak/elapsed preservados como machinery candidate evidence.

## Factual local ainda OPEN

Antes de pedir `LOCAL READY TO LOCK`:

1. implementar encoder CBOR e derivation/mutation/absence goldens;
2. produzir attempt/replay e exactly-one terminal registry receipts;
3. ligar roles/processes/UIDs/IPC e read-set/syscall/security receipts;
4. criar o environment/closure manifest V11 e fresh-install comparisons locais;
5. executar audit-chain overwrite/replay/crash/quarantine tests;
6. executar spool success/failure/dead-worker/crash/timeout tests;
7. rebindar evidence local ao ambiente exato e obter quatro `LOCAL READY TO LOCK` reviews.

Depois destes gates, uma nova autorização explícita do usuário continua obrigatória.

## Preservação e validação

- V10 SHA `918851086373512855ffc653e292037dd9af5af6649c16d51e58dbe8e4d74422` permaneceu byte-for-byte;
- Evidence V7 SHA `d160335f9efa72fcf61d2b1cc364aa1e1a5bf4ee81eb4af4d676243d3291993b` permaneceu byte-for-byte;
- 190/190 base artifacts verificaram path/size/SHA-256;
- quatro reviews exatas R3 retornaram normativa READY;
- `git diff --check` passou;
- nenhum external service/cost, held-out data, training, calibration, test opening, local lock, implementation snapshot, global freeze, seal, commit ou push ocorreu.

## Modified files

- [docs/ATTR_RTG_RCM_PREREGISTRATION.md](../ATTR_RTG_RCM_PREREGISTRATION.md)
- [docs/review/attr_rtg_rcm_v1/draft_v11_r1_bec11de5a24d9046.md](../review/attr_rtg_rcm_v1/draft_v11_r1_bec11de5a24d9046.md)
- [docs/review/attr_rtg_rcm_v1/draft_v11_r2_79723a69fbc4c7c6.md](../review/attr_rtg_rcm_v1/draft_v11_r2_79723a69fbc4c7c6.md)
- [docs/review/attr_rtg_rcm_v1/draft_v11_r3_fb838cc886a019e1.md](../review/attr_rtg_rcm_v1/draft_v11_r3_fb838cc886a019e1.md)
- [docs/review/attr_rtg_rcm_v1/evidence/prefreeze_evidence_manifest_v10_local_only_draft_r3.json](../review/attr_rtg_rcm_v1/evidence/prefreeze_evidence_manifest_v10_local_only_draft_r3.json)
- [docs/review/attr_rtg_rcm_v1/evidence/prefreeze_evidence_manifest_v11_local_only_normative_ready.json](../review/attr_rtg_rcm_v1/evidence/prefreeze_evidence_manifest_v11_local_only_normative_ready.json)
- [docs/review/attr_rtg_rcm_v1/evidence/prefreeze_evidence_manifest_v8_local_only_draft.json](../review/attr_rtg_rcm_v1/evidence/prefreeze_evidence_manifest_v8_local_only_draft.json)
- [docs/review/attr_rtg_rcm_v1/evidence/prefreeze_evidence_manifest_v9_local_only_draft_r2.json](../review/attr_rtg_rcm_v1/evidence/prefreeze_evidence_manifest_v9_local_only_draft_r2.json)
- [docs/review/attr_rtg_rcm_v1/local_factual_plan_v11.md](../review/attr_rtg_rcm_v1/local_factual_plan_v11.md)
- [docs/review/attr_rtg_rcm_v1/normative_readiness_v11.md](../review/attr_rtg_rcm_v1/normative_readiness_v11.md)
- [docs/review/attr_rtg_rcm_v1/objection_matrix_v11_local_only.md](../review/attr_rtg_rcm_v1/objection_matrix_v11_local_only.md)
- [docs/review/attr_rtg_rcm_v1/objection_matrix_v11_local_only_r2.md](../review/attr_rtg_rcm_v1/objection_matrix_v11_local_only_r2.md)
- [docs/review/attr_rtg_rcm_v1/objection_matrix_v11_local_only_r3.md](../review/attr_rtg_rcm_v1/objection_matrix_v11_local_only_r3.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round1_001_architecture.md](../review/attr_rtg_rcm_v1/review_v11_round1_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round1_002_statistics.md](../review/attr_rtg_rcm_v1/review_v11_round1_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round1_003_leakage.md](../review/attr_rtg_rcm_v1/review_v11_round1_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round1_004_gpu.md](../review/attr_rtg_rcm_v1/review_v11_round1_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round2_001_architecture.md](../review/attr_rtg_rcm_v1/review_v11_round2_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round2_002_statistics.md](../review/attr_rtg_rcm_v1/review_v11_round2_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round2_003_leakage.md](../review/attr_rtg_rcm_v1/review_v11_round2_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round2_004_gpu.md](../review/attr_rtg_rcm_v1/review_v11_round2_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round3_001_architecture.md](../review/attr_rtg_rcm_v1/review_v11_round3_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round3_002_statistics.md](../review/attr_rtg_rcm_v1/review_v11_round3_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round3_003_leakage.md](../review/attr_rtg_rcm_v1/review_v11_round3_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round3_004_gpu.md](../review/attr_rtg_rcm_v1/review_v11_round3_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_v11_round_1_manifest.json](../review/attr_rtg_rcm_v1/review_v11_round_1_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_v11_round_2_manifest.json](../review/attr_rtg_rcm_v1/review_v11_round_2_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_v11_round_3_manifest.json](../review/attr_rtg_rcm_v1/review_v11_round_3_manifest.json)
- [docs/review/attr_rtg_rcm_v1/user_authorization_v11_draft.md](../review/attr_rtg_rcm_v1/user_authorization_v11_draft.md)
- [docs/review/attr_rtg_rcm_v1/v11_local_only_amendment.md](../review/attr_rtg_rcm_v1/v11_local_only_amendment.md)
- [docs/review/attr_rtg_rcm_v1/v11_round1_disposition.md](../review/attr_rtg_rcm_v1/v11_round1_disposition.md)
- [docs/review/attr_rtg_rcm_v1/v11_round2_disposition.md](../review/attr_rtg_rcm_v1/v11_round2_disposition.md)
- [docs/report/0049_attr-rtg-rcm-v11-draft-local-only_2026-09-02.md](0049_attr-rtg-rcm-v11-draft-local-only_2026-09-02.md)

## Changes

- Criado V11 local-only com threat model single-administrator e nonclaims explícitos.
- Definido seed basis/commitments CBOR acíclico e byte-exato.
- Substituídos controles externos por controles locais não equivalentes.
- Criadas matrizes R1–R3, manifests append-only e três rounds de revisão.
- Obtidas quatro aprovações normativas R3; factual local permanece aberto e não locked.

## Validation

- V10 e Evidence V7 recomputados e preservados byte-for-byte
- 190/190 artifacts base verificados por path, size e SHA-256
- Round 3: quatro domínios NORMATIVE READY
- git diff --check — passou
- nenhum serviço externo, custo, dado, treino, calibration, test, lock ou freeze
