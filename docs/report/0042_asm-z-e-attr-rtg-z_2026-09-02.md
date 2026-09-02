# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Desenvolver em paralelo o ASM-Z conforme a equação zero-choice fornecida e a figura ASM-X versus ASM-Z; criar também o protocolo ATTR-RTG-Z. O usuário esclareceu que o protocolo deve comparar ASM-Z versus um Transformer novo e pareado, não ASM-X versus ASM-Z.

## Summary

## ASM-Z implementado

O ASM-Z foi implementado como uma recorrência zero-choice estrita:

```text
Phi(z,e) = phi_theta(z,e) + (lambda/2)||z||²
g = grad_z Phi(z,e)
G(z,e) v = g
z_next = z - eta v
```

A implementação usa exatamente um potencial escalar, uma métrica condicionada SPD e um solve por input. Não há catálogo de direções, `V_i`, `a_i`, `c_i`, soma de candidatos internos, gate, trust scalar, attention no caminho de estado, memória lateral ou bypass. `eta` é constante. `G=diag(d)+UUᵀ` tem diagonal limitada e fator low-rank com norma de Frobenius limitada. O runtime resolve o sistema pela identidade de Woodbury, sem formar a inversa explícita. O treino preserva os gradientes de ordem superior com `create_graph=True`; a inferência mantém somente o estado recorrente compacto.

A motivação filosófica foi preservada apenas como metáfora: a arquitetura representa um fluxo determinado pelas condições locais. Ela não prova ausência de livre-arbítrio, consciência, causalidade, safety ou superioridade.

Arquivos centrais: `src/drm_language_emitter/asm_z_core.py`, `asm_z_forward.py`, `asm_z.py`, builder `variants/zero_choice.py`, config inicial `configs/asm_z_tiny.yaml` e testes `tests/test_asm_z.py`.

## ASM-Z versus Transformer pareado

Foram criados cinco pares candidatos, com run-index seeds novos e provisórios `31,47,73,97,113`. Todos usam vocabulário 64 e contexto 256.

| Braço | Configuração | Parâmetros ativos | Tensors ativos |
|---|---|---:|---:|
| ASM-Z | state 24, token 16, rank 4, hidden 54 | 17.024 | 14 |
| Transformer | d16, 4 heads, 2 layers, FFN 128 | 17.024 | 30 |

O mismatch é exatamente zero. Os dez YAMLs são completos: 161 campos ASM-Z e 10 campos Transformer. Backward em length 65 alcançou todos os tensors com gradientes finitos e não nulos. Forward em length 256 foi finito com logits `(1,256,64)`. Isso é parameter-and-update matching candidato, não compute matching: a recorrência com derivadas de ordem superior e a atenção têm custos diferentes, que deverão ser publicados separadamente.

## Documentação e figura

`docs/ASM_Z_ARCHITECTURE.md` explica em linguagem humana a diferença ASM-X versus ASM-Z. A figura autoral está em SVG e PNG 1200×720. O lado Z mostra um campo/potencial, uma métrica, um gradiente, um solve e uma atualização, sem catálogo de direções.

## ATTR-RTG-Z

Foi criado e submetido a duas rodadas hostis o `DRAFT V2 — NOT FROZEN` em `docs/ATTR_RTG_Z_PREREGISTRATION.md`. O estudo compara somente ASM-Z e Transformer novo. ASM-X não é braço.

O draft define `common16`, physical truth comum de 485 logits/11 grupos, heads G/D/E/C pareados, calibration/decisão, estimando condicional, bootstrap Bayesiano hierárquico, duas branches arquiteturais (`RTG1-Y` físico e `RTG2-G` governança) com Bonferroni `m=2`, RTG3 ID∩shift∩OOD, FP32 sem autocast e análise FP64, KDF/custódia, causal forks, compromissos pré-truth e release atômico.

A revisão V2 concluiu `NOT READY TO FREEZE`. Blockers normativos ainda incluem: escopo correto de `detach`; fórmulas literais de folds/marginais/unsafe/persistência e losses dos heads; root-keyed KDF e attempt binding; lineage; separação T3/T4; receipt plaintext/release automático; backend/kernel/tolerâncias CUDA; `shared_log`; scheduling B512→B1024; peak e elapsed-time contracts. Adapters, common16, heads, generator, ceremony, CUDA full-shape, goldens, attestation e watchdog continuam factualmente ausentes. Nenhum dado foi gerado e nenhum treino/calibration/test foi executado.

## ATTR-RTG-RCM em paralelo

O DRAFT V7 recebeu `NORMATIVE READY` dos quatro domínios no snapshot SHA-256 `1be2ab4e08fe35f9a38c9af61b68aa17c3043168f2eafa68bf297afe93c36541`. A matriz tem 89 objeções; 18 continuam `OPEN FACTUAL`. Portanto o RCM permanece `NOT FROZEN`: identities/keys, generator closure, goldens, two-table/fork proofs, CUDA parity, peak VRAM, WCET, attestation, AEAD e watchdog ainda precisam existir e passar.

## Validação

- suíte ampla com `--ignore=tests/test_rtg_source_inventory.py`: passou;
- suíte final ASM-Z/pareamento/compatibilidade: 32 testes passaram;
- `tests/test_rtg_z_configs.py`: passou;
- `ruff check` nos módulos/testes novos ASM-Z: passou;
- `compileall`: passou;
- `git diff --check`: passou;
- audit SOLID: nenhum arquivo novo viola 300/500 linhas; `model.py` permanece exceção coesa preexistente de 437 linhas;
- SVG parseado; PNG 1200×720 renderizado e revisado visualmente;
- o teste isolado do inventário histórico ATTR-RTG falhou em 2 checks, como esperado, porque novos arquivos RCM/ASM-Z não pertencem ao inventário literal já selado. O inventário histórico não foi alterado.

## Modified files

- [src/aletheion_state_models/benchmarks/transition_risk/rtg_config.py](../../src/aletheion_state_models/benchmarks/transition_risk/rtg_config.py)
- [src/aletheion_state_models/variants/__init__.py](../../src/aletheion_state_models/variants/__init__.py)
- [src/drm_language_emitter/__init__.py](../../src/drm_language_emitter/__init__.py)
- [src/drm_language_emitter/config.py](../../src/drm_language_emitter/config.py)
- [src/drm_language_emitter/inference.py](../../src/drm_language_emitter/inference.py)
- [src/drm_language_emitter/model.py](../../src/drm_language_emitter/model.py)
- [tests/test_rtg_config.py](../../tests/test_rtg_config.py)
- [configs/asm_z_tiny.yaml](../../configs/asm_z_tiny.yaml)
- [configs/rtg_rcm_asm_cm_30k_seed107.yaml](../../configs/rtg_rcm_asm_cm_30k_seed107.yaml)
- [configs/rtg_rcm_asm_cm_30k_seed29.yaml](../../configs/rtg_rcm_asm_cm_30k_seed29.yaml)
- [configs/rtg_rcm_asm_cm_30k_seed43.yaml](../../configs/rtg_rcm_asm_cm_30k_seed43.yaml)
- [configs/rtg_rcm_asm_cm_30k_seed71.yaml](../../configs/rtg_rcm_asm_cm_30k_seed71.yaml)
- [configs/rtg_rcm_asm_cm_30k_seed89.yaml](../../configs/rtg_rcm_asm_cm_30k_seed89.yaml)
- [configs/rtg_rcm_asm_r_30k_seed107.yaml](../../configs/rtg_rcm_asm_r_30k_seed107.yaml)
- [configs/rtg_rcm_asm_r_30k_seed29.yaml](../../configs/rtg_rcm_asm_r_30k_seed29.yaml)
- [configs/rtg_rcm_asm_r_30k_seed43.yaml](../../configs/rtg_rcm_asm_r_30k_seed43.yaml)
- [configs/rtg_rcm_asm_r_30k_seed71.yaml](../../configs/rtg_rcm_asm_r_30k_seed71.yaml)
- [configs/rtg_rcm_asm_r_30k_seed89.yaml](../../configs/rtg_rcm_asm_r_30k_seed89.yaml)
- [configs/rtg_z_asm_z_17k_seed113.yaml](../../configs/rtg_z_asm_z_17k_seed113.yaml)
- [configs/rtg_z_asm_z_17k_seed31.yaml](../../configs/rtg_z_asm_z_17k_seed31.yaml)
- [configs/rtg_z_asm_z_17k_seed47.yaml](../../configs/rtg_z_asm_z_17k_seed47.yaml)
- [configs/rtg_z_asm_z_17k_seed73.yaml](../../configs/rtg_z_asm_z_17k_seed73.yaml)
- [configs/rtg_z_asm_z_17k_seed97.yaml](../../configs/rtg_z_asm_z_17k_seed97.yaml)
- [docs/ASM_Z_ARCHITECTURE.md](../ASM_Z_ARCHITECTURE.md)
- [docs/ATTR_RTG_RCM_PREREGISTRATION.md](../ATTR_RTG_RCM_PREREGISTRATION.md)
- [docs/ATTR_RTG_Z_PREREGISTRATION.md](../ATTR_RTG_Z_PREREGISTRATION.md)
- [docs/figures/asm_z_zero_choice_flow.png](../figures/asm_z_zero_choice_flow.png)
- [docs/figures/asm_z_zero_choice_flow.svg](../figures/asm_z_zero_choice_flow.svg)
- [docs/review/attr_rtg_rcm_v1/draft_v1_bf9146eb6e4c415a.md](../review/attr_rtg_rcm_v1/draft_v1_bf9146eb6e4c415a.md)
- [docs/review/attr_rtg_rcm_v1/draft_v2_f390e13db798c042.md](../review/attr_rtg_rcm_v1/draft_v2_f390e13db798c042.md)
- [docs/review/attr_rtg_rcm_v1/draft_v3_5d88f79ad295c42e.md](../review/attr_rtg_rcm_v1/draft_v3_5d88f79ad295c42e.md)
- [docs/review/attr_rtg_rcm_v1/draft_v4_373e6e223c45c9fb.md](../review/attr_rtg_rcm_v1/draft_v4_373e6e223c45c9fb.md)
- [docs/review/attr_rtg_rcm_v1/draft_v5_002c8ebff55f8abd.md](../review/attr_rtg_rcm_v1/draft_v5_002c8ebff55f8abd.md)
- [docs/review/attr_rtg_rcm_v1/draft_v6_9bf25457797ec1a2.md](../review/attr_rtg_rcm_v1/draft_v6_9bf25457797ec1a2.md)
- [docs/review/attr_rtg_rcm_v1/draft_v7_1be2ab4e08fe35f9.md](../review/attr_rtg_rcm_v1/draft_v7_1be2ab4e08fe35f9.md)
- [docs/review/attr_rtg_rcm_v1/normative_readiness_v7.md](../review/attr_rtg_rcm_v1/normative_readiness_v7.md)
- [docs/review/attr_rtg_rcm_v1/objection_disposition_matrix.md](../review/attr_rtg_rcm_v1/objection_disposition_matrix.md)
- [docs/review/attr_rtg_rcm_v1/objection_disposition_matrix_v2.md](../review/attr_rtg_rcm_v1/objection_disposition_matrix_v2.md)
- [docs/review/attr_rtg_rcm_v1/objection_disposition_matrix_v3.md](../review/attr_rtg_rcm_v1/objection_disposition_matrix_v3.md)
- [docs/review/attr_rtg_rcm_v1/objection_disposition_matrix_v4.md](../review/attr_rtg_rcm_v1/objection_disposition_matrix_v4.md)
- [docs/review/attr_rtg_rcm_v1/objection_disposition_matrix_v5.md](../review/attr_rtg_rcm_v1/objection_disposition_matrix_v5.md)
- [docs/review/attr_rtg_rcm_v1/objection_disposition_matrix_v6.md](../review/attr_rtg_rcm_v1/objection_disposition_matrix_v6.md)
- [docs/review/attr_rtg_rcm_v1/objection_disposition_matrix_v7.md](../review/attr_rtg_rcm_v1/objection_disposition_matrix_v7.md)
- [docs/review/attr_rtg_rcm_v1/objection_disposition_matrix_v7_reviewed.md](../review/attr_rtg_rcm_v1/objection_disposition_matrix_v7_reviewed.md)
- [docs/review/attr_rtg_rcm_v1/review_001_architecture.md](../review/attr_rtg_rcm_v1/review_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_002_statistics.md](../review/attr_rtg_rcm_v1/review_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_002_statistics_corrective_sap.md](../review/attr_rtg_rcm_v1/review_002_statistics_corrective_sap.md)
- [docs/review/attr_rtg_rcm_v1/review_002_statistics_corrective_sap_manifest.json](../review/attr_rtg_rcm_v1/review_002_statistics_corrective_sap_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_003_leakage.md](../review/attr_rtg_rcm_v1/review_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_004_gpu.md](../review/attr_rtg_rcm_v1/review_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_001_architecture.md](../review/attr_rtg_rcm_v1/review_round2_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_002_statistics.md](../review/attr_rtg_rcm_v1/review_round2_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_003_leakage.md](../review/attr_rtg_rcm_v1/review_round2_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_004_gpu.md](../review/attr_rtg_rcm_v1/review_round2_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_adendum_001_architecture_joint.md](../review/attr_rtg_rcm_v1/review_round2_adendum_001_architecture_joint.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_adendum_001_architecture_joint_followup.md](../review/attr_rtg_rcm_v1/review_round2_adendum_001_architecture_joint_followup.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_adendum_002_statistics_joint.md](../review/attr_rtg_rcm_v1/review_round2_adendum_002_statistics_joint.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_adendum_002_statistics_joint_followup.md](../review/attr_rtg_rcm_v1/review_round2_adendum_002_statistics_joint_followup.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_adendum_003_leakage_joint.md](../review/attr_rtg_rcm_v1/review_round2_adendum_003_leakage_joint.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_adendum_004_gpu_joint.md](../review/attr_rtg_rcm_v1/review_round2_adendum_004_gpu_joint.md)
- [docs/review/attr_rtg_rcm_v1/review_round2_adendum_004_gpu_joint_followup.md](../review/attr_rtg_rcm_v1/review_round2_adendum_004_gpu_joint_followup.md)
- [docs/review/attr_rtg_rcm_v1/review_round3_001_architecture.md](../review/attr_rtg_rcm_v1/review_round3_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_round3_002_statistics.md](../review/attr_rtg_rcm_v1/review_round3_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_round3_003_leakage.md](../review/attr_rtg_rcm_v1/review_round3_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_round3_004_gpu.md](../review/attr_rtg_rcm_v1/review_round3_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_round4_001_architecture.md](../review/attr_rtg_rcm_v1/review_round4_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_round4_002_statistics.md](../review/attr_rtg_rcm_v1/review_round4_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_round4_003_leakage.md](../review/attr_rtg_rcm_v1/review_round4_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_round4_004_gpu.md](../review/attr_rtg_rcm_v1/review_round4_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_round5_001_architecture.md](../review/attr_rtg_rcm_v1/review_round5_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_round5_002_statistics.md](../review/attr_rtg_rcm_v1/review_round5_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_round5_003_leakage.md](../review/attr_rtg_rcm_v1/review_round5_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_round5_004_gpu.md](../review/attr_rtg_rcm_v1/review_round5_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_round6_001_architecture.md](../review/attr_rtg_rcm_v1/review_round6_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_round6_002_statistics.md](../review/attr_rtg_rcm_v1/review_round6_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_round6_003_leakage.md](../review/attr_rtg_rcm_v1/review_round6_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_round6_004_gpu.md](../review/attr_rtg_rcm_v1/review_round6_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_round7_001_architecture.md](../review/attr_rtg_rcm_v1/review_round7_001_architecture.md)
- [docs/review/attr_rtg_rcm_v1/review_round7_002_statistics.md](../review/attr_rtg_rcm_v1/review_round7_002_statistics.md)
- [docs/review/attr_rtg_rcm_v1/review_round7_003_leakage.md](../review/attr_rtg_rcm_v1/review_round7_003_leakage.md)
- [docs/review/attr_rtg_rcm_v1/review_round7_004_gpu.md](../review/attr_rtg_rcm_v1/review_round7_004_gpu.md)
- [docs/review/attr_rtg_rcm_v1/review_round_1_manifest.json](../review/attr_rtg_rcm_v1/review_round_1_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_round_2_adenda_manifest.json](../review/attr_rtg_rcm_v1/review_round_2_adenda_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_round_2_manifest.json](../review/attr_rtg_rcm_v1/review_round_2_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_round_3_manifest.json](../review/attr_rtg_rcm_v1/review_round_3_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_round_4_manifest.json](../review/attr_rtg_rcm_v1/review_round_4_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_round_5_manifest.json](../review/attr_rtg_rcm_v1/review_round_5_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_round_6_manifest.json](../review/attr_rtg_rcm_v1/review_round_6_manifest.json)
- [docs/review/attr_rtg_rcm_v1/review_round_7_manifest.json](../review/attr_rtg_rcm_v1/review_round_7_manifest.json)
- [docs/review/attr_rtg_z_v1/README.md](../review/attr_rtg_z_v1/README.md)
- [docs/review/attr_rtg_z_v1/draft_v1_eeb173de5da67cb5.md](../review/attr_rtg_z_v1/draft_v1_eeb173de5da67cb5.md)
- [docs/review/attr_rtg_z_v1/draft_v2_e1eb8013e4dc334e.md](../review/attr_rtg_z_v1/draft_v2_e1eb8013e4dc334e.md)
- [docs/review/attr_rtg_z_v1/objection_disposition_matrix.md](../review/attr_rtg_z_v1/objection_disposition_matrix.md)
- [docs/review/attr_rtg_z_v1/parameter_match_candidate_v1.md](../review/attr_rtg_z_v1/parameter_match_candidate_v1.md)
- [docs/review/attr_rtg_z_v1/review_round1_001_architecture.md](../review/attr_rtg_z_v1/review_round1_001_architecture.md)
- [docs/review/attr_rtg_z_v1/review_round1_002_statistics.md](../review/attr_rtg_z_v1/review_round1_002_statistics.md)
- [docs/review/attr_rtg_z_v1/review_round1_003_gpu.md](../review/attr_rtg_z_v1/review_round1_003_gpu.md)
- [docs/review/attr_rtg_z_v1/review_round1_004_leakage.md](../review/attr_rtg_z_v1/review_round1_004_leakage.md)
- [docs/review/attr_rtg_z_v1/review_round2_001_architecture.md](../review/attr_rtg_z_v1/review_round2_001_architecture.md)
- [docs/review/attr_rtg_z_v1/review_round2_002_statistics.md](../review/attr_rtg_z_v1/review_round2_002_statistics.md)
- [docs/review/attr_rtg_z_v1/review_round2_003_gpu.md](../review/attr_rtg_z_v1/review_round2_003_gpu.md)
- [docs/review/attr_rtg_z_v1/review_round2_004_leakage.md](../review/attr_rtg_z_v1/review_round2_004_leakage.md)
- [docs/review/attr_rtg_z_v1/review_round_1_manifest.json](../review/attr_rtg_z_v1/review_round_1_manifest.json)
- [docs/review/attr_rtg_z_v1/review_round_2_manifest.json](../review/attr_rtg_z_v1/review_round_2_manifest.json)
- [docs/review/attr_rtg_z_v1/round2_disposition_summary.md](../review/attr_rtg_z_v1/round2_disposition_summary.md)
- [src/aletheion_state_models/variants/zero_choice.py](../../src/aletheion_state_models/variants/zero_choice.py)
- [src/drm_language_emitter/asm_z.py](../../src/drm_language_emitter/asm_z.py)
- [src/drm_language_emitter/asm_z_core.py](../../src/drm_language_emitter/asm_z_core.py)
- [src/drm_language_emitter/asm_z_forward.py](../../src/drm_language_emitter/asm_z_forward.py)
- [src/drm_language_emitter/config_validation_modes.py](../../src/drm_language_emitter/config_validation_modes.py)
- [src/drm_language_emitter/config_validation_numeric.py](../../src/drm_language_emitter/config_validation_numeric.py)
- [tests/test_asm_z.py](../../tests/test_asm_z.py)
- [tests/test_rtg_z_configs.py](../../tests/test_rtg_z_configs.py)
- [transformer/rtg_rcm_transformer_30k_seed107.yaml](../../transformer/rtg_rcm_transformer_30k_seed107.yaml)
- [transformer/rtg_rcm_transformer_30k_seed29.yaml](../../transformer/rtg_rcm_transformer_30k_seed29.yaml)
- [transformer/rtg_rcm_transformer_30k_seed43.yaml](../../transformer/rtg_rcm_transformer_30k_seed43.yaml)
- [transformer/rtg_rcm_transformer_30k_seed71.yaml](../../transformer/rtg_rcm_transformer_30k_seed71.yaml)
- [transformer/rtg_rcm_transformer_30k_seed89.yaml](../../transformer/rtg_rcm_transformer_30k_seed89.yaml)
- [transformer/rtg_z_transformer_17k_seed113.yaml](../../transformer/rtg_z_transformer_17k_seed113.yaml)
- [transformer/rtg_z_transformer_17k_seed31.yaml](../../transformer/rtg_z_transformer_17k_seed31.yaml)
- [transformer/rtg_z_transformer_17k_seed47.yaml](../../transformer/rtg_z_transformer_17k_seed47.yaml)
- [transformer/rtg_z_transformer_17k_seed73.yaml](../../transformer/rtg_z_transformer_17k_seed73.yaml)
- [transformer/rtg_z_transformer_17k_seed97.yaml](../../transformer/rtg_z_transformer_17k_seed97.yaml)
- [docs/report/0042_asm-z-e-attr-rtg-z_2026-09-02.md](0042_asm-z-e-attr-rtg-z_2026-09-02.md)

## Changes

- Implementado o núcleo, forward, streaming, configuração, builder e testes strict zero-choice do ASM-Z.
- Criados dez configs completos ASM-Z/Transformer com contexto 256 e 17.024 parâmetros ativos exatos por braço.
- Criadas documentação e figura SVG/PNG ASM-X versus ASM-Z.
- Criado, revisado hostilmente e mantido NOT FROZEN o DRAFT V2 ATTR-RTG-Z.
- Concluída a prontidão normativa V7 do ATTR-RTG-RCM, mantendo blockers factuais e NOT FROZEN.

## Validation

- pytest -q --ignore=tests/test_rtg_source_inventory.py — passou
- pytest focused ASM-Z/RTG-Z/RTG compatibility — 32 passaram
- pytest -q tests/test_rtg_z_configs.py — passou
- ruff check nos módulos/testes novos ASM-Z — passou
- python -m compileall -q src transformer — passou
- git diff --check — passou
- solid-source-modularity audit — novos arquivos conformes; quatro violações >500 preexistentes
- SVG XML + render PNG 1200x720 + revisão visual — passaram
- pytest -q tests/test_rtg_source_inventory.py — 2 falhas esperadas; inventário histórico selado não foi alterado
