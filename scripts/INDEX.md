# Scripts index

Last updated: 2026-07-30

This index preserves the existing filenames and records their chronological
introduction into Git. It is safer than numeric filename prefixes because
launchers, reports and external commands can continue to use stable paths.

## Ordering rule

Scripts are grouped by the first commit in which Git records the current path.
Groups are ordered by commit author date. Files introduced in the same commit
are ordered alphabetically because Git does not retain a reliable creation
order within one commit.

Renames and files that existed before their current path may make this a
repository chronology rather than an operating-system creation chronology.

## Current independent 125M workflow

These are the primary entry points for the active frozen benchmark:

1. `download_pg19_test.py` — download and verify the official PG-19 test set.
2. `prepare_wikipedia_document_split.py` — create document-disjoint Wikipedia
   train and validation manifests.
3. `prepare_independent_benchmark.py` — prepare external benchmark manifests.
4. `audit_dataset_contamination.py` — audit exact block overlap.
5. `run_independent_125m_smoke.sh` — verify CUDA and both 125M model forwards.
6. `run_independent_125m_benchmark.sh` — run three DRM and three GPT-2 seeds.
7. `evaluate_frozen_test.py` — evaluate selected checkpoints on the frozen
   external test set.

The active training launcher must not be renamed while a run is in progress.

## Chronology

### 001 — 2026-06-17 — initial language-emitter and competition suite

Commit: `7f374d2`

1. `compare_drm_transformer.py`
2. `eval_ablations.py`
3. `eval_bridge_task.py`
4. `eval_geodesic_paths.py`
5. `eval_geometry.py`
6. `eval_robustness.py`
7. `eval_sequence_stability.py`
8. `generate.py`
9. `make_competition_dashboard.py`
10. `profile_drm.py`
11. `run_full_trainings.py`
12. `summarize_competition.py`
13. `summarize_runs.py`
14. `sweep_drm_transformer.py`
15. `time_matched_competition.py`
16. `train_tiny.py`

### 002 — 2026-06-18 — symbolic world-model benchmark

Commit: `ec4dab5`

1. `make_tiny_world_dataset.py`
2. `make_world_model_dashboard.py`
3. `sweep_world_model_competition.py`
4. `train_tiny_world_model.py`

### 003 — 2026-07-11 — scaled language-model pipeline

Commit: `0b9caa3`

1. `prepare_wikipedia_en.py`
2. `run_scale_lm_comparison.py`

### 004 — 2026-07-11 — matched Wikipedia benchmarks

Commit: `415cdab`

1. `run_wiki_en_125m_matched.ps1`
2. `run_wiki_en_125m_real_matched.ps1`

### 005 — 2026-07-13 — real 125M benchmark tooling

Commit: `65196b4`

1. `chat_drm_125m_real.py`
2. `chat_gpt2_125m_real.py`
3. `prepare_wikipedia_en_125m_real.ps1`
4. `prepare_wikipedia_en_5b.ps1`

### 006 — 2026-07-22 — RTX 4090 and large-scale training

Commit: `b3ae9f2`

1. `chat_drm_125m_4090.ps1`
2. `chat_drm_125m_4090_base.py`
3. `chat_drm_tta_demo.ps1`
4. `chat_gpt2_125m_4090.ps1`
5. `chat_gpt2_125m_4090_base.py`
6. `run_drm_125m_4090_base.ps1`
7. `run_drm_500m_5b.ps1`
8. `run_gpt2_125m_4090_base.ps1`
9. `tokenize_corpus_to_uint8.py`
10. `train_drm_memmap.py`
11. `train_gpt2_memmap.py`

### 007 — 2026-07-26 — causal blockwise Anderson benchmark

Commit: `ed5befa`

1. `analyze_time_to_quality.py`
2. `run_10m_seed1_drm_vs_gpt2.ps1`
3. `run_time_to_quality.ps1`

### 008 — 2026-07-26 — multiseed TTA confirmation

Commit: `1cd0b81`

1. `run_tta_multiseed_confirmation.ps1`

### 009 — 2026-07-29 — causal local-mixer validation

Commit: `ada18ff`

1. `check_125m_local_mixer_causality.py`
2. `profile_125m_b8_anderson.py`
3. `profile_125m_drm_throughput_sweep.ps1`
4. `run_125m_150m_multiseed_competition.ps1`
5. `run_125m_curriculum_probe.ps1`
6. `run_125m_local_mixer_probe.ps1`
7. `run_125m_local_mixer_validation_sequence.ps1`
8. `run_125m_sampled_teacher_probe.ps1`
9. `run_125m_superblock_probe_suite.ps1`

### 010 — 2026-07-30 — archived 150M-token chat launchers

Commit: `7386173`

1. `chat_drm_125m_local_mixer_150m.ps1`
2. `chat_gpt2_125m_150m.ps1`

### 011 — 2026-07-30 — Linux shell compatibility

Commit: `f5cad4e`

1. `_run_ps1_from_sh.sh`
2. `chat_drm_125m_4090.sh`
3. `chat_drm_125m_local_mixer_150m.sh`
4. `chat_drm_tta_demo.sh`
5. `chat_gpt2_125m_150m.sh`
6. `chat_gpt2_125m_4090.sh`
7. `prepare_wikipedia_en_125m_real.sh`
8. `prepare_wikipedia_en_5b.sh`
9. `profile_125m_drm_throughput_sweep.sh`
10. `run_10m_seed1_drm_vs_gpt2.sh`
11. `run_125m_150m_multiseed_competition.sh`
12. `run_125m_curriculum_probe.sh`
13. `run_125m_local_mixer_probe.sh`
14. `run_125m_local_mixer_validation_sequence.sh`
15. `run_125m_sampled_teacher_probe.sh`
16. `run_125m_superblock_probe_suite.sh`
17. `run_drm_125m_4090_base.sh`
18. `run_drm_500m_5b.sh`
19. `run_gpt2_125m_4090_base.sh`
20. `run_time_to_quality.sh`
21. `run_tta_multiseed_confirmation.sh`
22. `run_wiki_en_125m_matched.sh`
23. `run_wiki_en_125m_real_matched.sh`

Most files in this group delegate to the same-named PowerShell launcher through
`_run_ps1_from_sh.sh`; the two independent benchmark launchers introduced
later are native Bash.

### 012 — 2026-07-30 — independent validation pipeline

Commit: `1b19253`

1. `audit_dataset_contamination.py`
2. `download_pg19_test.py`
3. `evaluate_frozen_test.py`
4. `prepare_independent_benchmark.py`
5. `prepare_wikipedia_document_split.py`

### 013 — 2026-07-30 — independent CUDA smoke test

Commit: `4f98b97`

1. `run_independent_125m_smoke.sh`

### 014 — 2026-07-30 — frozen independent benchmark

Commit: `b6e9587`

1. `run_independent_125m_benchmark.sh`

### 015 — 2026-07-30 — corrected validation and DRM CE ablations

Commit: pending

1. `rescore_independent_125m_validation.py`
2. `run_independent_125m_validation_rescore.sh`
3. `run_drm_fix_ablation.py`
4. `run_drm_fix_ablation.sh`
5. `summarize_drm_fix_validation.py`
6. `rescore_drm_fix_validation.py`
7. `run_drm_fix_paired_5m.sh`
8. `run_mqar_architecture_probe.py`
9. `check_drm_fix_promotion.py`
10. `run_drm_geometry_component_suite.sh`
11. `run_drm_direct_control_suite.sh`
12. `run_drm_metric_order_suite.sh`
13. `rescore_drm_scaling_law.py`
14. `run_drm_scaling_law_100m.sh`

### 016 — 2026-07-31 — cached causal inference

Commit: pending

1. `benchmark_incremental_decode.py`

### 017 — 2026-08-01 — ASM-R post-promotion evaluation

Commit: pending

1. `evaluate_asm_r_checkpoint.py`
2. `evaluate_asm_r_mqar_curve.py`
3. `run_asm_r_mqar_curve.sh`
4. `run_asm_r_mqar_architecture_comparison.sh`
5. `run_asm_r_post_promotion_suite.sh`
6. `run_mqar_architecture_comparison.py`
7. `summarize_asm_r_post_promotion.py`
8. `run_transformer_asm_r_matched_100m.sh` — trains and frozen-rescores the
   83.0M-parameter Transformer control on the same 100M-token protocol as ASM-R.
9. `plot_asm_scaling_law.py` — generates publication-ready SVG charts and CSV
   data from frozen scaling-law results.
10. `plot_100m_model_comparison.py` — plots frozen quality, learning curves,
    throughput, GPU time, and Pareto comparisons for ASM and Transformer at 100M.
11. `rescore_asm_transformer_100m.py` — frozen paired rescoring over all eight
    ASM-R and Transformer milestones.
12. `benchmark_asm_transformer_paired.py` — paired context, prefill, cached
    decode, VRAM, and qualitative-generation benchmark.
13. `summarize_asm_transformer_paired.py` — consolidates the paired suite.
14. `run_asm_transformer_paired_suite.sh` — executes the full protocol with one
    command and optionally adds PG-19 context evaluation.
15. `benchmark_asm_r_long_streaming.py` — probes cache growth, long decode, CUDA
    memory, and delayed MQAR retention through 32K tokens.
16. `run_asm_r_long_streaming_suite.sh` — runs the complete 32K streaming suite.
17. `plot_asm_r_long_streaming.py` — generates reproducible SVG and CSV
    artifacts for cache growth, decode throughput, VRAM, and delayed MQAR.
18. `check_asm_c_parity.py` — quantifies real-checkpoint BF16 and argmax parity.
19. `compare_asm_r_asm_c_streaming.py` — applies ASM-C promotion criteria.
20. `run_asm_c_validation_suite.sh` — executes all five ASM-C validation phases.
21. `plot_asm_c_validation.py` — generates versioned ASM-C/ASM-R/Transformer
    streaming, memory, context, decode, and MQAR charts and CSV tables.
22. `run_asm_c_mqar_diagnostic.sh` — compares 5K/10K/20K MQAR learning for
    ASM-C, ASM-C with 2x memory width, ASM-S, and the matched Transformer.
23. `plot_mqar_architecture_comparison.py` — plots paired MQAR accuracy and CE
    learning curves and exports their source CSV.
24. `compare_asm_c2_controls.py` — applies short MQAR, causal-ablation, cache,
    VRAM, throughput, and long-retention promotion gates.
25. `plot_asm_c2_results.py` — renders ASM-C2 short-learning and long-streaming
    figures.
26. `run_asm_c2_mqar_suite.sh` — runs the gated ASM-C2 protocol end to end and
    blocks expensive long evaluation when short or causal-ablation gates fail.
27. `probe_addressable_memory.py` — compares oracle routing, the dense-slot
    learning curve, and fixed-capacity fast-weight memory on isolated MQAR.
28. `run_asm_c2_memory_learnability.sh` — runs the oracle, extended dense, and
    fast-weight isolated gate; reintegration remains blocked unless the
    non-oracle fast-weight control reaches 95% accuracy.
29. `run_asm_c2_sparse_probe.sh` — compatibility alias for the memory
    learnability runner formerly limited to sparse-slot controls.
30. `run_asm_c2_fw_suite.sh` — runs the gated ASM-C2-FW short MQAR, causal
    ablations, and 32K compact-streaming validation.
31. `train_asm_c2_fw_durable.py` — trains fast/slow ASM-C2-FW with delayed-MQAR
    curriculum, selective consolidation, FP32 memory, and language replay.
32. `summarize_asm_c2_fw_durable.py` — applies multiseed, real long-retention,
    streaming, language, and BF16 gates.
33. `run_asm_c2_fw_durable_suite.sh` — executes the complete durable-memory
    protocol and blocks expensive stages when the curriculum gate fails.
34. `train_asm_c2_fw_lm.py` — specializes ASM-R into ASM-C2-FW-LM using an
    80/20 language/MQAR mixture, ASM-R logit distillation, and separate
    backbone and fast-weight learning rates.
35. `run_asm_c2_fw_lm_suite.sh` — runs the three-seed compatibility protocol
    and repeats language CE, 32K MQAR, bounded-streaming, and BF16 gates.
36. `run_asm_c2_fw_lm_confirmation.sh` — performs independent three-lineage
    confirmation against ASM-R and matched Transformers on the complete frozen
    validation corpus, MQAR 32K, streaming, VRAM, throughput, and BF16 gates.
37. `summarize_asm_c2_fw_lm_confirmation.py` — applies the official promotion
    decision without treating Transformer CE superiority as a gate.
38. `plot_asm_c2_fw_lm_confirmation.py` — plots language CE and MQAR 32K with
    distinct family colors.
39. `rerun_asm_c2_fw_lm_parity.sh` — reruns only the three BF16 parity checks
    after numerical corrections and reapplies the official confirmation gate.
40. `run_asm_cm_post_fp32_validation.sh` — remeasures frozen CE, compact
    throughput, VRAM, cache, and BF16 parity for all three promoted candidate
    lineages without retraining.
41. `summarize_asm_c2_fw_lm_post_fp32.py` — applies the final post-FP32 gate
    and publishes the technical ASM-C2-FW-LM variant under the proposed public
    name ASM-CM only when every frozen measurement passes.
42. `plot_asm_cm_post_fp32.py` — generates dependency-free SVG charts and CSV
    source data for frozen CE, retained state, allocated VRAM, and streaming
    throughput.

## Naming guidance

- `train_*`: training entry points.
- `run_*`: experiment orchestration.
- `eval_*` / `evaluate_*`: read-only evaluation.
- `prepare_*` / `tokenize_*` / `make_*`: data or artifact preparation.
- `profile_*` / `check_*` / `audit_*`: diagnostics and verification.
- `chat_*` / `generate.py`: interactive inference and sampling.
- `summarize_*` / `analyze_*`: aggregation and reporting.

New scripts should be added to this index in their first commit. Stable,
descriptive filenames are preferred over numeric prefixes.
