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
