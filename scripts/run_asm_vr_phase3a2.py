"""Run the complete matched ASM-VR-R versus ASM-VR-S Phase 3A.2 matrix."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits
from aletheion_state_models.benchmarks.phase3a_training import train_phase3a_run
from aletheion_state_models.benchmarks.phase3a2_calibration import calibrate_phase3a2, open_phase3a2_test
from aletheion_state_models.benchmarks.phase3a2_plots import render_phase3a2_charts
from aletheion_state_models.benchmarks.phase3a2_summary import select_common_policy, summarize_phase3a2
from aletheion_state_models.benchmarks.phase3a2_variants import ASM_S_MEMORY_HIDDEN_SIZE, BASES, PHASE3A2_VARIANTS, RANK_ARMS, build_phase3a2_variant


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/wikipedia_en_20231101_sample.txt"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/asm_vr_phase3a2"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/asm_vr_phase3a2"))
    parser.add_argument("--steps", type=int, default=489); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256); parser.add_argument("--evaluation-batches", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43]); parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _implementation_hash():
    files = (
        "src/drm_language_emitter/directional_blocks.py",
        "src/aletheion_state_models/variants/variable_rank.py",
        "src/aletheion_state_models/variants/selective_state.py",
        "src/aletheion_state_models/benchmarks/phase3a2_variants.py",
        "src/aletheion_state_models/benchmarks/phase3a2_calibration.py",
        "src/aletheion_state_models/benchmarks/phase3a2_summary.py",
        "scripts/run_asm_vr_phase3a2.py",
    )
    digest = hashlib.sha256()
    for name in files: digest.update(name.encode()); digest.update(Path(name).read_bytes())
    return {"sha256": digest.hexdigest(), "files": list(files)}


def main():
    args = _arguments(); splits = load_document_hash_splits(args.corpus); results = []
    order = [f"{base}_{arm}" for arm in RANK_ARMS for base in BASES]
    for seed in args.seeds:
        for variant in order:
            print(f"training {variant} seed={seed}", flush=True)
            result = train_phase3a_run(
                variant, seed, splits, output_directory=args.run_root,
                steps=args.steps, batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                evaluation_batches=args.evaluation_batches,
                milestones=tuple(sorted({100, 200, 300, 400, args.steps})),
                device=args.device, variant_builder=build_phase3a2_variant,
                adaptive_variants=frozenset({"vr_r_adaptive_32", "vr_s_adaptive_32"}),
                evaluate_test=False,
            )
            results.append(result)
            print(f"done {variant} seed={seed} val={result.validation_ce:.4f} rank={result.mean_rank:.2f}", flush=True)
    results, thresholds = calibrate_phase3a2(
        results, splits, run_root=args.run_root, builder=build_phase3a2_variant,
        bases=BASES, seeds=args.seeds, batches=args.evaluation_batches,
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        device=args.device,
    )
    policy = select_common_policy(results)
    counts = {}
    for base in BASES:
        model, _ = build_phase3a2_variant(f"{base}_full", args.seeds[0])
        counts[base] = {"total": sum(parameter.numel() for parameter in model.parameters()), "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
    manifest = {
        "phase": "3A.2", "corpus": splits.manifest, "seeds": args.seeds,
        "steps": args.steps, "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "tokens_per_run": args.steps * args.batch_size * args.sequence_length,
        "variants": list(PHASE3A2_VARIANTS), "optimizer": "AdamW",
        "asm_s_memory_hidden_size": ASM_S_MEMORY_HIDDEN_SIZE,
        "parameter_counts": counts, "thresholds": thresholds,
        "validation_policy_selection": policy,
        "test_protocol": {"opened_after_validation_freeze": False},
        "provenance": {"repository_dirty": True, "implementation": _implementation_hash()},
    }
    args.output.mkdir(parents=True, exist_ok=True); write_result(args.output / "manifest.json", manifest)
    results = open_phase3a2_test(
        results, splits, run_root=args.run_root, builder=build_phase3a2_variant,
        thresholds=thresholds, batches=args.evaluation_batches,
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        device=args.device,
    )
    manifest["test_protocol"]["opened_after_validation_freeze"] = True; write_result(args.output / "manifest.json", manifest)
    summary = summarize_phase3a2(results, thresholds); write_result(args.output / "summary.json", summary); render_phase3a2_charts(summary, args.output)
    print(json.dumps({"base_gates": summary["base_gates"], "adaptive": {base: value["gates"] for base, value in summary["adaptive"].items()}, "selection": summary["base_selection"], "policy": policy}, indent=2))
    raise SystemExit(0 if summary["technical_passed"] else 1)


if __name__ == "__main__": main()
