"""Run both projected-scaffold stages of ASM-VR Phase 3A.1."""
from __future__ import annotations
from dataclasses import replace
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from aletheion_state_models.benchmarks.phase3a1_plots import render_stage_a, render_stage_b
from aletheion_state_models.benchmarks.phase3a1_summary import summarize_stage_a, summarize_stage_b
from aletheion_state_models.benchmarks.phase3a1_variants import (
    STAGE_A_COMPONENTS, STAGE_A_VARIANTS, STAGE_B_VARIANTS,
    build_stage_a_variant, stage_b_builder,
)
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits, sample_byte_windows
from aletheion_state_models.benchmarks.phase3a_training import (
    Phase3ARunResult, evaluate_language_model, measure_streaming_error, train_phase3a_run,
)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/wikipedia_en_20231101_sample.txt"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/asm_vr_phase3a1"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/asm_vr_phase3a1"))
    parser.add_argument("--steps", type=int, default=489)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--evaluation-batches", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _train_matrix(variants, seeds, splits, directory, args, builder, adaptive):
    results = []
    for seed in seeds:
        for variant in variants:
            print(f"training {variant} seed={seed}", flush=True)
            result = train_phase3a_run(
                variant, seed, splits, output_directory=directory,
                steps=args.steps, batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                evaluation_batches=args.evaluation_batches,
                milestones=tuple(sorted({100, 200, 300, 400, args.steps})),
                device=args.device, variant_builder=builder,
                adaptive_variants=frozenset(adaptive),
            )
            results.append(result)
            print(f"done {variant} seed={seed} val={result.validation_ce:.4f} test={result.test_ce:.4f} rank={result.mean_rank:.2f}", flush=True)
    return results


def _load_adaptive_model(run_root, seed, builder, device):
    model, _ = builder("selected_adaptive_32", seed)
    checkpoint = torch.load(run_root / "selected_adaptive_32" / f"seed_{seed}" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
    return model, checkpoint


def _validation_scores(model, tokens, seed, args, device):
    values = []
    with torch.no_grad():
        for index in range(args.evaluation_batches):
            inputs, _ = sample_byte_windows(tokens, batch_size=args.batch_size, sequence_length=args.sequence_length, seed=10_000 + seed, step=index, device=device)
            embeddings = model.token_embedding(inputs)[:, :: model.config.directional_cumsum_block_size]
            observation = model.variable_rank_core.controller(embeddings.reshape(-1, embeddings.shape[-1]))
            values.append(observation.scores.cpu().numpy())
    return np.concatenate(values)


def _calibrate(results, splits, run_root, seeds, builder, args):
    device = torch.device(args.device); score_sets = []
    for seed in seeds:
        model, _ = _load_adaptive_model(run_root, seed, builder, device)
        score_sets.append(_validation_scores(model, splits.validation, seed, args, device))
    scores = np.concatenate(score_sets)
    candidates = np.linspace(.05, .99, 941)
    ranks = np.asarray([np.maximum((scores >= threshold).sum(1), 16).mean() for threshold in candidates])
    threshold = float(candidates[np.argmin(np.abs(ranks - 32.0))])
    updated = []
    for result in results:
        if result.variant != "selected_adaptive_32":
            updated.append(result); continue
        model, checkpoint = _load_adaptive_model(run_root, result.seed, builder, device)
        model.variable_rank_core.controller.threshold = threshold
        validation = evaluate_language_model(model, splits.validation, seed=10_000 + result.seed, batches=args.evaluation_batches, batch_size=args.batch_size, sequence_length=args.sequence_length, device=device)
        test = evaluate_language_model(model, splits.test, seed=20_000 + result.seed, batches=args.evaluation_batches, batch_size=args.batch_size, sequence_length=args.sequence_length, device=device)
        calibrated = replace(result, validation_ce=validation["ce"], validation_ppl=validation["ppl"], test_ce=test["ce"], test_ppl=test["ppl"], mean_rank=test["mean_rank"], rank_std=test["rank_std"], rank_min=test["rank_min"], rank_max=test["rank_max"], rank_ce_correlation=test["rank_ce_correlation"], streaming_error=measure_streaming_error(model, splits.validation, device))
        directory = run_root / result.variant / f"seed_{result.seed}"
        write_result(directory / "result.json", calibrated)
        checkpoint["hard_budget_calibration"] = {"threshold": threshold, "source": "combined validation score distributions", "target_rank": 32}
        torch.save(checkpoint, directory / "best.pt")
        updated.append(calibrated)
    return updated, threshold


def main():
    args = _arguments(); splits = load_document_hash_splits(args.corpus)
    args.output.mkdir(parents=True, exist_ok=True); args.run_root.mkdir(parents=True, exist_ok=True)
    stage_a_results = _train_matrix(STAGE_A_VARIANTS, args.seeds, splits, args.run_root / "stage_a", args, build_stage_a_variant, ())
    stage_a = summarize_stage_a(stage_a_results); write_result(args.output / "stage_a" / "summary.json", stage_a); render_stage_a(stage_a, args.output / "stage_a")
    selected = stage_a["selection"]; components = tuple(selected["components"]); builder = stage_b_builder(components)
    stage_b_results = _train_matrix(STAGE_B_VARIANTS, args.seeds, splits, args.run_root / "stage_b", args, builder, ("selected_adaptive_32",))
    stage_b_results, threshold = _calibrate(stage_b_results, splits, args.run_root / "stage_b", args.seeds, builder, args)
    stage_b = summarize_stage_b(stage_b_results, selected, threshold); write_result(args.output / "stage_b" / "summary.json", stage_b); render_stage_b(stage_b, args.output / "stage_b")
    manifest = {"phase": "3A.1", "corpus": splits.manifest, "seeds": args.seeds, "steps": args.steps, "batch_size": args.batch_size, "sequence_length": args.sequence_length, "tokens_per_run": args.steps * args.batch_size * args.sequence_length, "stage_a_variants": list(STAGE_A_VARIANTS), "stage_b_variants": list(STAGE_B_VARIANTS), "selection": selected, "hard_budget_calibration": {"threshold": threshold, "source": "combined validation score distributions; no test labels"}}
    write_result(args.output / "manifest.json", manifest)
    print(json.dumps({"stage_a_gates": stage_a["gates"], "selection": selected, "stage_b_gates": stage_b["gates"]}, indent=2))
    raise SystemExit(0 if stage_a["operational_passed"] and stage_b["operational_passed"] else 1)


if __name__ == "__main__":
    main()
