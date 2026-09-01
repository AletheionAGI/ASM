"""Revalidate Phase 3A.1 streaming and regenerate both stage reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from aletheion_state_models.benchmarks.phase3a1_plots import render_stage_a, render_stage_b
from aletheion_state_models.benchmarks.phase3a1_summary import summarize_stage_a, summarize_stage_b
from aletheion_state_models.benchmarks.phase3a1_variants import STAGE_A_VARIANTS, STAGE_B_VARIANTS, build_stage_a_variant, stage_b_builder
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits
from aletheion_state_models.benchmarks.phase3a_training import Phase3ARunResult, measure_streaming_error


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/wikipedia_en_20231101_sample.txt"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/asm_vr_phase3a1"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/asm_vr_phase3a1"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _revalidate(variants, seeds, run_root, builder, tokens, device, threshold=None):
    results = []
    for seed in seeds:
        for variant in variants:
            directory = run_root / variant / f"seed_{seed}"
            result = Phase3ARunResult(**json.loads((directory / "result.json").read_text()))
            checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=False)
            model, _ = builder(variant, seed); model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
            if threshold is not None and "adaptive" in variant:
                model.variable_rank_core.controller.threshold = threshold
            result = Phase3ARunResult(**(result.__dict__ | {"streaming_error": measure_streaming_error(model, tokens, device)}))
            write_result(directory / "result.json", result); results.append(result)
    return results


def main():
    args = _arguments(); device = torch.device(args.device); splits = load_document_hash_splits(args.corpus)
    manifest = json.loads((args.output / "manifest.json").read_text()); seeds = manifest["seeds"]
    stage_a_results = _revalidate(STAGE_A_VARIANTS, seeds, args.run_root / "stage_a", build_stage_a_variant, splits.validation, device)
    stage_a = summarize_stage_a(stage_a_results); write_result(args.output / "stage_a/summary.json", stage_a); render_stage_a(stage_a, args.output / "stage_a")
    selection = stage_a["selection"]; builder = stage_b_builder(tuple(selection["components"])); threshold = manifest["hard_budget_calibration"]["threshold"]
    stage_b_results = _revalidate(STAGE_B_VARIANTS, seeds, args.run_root / "stage_b", builder, splits.validation, device, threshold)
    stage_b = summarize_stage_b(stage_b_results, selection, threshold); write_result(args.output / "stage_b/summary.json", stage_b); render_stage_b(stage_b, args.output / "stage_b")
    print(json.dumps({"stage_a": stage_a["gates"], "stage_b": stage_b["gates"]}, indent=2))
    raise SystemExit(0 if stage_a["operational_passed"] and stage_b["operational_passed"] else 1)


if __name__ == "__main__":
    main()
