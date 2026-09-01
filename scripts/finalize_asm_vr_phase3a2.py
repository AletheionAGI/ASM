"""Revalidate Phase 3A.2 streaming after fixed-shape open-block inference."""
from __future__ import annotations
from dataclasses import replace
import argparse
import json
from pathlib import Path
import torch
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits
from aletheion_state_models.benchmarks.phase3a_training import Phase3ARunResult, measure_streaming_error
from aletheion_state_models.benchmarks.phase3a2_plots import render_phase3a2_charts
from aletheion_state_models.benchmarks.phase3a2_summary import summarize_phase3a2
from aletheion_state_models.benchmarks.phase3a2_variants import PHASE3A2_VARIANTS, build_phase3a2_variant


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--corpus", type=Path, default=Path("data/wikipedia_en_20231101_sample.txt")); parser.add_argument("--run-root", type=Path, default=Path("runs/asm_vr_phase3a2")); parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/asm_vr_phase3a2")); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    splits = load_document_hash_splits(args.corpus); device = torch.device(args.device); manifest = json.loads((args.output / "manifest.json").read_text()); results = []
    for seed in manifest["seeds"]:
        for variant in PHASE3A2_VARIANTS:
            directory = args.run_root / variant / f"seed_{seed}"; result = Phase3ARunResult(**json.loads((directory / "result.json").read_text())); checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=False)
            model, _ = build_phase3a2_variant(variant, seed); model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
            if variant.endswith("adaptive_32"): model.variable_rank_core.controller.threshold = manifest["thresholds"]["vr_s" if variant.startswith("vr_s_") else "vr_r"]
            result = replace(result, streaming_error=measure_streaming_error(model, splits.validation, device)); write_result(directory / "result.json", result); results.append(result)
    summary = summarize_phase3a2(results, manifest["thresholds"]); write_result(args.output / "summary.json", summary); render_phase3a2_charts(summary, args.output)
    manifest["streaming_fix"] = "configured block size with zero-padded causal open block"; write_result(args.output / "manifest.json", manifest)
    print(json.dumps({"base_gates": summary["base_gates"], "selection": summary["base_selection"]}, indent=2)); raise SystemExit(0 if summary["technical_passed"] else 1)


if __name__ == "__main__": main()
