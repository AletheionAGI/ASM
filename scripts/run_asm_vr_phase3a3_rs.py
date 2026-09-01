"""Run the explicit ASM-VR-RS full-rank comparison."""
import argparse
import hashlib
import json
from pathlib import Path
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits
from aletheion_state_models.benchmarks.phase3a_training import train_phase3a_run
from aletheion_state_models.benchmarks.phase3a3_plots import render_phase3a3_charts
from aletheion_state_models.benchmarks.phase3a3_summary import summarize_phase3a3
from aletheion_state_models.benchmarks.phase3a3_variants import build_phase3a3_variant


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--corpus", type=Path, default=Path("data/wikipedia_en_20231101_sample.txt")); parser.add_argument("--phase3a2", type=Path, default=Path("docs/benchmarks/asm_vr_phase3a2/summary.json")); parser.add_argument("--run-root", type=Path, default=Path("runs/asm_vr_phase3a3_rs")); parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/asm_vr_phase3a3_rs")); parser.add_argument("--device", default="cuda"); args = parser.parse_args()
    splits = load_document_hash_splits(args.corpus); results = []
    for seed in (17, 29, 43):
        print(f"training vr_rs_full seed={seed}", flush=True)
        result = train_phase3a_run("vr_rs_full", seed, splits, output_directory=args.run_root, steps=489, batch_size=16, sequence_length=256, evaluation_batches=16, device=args.device, variant_builder=build_phase3a3_variant, adaptive_variants=frozenset())
        results.append(result); print(f"done seed={seed} test={result.test_ce:.4f}", flush=True)
    baseline = json.loads(args.phase3a2.read_text()); summary = summarize_phase3a3(results, baseline); args.output.mkdir(parents=True, exist_ok=True); write_result(args.output / "summary.json", summary); render_phase3a3_charts(summary, args.output)
    sources = ("src/aletheion_state_models/variants/relational_selective_state.py", "src/aletheion_state_models/benchmarks/phase3a3_variants.py", "scripts/run_asm_vr_phase3a3_rs.py"); digest = hashlib.sha256()
    for source in sources: digest.update(source.encode()); digest.update(Path(source).read_bytes())
    model, _ = build_phase3a3_variant("vr_rs_full", 17); manifest = {"phase": "3A.3", "name": summary["name"], "seeds": [17,29,43], "tokens_per_run": 489*16*256, "corpus": splits.manifest, "parameter_count": sum(p.numel() for p in model.parameters()), "parameter_matched_to_r_s": False, "historical_note": "formalizes the relational+selective recipe already used by practical ASM-R", "provenance": {"repository_dirty": True, "implementation_sha256": digest.hexdigest(), "files": sources}}; write_result(args.output / "manifest.json", manifest)
    print(json.dumps({"winner": summary["quality_winner"], "gates": summary["gates"]}, indent=2)); raise SystemExit(0 if summary["technical_passed"] else 1)


if __name__ == "__main__": main()
