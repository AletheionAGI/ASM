"""AdamM optimizer ablation on the selected ASM-VR Phase 3A.1 scaffold."""
from __future__ import annotations
from dataclasses import asdict, replace
from functools import partial
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean, pstdev
import torch
import numpy as np
from aletheion_state_models.benchmarks.phase3a_checkpoint import atomic_torch_save, write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits, sample_byte_windows
from aletheion_state_models.benchmarks.phase3a_training import Phase3ARunResult, evaluate_language_model, measure_streaming_error, train_phase3a_run
from aletheion_state_models.benchmarks.phase3a1_adamm_plots import render_adamm_charts
from aletheion_state_models.benchmarks.phase3a1_variants import build_stage_b_variant

VARIANTS = ("adamm_fixed_32", "adamm_adaptive_32")
SEEDS = (17, 29, 43)
COMPONENTS = (True, True, False)
ADAMM_SHA256 = "79495581868147a5bed69acc3e3a85e838634c3ced0aeb9ab98b35223722c877"
ADAMM_COMMIT = "980d84ce96825c3d11d6bc8dd98f0c5168897643"


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/wikipedia_en_20231101_sample.txt"))
    parser.add_argument("--adamm-source", type=Path, default=Path("../AdamM/adamm.py"))
    parser.add_argument("--baseline", type=Path, default=Path("docs/benchmarks/asm_vr_phase3a1/stage_b/summary.json"))
    parser.add_argument("--baseline-run-root", type=Path, default=Path("runs/asm_vr_phase3a1/stage_b"))
    parser.add_argument("--expected-adamm-sha256", default=ADAMM_SHA256)
    parser.add_argument("--run-root", type=Path, default=Path("runs/asm_vr_phase3a1_adamm"))
    parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/asm_vr_phase3a1_adamm"))
    parser.add_argument("--steps", type=int, default=489); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256); parser.add_argument("--evaluation-batches", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_adamm(path: Path):
    spec = importlib.util.spec_from_file_location("asm_vr_external_adamm", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load AdamM from {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.AdamM


def _variant_builder(variant: str, seed: int):
    mapped = {"adamm_fixed_32": "selected_fixed_32", "adamm_adaptive_32": "selected_adaptive_32"}[variant]
    return build_stage_b_variant(mapped, seed, components=COMPONENTS)


def _scores(model, tokens, seed, args, device):
    values = []
    with torch.no_grad():
        for index in range(args.evaluation_batches):
            inputs, _ = sample_byte_windows(tokens, batch_size=args.batch_size, sequence_length=args.sequence_length, seed=10_000 + seed, step=index, device=device)
            embedded = model.token_embedding(inputs)[:, ::model.config.directional_cumsum_block_size]
            observation = model.variable_rank_core.controller(embedded.reshape(-1, embedded.shape[-1]))
            values.append(observation.scores.cpu().numpy())
    return np.concatenate(values)


def _load_model(run_root, seed, device):
    model, _ = _variant_builder("adamm_adaptive_32", seed)
    directory = run_root / "adamm_adaptive_32" / f"seed_{seed}"
    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"]); model.to(device).eval()
    return model, checkpoint, directory


def _calibrate(results, splits, args):
    device = torch.device(args.device); score_sets = []
    for seed in SEEDS:
        model, _, _ = _load_model(args.run_root, seed, device)
        score_sets.append(_scores(model, splits.validation, seed, args, device))
    scores = np.concatenate(score_sets); candidates = np.linspace(.05, .99, 941)
    ranks = np.asarray([np.maximum((scores >= value).sum(1), 16).mean() for value in candidates])
    threshold = float(candidates[np.argmin(np.abs(ranks - 32.0))]); updated = []; sensitivity = []
    for result in results:
        if result.variant != "adamm_adaptive_32": updated.append(result); continue
        model, checkpoint, directory = _load_model(args.run_root, result.seed, device)
        model.variable_rank_core.controller.threshold = .672
        fixed_threshold = evaluate_language_model(model, splits.test, seed=20_000 + result.seed, batches=args.evaluation_batches, batch_size=args.batch_size, sequence_length=args.sequence_length, device=device)
        sensitivity.append({"seed": result.seed, "threshold": .672, "test_ce": fixed_threshold["ce"], "mean_rank": fixed_threshold["mean_rank"]})
        model.variable_rank_core.controller.threshold = threshold
        validation = evaluate_language_model(model, splits.validation, seed=10_000 + result.seed, batches=args.evaluation_batches, batch_size=args.batch_size, sequence_length=args.sequence_length, device=device)
        test = evaluate_language_model(model, splits.test, seed=20_000 + result.seed, batches=args.evaluation_batches, batch_size=args.batch_size, sequence_length=args.sequence_length, device=device)
        result = replace(result, validation_ce=validation["ce"], validation_ppl=validation["ppl"], test_ce=test["ce"], test_ppl=test["ppl"], mean_rank=test["mean_rank"], rank_std=test["rank_std"], rank_min=test["rank_min"], rank_max=test["rank_max"], rank_ce_correlation=test["rank_ce_correlation"], streaming_error=measure_streaming_error(model, splits.validation, device))
        checkpoint["hard_budget_calibration"] = {"threshold": threshold, "source": "combined validation score distributions", "target_rank": 32}
        write_result(directory / "result.json", result); atomic_torch_save(directory / "best.pt", checkpoint); updated.append(result)
    return updated, threshold, sensitivity


def _aggregate(runs):
    output = {"runs": [asdict(run) for run in runs]}
    for field in ("validation_ce", "test_ce", "mean_rank", "rank_std", "tokens_per_second", "peak_memory_mb", "streaming_error"):
        values = [getattr(run, field) for run in runs]; output[f"{field}_mean"] = mean(values); output[f"{field}_std"] = pstdev(values)
    return output


def _summary(results, baseline, threshold, sensitivity, source, source_hash, state_bytes, baseline_state_bytes):
    grouped = {name: sorted((run for run in results if run.variant == name), key=lambda run: run.seed) for name in VARIANTS}
    variants = {name: _aggregate(runs) for name, runs in grouped.items()}; comparisons = {}
    mapping = {"adamm_fixed_32": "selected_fixed_32", "adamm_adaptive_32": "selected_adaptive_32"}
    for name, baseline_name in mapping.items():
        old = {run["seed"]: run for run in baseline["variants"][baseline_name]["runs"]}
        comparisons[name] = [{"seed": run.seed, "adamm_minus_adamw_test_ce": run.test_ce - old[run.seed]["test_ce"]} for run in grouped[name]]
    adaptive = variants["adamm_adaptive_32"]; fixed = variants["adamm_fixed_32"]
    fixed_deltas = [item["adamm_minus_adamw_test_ce"] for item in comparisons["adamm_fixed_32"]]
    adaptive_deltas = [item["adamm_minus_adamw_test_ce"] for item in comparisons["adamm_adaptive_32"]]
    fixed_delta, adaptive_delta = mean(fixed_deltas), mean(adaptive_deltas)
    gates = {"complete": all(len(runs) == 3 for runs in grouped.values()), "finite": all(run.finite for run in results), "streaming": max(run.streaming_error for run in results) <= 1e-4, "budget": 28.8 <= adaptive["mean_rank_mean"] <= 35.2, "variation": min(run.rank_std for run in grouped["adamm_adaptive_32"]) > 1, "controller_gradient": min(run.controller_gradient_fraction for run in grouped["adamm_adaptive_32"]) >= .9, "adaptive_optimizer_noninferior": adaptive_delta <= .02 and max(adaptive_deltas) <= .05, "adaptive_optimizer_superior": adaptive_delta <= -.02 and sum(value < 0 for value in adaptive_deltas) >= 2, "near_adamm_fixed32": adaptive["test_ce_mean"] - fixed["test_ce_mean"] <= .05, "pareto_vs_adamm_fixed32": not (fixed["mean_rank_mean"] <= adaptive["mean_rank_mean"] and fixed["test_ce_mean"] <= adaptive["test_ce_mean"] - .01)}
    technical = ("complete", "finite", "streaming", "budget", "variation", "controller_gradient")
    return {"experiment": "3A.1-AdamM", "optimizer": {"source": str(source), "sha256": source_hash, "commit": ADAMM_COMMIT, "parameters": {"lr": 3e-4, "betas": [.9, .999], "beta1_min": .5, "shock_ratio": 1.5, "adapt_strength": .03, "weight_decay": .01}, "state_bytes_by_variant": state_bytes, "adamw_state_bytes_by_variant": baseline_state_bytes}, "calibrated_threshold": threshold, "fixed_threshold_sensitivity": sensitivity, "variants": variants, "adamw_baseline": baseline["variants"], "optimizer_comparisons": comparisons, "optimizer_effects": {"fixed32_test_ce": fixed_delta, "adaptive_test_ce": adaptive_delta, "interaction_adaptive_minus_fixed": adaptive_delta - fixed_delta}, "gates": gates, "technical_passed": all(gates[name] for name in technical), "scientific_passed": gates["adaptive_optimizer_superior"] and gates["near_adamm_fixed32"] and gates["pareto_vs_adamm_fixed32"]}


def _optimizer_state_bytes(run_root, variants):
    output = {}
    for variant in variants:
        values = []
        for seed in SEEDS:
            checkpoint = torch.load(run_root / variant / f"seed_{seed}" / "best.pt", map_location="cpu", weights_only=False)
            values.append(sum(value.numel() * value.element_size() for state in checkpoint["optimizer_state"]["state"].values() for value in state.values() if isinstance(value, torch.Tensor)))
        output[variant] = {"mean": mean(values), "by_seed": values}
    return output


def main():
    args = _arguments(); source_hash = hashlib.sha256(args.adamm_source.read_bytes()).hexdigest()
    if source_hash != args.expected_adamm_sha256: raise RuntimeError(f"AdamM source hash mismatch: {source_hash}")
    splits = load_document_hash_splits(args.corpus); AdamM = _load_adamm(args.adamm_source)
    optimizer_factory = lambda parameters, lr: AdamM(parameters, lr=lr, weight_decay=.01)
    results = []
    for seed in SEEDS:
        for variant in VARIANTS:
            print(f"training {variant} seed={seed}", flush=True)
            result = train_phase3a_run(variant, seed, splits, output_directory=args.run_root, steps=args.steps, batch_size=args.batch_size, sequence_length=args.sequence_length, evaluation_batches=args.evaluation_batches, device=args.device, variant_builder=_variant_builder, adaptive_variants=frozenset({"adamm_adaptive_32"}), optimizer_factory=optimizer_factory)
            results.append(result); print(f"done {variant} seed={seed} test={result.test_ce:.4f} rank={result.mean_rank:.2f}", flush=True)
    results, threshold, sensitivity = _calibrate(results, splits, args); baseline = json.loads(args.baseline.read_text())
    state_bytes = _optimizer_state_bytes(args.run_root, VARIANTS)
    baseline_state_bytes = _optimizer_state_bytes(args.baseline_run_root, ("selected_fixed_32", "selected_adaptive_32"))
    summary = _summary(results, baseline, threshold, sensitivity, args.adamm_source, source_hash, state_bytes, baseline_state_bytes)
    args.output.mkdir(parents=True, exist_ok=True); write_result(args.output / "summary.json", summary); render_adamm_charts(summary, args.output)
    print(json.dumps({"gates": summary["gates"], "threshold": threshold}, indent=2)); raise SystemExit(0 if summary["technical_passed"] else 1)


if __name__ == "__main__": main()
