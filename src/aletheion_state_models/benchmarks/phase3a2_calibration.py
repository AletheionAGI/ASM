"""Validation-only hard-rank calibration for Phase 3A.2."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .phase3a_checkpoint import atomic_torch_save, write_result
from .phase3a_data import sample_byte_windows
from .phase3a_training import evaluate_language_model, measure_streaming_error


def _load_model(run_root, variant, seed, builder, device):
    directory = Path(run_root) / variant / f"seed_{seed}"
    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=False)
    model, _ = builder(variant, seed); model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval(); return model, checkpoint, directory


def _scores(model, tokens, seed, *, batches, batch_size, sequence_length, device):
    values = []
    with torch.no_grad():
        for index in range(batches):
            inputs, _ = sample_byte_windows(tokens, batch_size=batch_size, sequence_length=sequence_length, seed=10_000 + seed, step=index, device=device)
            embedded = model.token_embedding(inputs)[:, ::model.config.directional_cumsum_block_size]
            observation = model.variable_rank_core.controller(embedded.reshape(-1, embedded.shape[-1]))
            values.append(observation.scores.cpu().numpy())
    return np.concatenate(values)


def calibrate_phase3a2(
    results, splits, *, run_root, builder, bases, seeds, batches,
    batch_size, sequence_length, device,
):
    """Calibrate one global validation threshold per base and rescore test once."""
    torch_device = torch.device(device); thresholds = {}; updated = list(results)
    for base in bases:
        variant = f"{base}_adaptive_32"; score_sets = []
        for seed in seeds:
            model, _, _ = _load_model(run_root, variant, seed, builder, torch_device)
            score_sets.append(_scores(model, splits.validation, seed, batches=batches, batch_size=batch_size, sequence_length=sequence_length, device=torch_device))
        scores = np.concatenate(score_sets); candidates = np.linspace(.05, .99, 941)
        ranks = np.asarray([np.maximum((scores >= value).sum(1), 16).mean() for value in candidates])
        threshold = float(candidates[np.argmin(np.abs(ranks - 32.0))]); thresholds[base] = threshold
        for index, result in enumerate(updated):
            if result.variant != variant: continue
            model, checkpoint, directory = _load_model(run_root, variant, result.seed, builder, torch_device)
            model.variable_rank_core.controller.threshold = threshold
            validation = evaluate_language_model(model, splits.validation, seed=10_000 + result.seed, batches=batches, batch_size=batch_size, sequence_length=sequence_length, device=torch_device)
            calibrated = replace(result, validation_ce=validation["ce"], validation_ppl=validation["ppl"], mean_rank=validation["mean_rank"], rank_std=validation["rank_std"], rank_min=validation["rank_min"], rank_max=validation["rank_max"], rank_ce_correlation=validation["rank_ce_correlation"], streaming_error=measure_streaming_error(model, splits.validation, torch_device))
            checkpoint["hard_budget_calibration"] = {"threshold": threshold, "source": "combined validation score distributions", "target_rank": 32}
            write_result(directory / "result.json", calibrated); atomic_torch_save(directory / "best.pt", checkpoint); updated[index] = calibrated
    return updated, thresholds


def open_phase3a2_test(
    results, splits, *, run_root, builder, thresholds, batches,
    batch_size, sequence_length, device,
):
    """Evaluate test after every validation-only decision has been frozen."""
    torch_device = torch.device(device); updated = []
    for result in results:
        model, checkpoint, directory = _load_model(
            run_root, result.variant, result.seed, builder, torch_device
        )
        base = "vr_s" if result.variant.startswith("vr_s_") else "vr_r"
        if result.variant.endswith("adaptive_32"):
            model.variable_rank_core.controller.threshold = thresholds[base]
        test = evaluate_language_model(
            model, splits.test, seed=20_000 + result.seed, batches=batches,
            batch_size=batch_size, sequence_length=sequence_length,
            device=torch_device,
        )
        final = replace(
            result, test_ce=test["ce"], test_ppl=test["ppl"],
            mean_rank=test["mean_rank"], rank_std=test["rank_std"],
            rank_min=test["rank_min"], rank_max=test["rank_max"],
            rank_ce_correlation=test["rank_ce_correlation"],
            streaming_error=measure_streaming_error(model, splits.validation, torch_device),
        )
        checkpoint["test_protocol"] = {"opened_after_validation_freeze": True}
        write_result(directory / "result.json", final)
        atomic_torch_save(directory / "best.pt", checkpoint); updated.append(final)
    return updated


__all__ = ["calibrate_phase3a2", "open_phase3a2_test"]
