"""Matched mixed-language/MQAR specialization for the purpose suite."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import time
import torch
from torch.nn import functional as F
from .phase3a_checkpoint import atomic_torch_save, write_result
from .phase3a_data import sample_byte_windows
from .phase3a_training import evaluate_language_model

CURRICULUM_LENGTHS = (40, 80, 160, 320, 512, 1024, 4096)
TEST_LENGTHS = (40, 512, 4096, 32768)


def delayed_mqar_batch(length, batch_size, generator, device):
    n_pairs = n_queries = 8; key_offset = 2; value_offset = 34; query_token = 1
    minimum = 2 * n_pairs + 3 * n_queries
    if length < minimum: raise ValueError("MQAR length is too short")
    sequences, masks = [], []
    for _ in range(batch_size):
        keys = torch.randperm(32, generator=generator)[:n_pairs] + key_offset
        values = torch.randint(0, 64, (n_pairs,), generator=generator) + value_offset
        queries = torch.randperm(n_pairs, generator=generator)[:n_queries]
        tokens = []
        for key, value in zip(keys.tolist(), values.tolist()): tokens += [key, value]
        tokens += torch.randint(98, 256, (length - minimum,), generator=generator).tolist()
        answers = []
        for index in queries.tolist():
            tokens += [query_token, int(keys[index]), int(values[index])]; answers.append(len(tokens) - 2)
        sequence = torch.tensor(tokens); mask = torch.zeros(length - 1, dtype=torch.bool); mask[torch.tensor(answers)] = True
        sequences.append(sequence); masks.append(mask)
    stacked = torch.stack(sequences).to(device)
    return stacked[:, :-1], stacked[:, 1:], torch.stack(masks).to(device)


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.inference_mode()
def evaluate_mqar(model, length, *, seed, batches, device):
    model.eval(); generator = torch.Generator().manual_seed(seed + length); losses = []; rank_rows = []; correct = total = 0
    batch_size = max(1, min(16, 4096 // length))
    for _ in range(batches):
        x, y, mask = delayed_mqar_batch(length, batch_size, generator, device)
        with _autocast(device): output = model(x, collect_diagnostics=False); selected = output["logits"][mask]
        if output.get("variable_rank_ranks") is not None: rank_rows.append(output["variable_rank_ranks"].float().reshape(-1).cpu())
        targets = y[mask]; losses.append(float(F.cross_entropy(selected.float(), targets))); correct += int((selected.argmax(-1) == targets).sum()); total += targets.numel()
    ce = sum(losses) / len(losses); ranks = torch.cat(rank_rows) if rank_rows else None
    return {"length": length, "ce": ce if math.isfinite(ce) else None, "ce_finite": math.isfinite(ce), "accuracy": correct / total, "correct": correct, "targets": total, "mean_rank": float(ranks.mean()) if ranks is not None else None, "rank_std": float(ranks.std(unbiased=False)) if ranks is not None else None, "rank_min": float(ranks.min()) if ranks is not None else None, "rank_max": float(ranks.max()) if ranks is not None else None}


def specialize_and_evaluate(
    model, splits, *, variant, seed, output_directory, steps=1000,
    test_lengths=TEST_LENGTHS, device="cuda",
):
    """Run identical 80% language / 20% MQAR specialization and frozen tests."""
    torch_device = torch.device(device); model.to(torch_device).train(); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed); history = []; started = time.perf_counter(); mqar_index = 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if step % 5:
            x, y = sample_byte_windows(splits.train, batch_size=16, sequence_length=256, seed=seed, step=10_000 + step, device=torch_device); mask = None
        else:
            stage = min(mqar_index * len(CURRICULUM_LENGTHS) // max(steps // 5, 1), len(CURRICULUM_LENGTHS) - 1); length = CURRICULUM_LENGTHS[stage]; batch_size = max(1, min(16, 4096 // length)); x, y, mask = delayed_mqar_batch(length, batch_size, generator, torch_device); mqar_index += 1
        with _autocast(torch_device):
            output = model(x, collect_diagnostics=False, global_step=step); logits = output["logits"]; selected = logits[mask] if mask is not None else logits.reshape(-1, 256); targets = y[mask] if mask is not None else y.reshape(-1); loss = F.cross_entropy(selected.float(), targets)
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite specialization loss for {variant}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        if step in {250, 500, 750, steps}:
            history.append({"step": step, "loss": float(loss.detach()), "mqar_batches_seen": mqar_index}); print({"variant": variant, "seed": seed, **history[-1]}, flush=True)
    elapsed = time.perf_counter() - started; model.eval()
    language = evaluate_language_model(model, splits.test, seed=30_000 + seed, batches=16, batch_size=16, sequence_length=256, device=torch_device)
    mqar = []
    for length in test_lengths:
        batches = 16 if length <= 512 else (8 if length <= 4096 else 4)
        try: mqar.append(evaluate_mqar(model, length, seed=40_000 + seed, batches=batches, device=torch_device))
        except (torch.OutOfMemoryError, RuntimeError) as exc: mqar.append({"length": length, "status": "failed", "error": str(exc)})
    directory = Path(output_directory) / variant / f"seed_{seed}"; atomic_torch_save(directory / "mqar_best.pt", {"model_state": deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()}), "optimizer_state": optimizer.state_dict(), "steps": steps}); result = {"variant": variant, "seed": seed, "steps": steps, "elapsed_sec": elapsed, "language_test_ce_after": language["ce"], "language_test_ppl_after": language["ppl"], "mqar": mqar, "history": history}; write_result(directory / "mqar_result.json", result); return result


MEMORY_HEAVY_CURRICULUM = (
    (40, 1000, 4), (80, 500, 4), (160, 500, 4), (320, 300, 4),
    (512, 200, 2), (1024, 100, 1), (4096, 25, 1),
)


def specialize_memory_heavy(
    model, splits, *, variant, seed, output_directory,
    test_lengths=TEST_LENGTHS, device="cuda",
):
    """Specialize with the validated 80% MQAR / 20% language replay curriculum."""
    torch_device = torch.device(device); model.to(torch_device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed); history = []; update = 0; mqar_seen = 0
    started = time.perf_counter()
    for length, batches, batch_size in MEMORY_HEAVY_CURRICULUM:
        stage_losses = []
        for _ in range(batches):
            update += 1; mqar_seen += 1; optimizer.zero_grad(set_to_none=True)
            x, y, mask = delayed_mqar_batch(length, batch_size, generator, torch_device)
            with _autocast(torch_device):
                selected = model(x, collect_diagnostics=False, global_step=update)["logits"][mask]
                loss = F.cross_entropy(selected.float(), y[mask])
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite MQAR loss for {variant}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step(); stage_losses.append(float(loss.detach()))
            if mqar_seen % 4 == 0:
                update += 1; optimizer.zero_grad(set_to_none=True)
                lx, ly = sample_byte_windows(splits.train, batch_size=4, sequence_length=256, seed=seed, step=20_000 + update, device=torch_device)
                with _autocast(torch_device):
                    logits = model(lx, collect_diagnostics=False, global_step=update)["logits"]
                    language_loss = F.cross_entropy(logits.reshape(-1, 256).float(), ly.reshape(-1))
                language_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        model.eval(); control = evaluate_mqar(model, 40, seed=35_000 + seed + length, batches=8, device=torch_device); model.train()
        row = {"length": length, "mqar_batches_seen": mqar_seen, "updates": update, "stage_loss_mean": sum(stage_losses) / len(stage_losses), "control_40_accuracy": control["accuracy"]}; history.append(row); print({"variant": variant, "seed": seed, **row}, flush=True)
    elapsed = time.perf_counter() - started; model.eval()
    language = evaluate_language_model(model, splits.test, seed=30_000 + seed, batches=16, batch_size=16, sequence_length=256, device=torch_device)
    mqar = []
    for length in test_lengths:
        batches = 16 if length <= 512 else (8 if length <= 4096 else 4)
        try: mqar.append(evaluate_mqar(model, length, seed=40_000 + seed, batches=batches, device=torch_device))
        except (torch.OutOfMemoryError, RuntimeError) as exc: mqar.append({"length": length, "status": "failed", "error": str(exc)})
    directory = Path(output_directory) / variant / f"seed_{seed}"
    atomic_torch_save(directory / "mqar_best.pt", {"model_state": deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()}), "optimizer_state": optimizer.state_dict(), "updates": update, "mqar_batches": mqar_seen})
    result = {"variant": variant, "seed": seed, "protocol": "80pct_mqar_20pct_language", "updates": update, "mqar_batches": mqar_seen, "elapsed_sec": elapsed, "language_test_ce_after": language["ce"], "language_test_ppl_after": language["ppl"], "mqar": mqar, "history": history}; write_result(directory / "mqar_result.json", result); return result


__all__ = ["CURRICULUM_LENGTHS", "TEST_LENGTHS", "delayed_mqar_batch", "evaluate_mqar", "specialize_and_evaluate", "specialize_memory_heavy", "MEMORY_HEAVY_CURRICULUM"]
