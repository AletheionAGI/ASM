"""Seed-17 structural and MQAR smoke for ASM-CM-VR fixed-32."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import time
import torch
from torch.nn import functional as F
from .phase3a_checkpoint import atomic_torch_save, write_result
from .phase3a_training import measure_streaming_error
from .phase3a_variants import phase3a_config
from .purpose_mqar import delayed_mqar_batch, evaluate_mqar
from .purpose_streaming import probe_streaming
from aletheion_state_models.variants import build_compact_memory_variable_rank

ARMS = {"cm_vr_full64": 64, "cm_vr_fixed32": 32}


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def train_short_mqar(arm, *, seed, steps, output_root, device):
    """Train one frozen-rank arm on the same causal short-MQAR stream."""
    rank = ARMS[arm]; torch.manual_seed(seed); model = build_compact_memory_variable_rank(phase3a_config(seed), fixed_rank=rank).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=.01); generator = torch.Generator().manual_seed(seed); history = []; started = time.perf_counter(); model.train()
    milestones = {1, 100, 250, 500, 750, steps}
    for step in range(1, steps + 1):
        x, y, mask = delayed_mqar_batch(40, 4, generator, device); optimizer.zero_grad(set_to_none=True)
        with _autocast(device): selected = model(x, collect_diagnostics=False, global_step=step)["logits"][mask]; loss = F.cross_entropy(selected.float(), y[mask])
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss for {arm} at {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        if step in milestones:
            model.eval(); validation = evaluate_mqar(model, 40, seed=60_000 + seed, batches=8, device=device); model.train(); row = {"step": step, "train_ce": float(loss.detach()), "validation_ce": validation["ce"], "validation_accuracy": validation["accuracy"]}; history.append(row); print({"arm": arm, **row}, flush=True)
    elapsed = time.perf_counter() - started; model.eval(); test = evaluate_mqar(model, 40, seed=70_000 + seed, batches=16, device=device)
    memory = model.addressable_memory; read_enabled, write_enabled = memory.read_enabled, memory.write_enabled
    memory.read_enabled = False; no_read = evaluate_mqar(model, 40, seed=70_000 + seed, batches=16, device=device); memory.read_enabled = read_enabled
    memory.write_enabled = False; no_write = evaluate_mqar(model, 40, seed=70_000 + seed, batches=16, device=device); memory.write_enabled = write_enabled
    corpus = torch.arange(256, dtype=torch.uint8).repeat(8); parity = measure_streaming_error(model, corpus, device)
    streaming = probe_streaming(model, (512, 4096), seed=80_000 + seed, device=str(device))
    directory = Path(output_root) / arm; checkpoint = {"model_state": deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()}), "optimizer_state": optimizer.state_dict(), "seed": seed, "rank": rank, "steps": steps}; atomic_torch_save(directory / "final.pt", checkpoint)
    result = {"arm": arm, "seed": seed, "logical_rank": rank, "steps": steps, "parameters_total": sum(p.numel() for p in model.parameters()), "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad), "elapsed_sec": elapsed, "history": history, "test": test, "no_read": no_read, "no_write": no_write, "streaming_error": parity, "streaming": streaming, "finite": all(math.isfinite(value) for value in (float(loss.detach()), parity))}; write_result(directory / "result.json", result); return result


def run_phase1(*, seed=17, steps=1000, run_root="runs/asm_cm_vr_fixed32_phase1", device="cuda"):
    torch_device = torch.device(device); results = [train_short_mqar(arm, seed=seed, steps=steps, output_root=run_root, device=torch_device) for arm in ARMS]
    full, fixed = results; gates = {"parameter_match_trainable": full["parameters_trainable"] == fixed["parameters_trainable"], "fixed_rank_exact": fixed["logical_rank"] == 32, "fixed_short_mqar_95pct": fixed["test"]["accuracy"] >= .95, "fixed_memory_read_required": fixed["no_read"]["accuracy"] <= fixed["test"]["accuracy"] - .50, "fixed_memory_write_required": fixed["no_write"]["accuracy"] <= fixed["test"]["accuracy"] - .50, "full_streaming_parity": full["streaming_error"] <= 1e-4, "fixed_streaming_parity": fixed["streaming_error"] <= 1e-4, "fixed_streaming_4k_complete": any(row.get("length") == 4096 and row.get("status") != "failed" for row in fixed["streaming"])}
    return {"experiment": "ASM-CM-VR fixed-32 Phase 1", "seed": seed, "arms": results, "gates": gates, "passed": all(gates.values())}


__all__ = ["ARMS", "run_phase1", "train_short_mqar"]
