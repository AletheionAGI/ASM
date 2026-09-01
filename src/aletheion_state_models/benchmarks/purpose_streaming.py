"""Bounded-state streaming probes for ASM-CM and ASM-VR-S."""
from __future__ import annotations
from dataclasses import fields, is_dataclass
import time
import numpy as np
import torch


def tensor_bytes(value) -> int:
    if torch.is_tensor(value): return value.numel() * value.element_size()
    if is_dataclass(value): return sum(tensor_bytes(getattr(value, item.name)) for item in fields(value))
    if isinstance(value, (tuple, list)): return sum(tensor_bytes(item) for item in value)
    if isinstance(value, dict): return sum(tensor_bytes(item) for item in value.values())
    return 0


def _sync(device):
    if device.type == "cuda": torch.cuda.synchronize(device)


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.inference_mode()
def probe_streaming(model, lengths, *, seed, device="cuda", prompt_tokens=64, chunk=128):
    """Decode one continuous random stream and measure retained state by length."""
    torch_device = torch.device(device); model.to(torch_device).eval(); generator = torch.Generator(device=torch_device).manual_seed(seed)
    prompt = torch.randint(0, model.config.vocab_size, (1, prompt_tokens), generator=generator, device=torch_device)
    with _autocast(torch_device): _, state = model.prefill(prompt)
    _sync(torch_device); rows = []; position = prompt_tokens
    if torch_device.type == "cuda": torch.cuda.reset_peak_memory_stats(torch_device)
    for target in lengths:
        chunk_times = []; segment_start = time.perf_counter(); chunk_start = segment_start; chunk_tokens = 0
        try:
            while position < target:
                token = torch.randint(0, model.config.vocab_size, (1,), generator=generator, device=torch_device)
                with _autocast(torch_device): _, state = model.decode_step(token, state)
                position += 1; chunk_tokens += 1
                if chunk_tokens == chunk or position == target:
                    _sync(torch_device); now = time.perf_counter(); chunk_times.append((now - chunk_start) / chunk_tokens); chunk_start = now; chunk_tokens = 0
        except (RuntimeError, ValueError) as exc:
            rows.append({"length": target, "status": "failed", "failed_at_position": position, "error": str(exc)})
            print({"streaming": rows[-1]}, flush=True); break
        elapsed = time.perf_counter() - segment_start; values = np.asarray(chunk_times)
        rows.append({"length": target, "segment_tokens": target - (rows[-1]["length"] if rows else prompt_tokens), "segment_elapsed_sec": elapsed, "tokens_per_second": (target - (rows[-1]["length"] if rows else prompt_tokens)) / elapsed, "latency_ms_p50": float(np.quantile(values, .5) * 1000), "latency_ms_p95": float(np.quantile(values, .95) * 1000), "retained_state_bytes": tensor_bytes(state), "cuda_peak_mb": torch.cuda.max_memory_allocated(torch_device) / 2**20 if torch_device.type == "cuda" else 0.0, "open_block_tokens": int(state.block_tokens.shape[1]) if state.block_tokens is not None else 0})
        print({"streaming": rows[-1]}, flush=True)
        if torch_device.type == "cuda": torch.cuda.reset_peak_memory_stats(torch_device)
    return rows


__all__ = ["probe_streaming", "tensor_bytes"]
