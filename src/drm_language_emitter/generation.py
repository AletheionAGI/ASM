from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import DRMEmitterModel


@torch.no_grad()
def generate(
    model: DRMEmitterModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float | None = None,
    top_k: int | None = None,
) -> torch.Tensor:
    model.eval()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    config = model.config
    temperature = config.generation_temperature if temperature is None else temperature
    top_k = config.top_k if top_k is None else top_k
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one prompt token")
    state = model.init_inference_state(input_ids.shape[0], input_ids.device)
    logits, state = model.prefill(input_ids, state)
    generated = [input_ids]
    next_logits = logits[:, -1]
    for index in range(max_new_tokens):
        sampling_logits = next_logits / max(temperature, 1e-6)
        if top_k and top_k > 0 and top_k < sampling_logits.shape[-1]:
            values, indices = torch.topk(sampling_logits, top_k, dim=-1)
            filtered = torch.full_like(sampling_logits, float("-inf"))
            sampling_logits = filtered.scatter(-1, indices, values)
        probs = F.softmax(sampling_logits, dim=-1)
        current = torch.multinomial(probs, num_samples=1).squeeze(-1)
        generated.append(current[:, None])
        if index + 1 < max_new_tokens:
            next_logits, state = model.decode_step(current, state)
    return torch.cat(generated, dim=1)
