from __future__ import annotations

import torch


def make_mqar_batch(
    batch_size: int,
    n_pairs: int,
    n_queries: int,
    key_vocab_size: int,
    value_vocab_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate multi-query associative-recall examples.

    Each sequence first presents random key/value pairs and then queries keys in
    a shuffled order. The boolean mask selects positions whose next-token
    target is the value associated with the queried key.
    """
    if not 0 < n_queries <= n_pairs <= key_vocab_size:
        raise ValueError("require 0 < n_queries <= n_pairs <= key_vocab_size")
    if batch_size <= 0 or value_vocab_size <= 0:
        raise ValueError("batch_size and value_vocab_size must be positive")

    query_token = 1
    key_offset = 2
    value_offset = key_offset + key_vocab_size
    sequences = []
    masks = []
    for _ in range(batch_size):
        keys = torch.randperm(key_vocab_size, generator=generator)[:n_pairs] + key_offset
        values = (
            torch.randint(0, value_vocab_size, (n_pairs,), generator=generator)
            + value_offset
        )
        query_indices = torch.randperm(n_pairs, generator=generator)[:n_queries]
        tokens: list[int] = []
        answer_positions: list[int] = []
        for key, value in zip(keys.tolist(), values.tolist()):
            tokens.extend([key, value])
        for index in query_indices.tolist():
            tokens.extend([query_token, int(keys[index]), int(values[index])])
            answer_positions.append(len(tokens) - 2)
        sequence = torch.tensor(tokens, dtype=torch.long)
        mask = torch.zeros(sequence.numel() - 1, dtype=torch.bool)
        mask[torch.tensor(answer_positions, dtype=torch.long)] = True
        sequences.append(sequence)
        masks.append(mask)

    stacked = torch.stack(sequences).to(device)
    return stacked[:, :-1], stacked[:, 1:], torch.stack(masks).to(device)
