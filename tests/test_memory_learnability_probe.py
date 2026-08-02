import importlib.util
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location(
    "probe_addressable_memory",
    Path(__file__).parents[1] / "scripts" / "probe_addressable_memory.py",
)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def fixed_mqar_example() -> torch.Tensor:
    return torch.tensor([[2, 34, 3, 35, 1, 2, 34, 1, 3]], dtype=torch.long)


def test_token_roles_distinguish_pair_values_and_query_keys():
    pair_value, query_key, key_ids = PROBE.token_roles(fixed_mqar_example())
    assert pair_value.nonzero().tolist() == [[0, 1], [0, 3]]
    assert query_key.nonzero().tolist() == [[0, 5], [0, 8]]
    assert key_ids[0, 5].item() == 0
    assert key_ids[0, 8].item() == 1


def test_oracle_slot_recalls_values_at_query_positions():
    logits, diagnostics = PROBE.OracleSlotProbe()(fixed_mqar_example())
    predictions = logits.argmax(-1)
    assert predictions[0, 5].item() == 34
    assert predictions[0, 8].item() == 35
    assert diagnostics["writes_per_example"].item() == 2
    assert diagnostics["reads_per_example"].item() == 2


def test_fast_weight_probe_is_fixed_capacity_and_differentiable():
    model = PROBE.FastWeightProbe(d_value=8)
    logits, diagnostics = model(fixed_mqar_example())
    assert logits.shape == (1, 9, PROBE.VOCAB_SIZE)
    logits[:, [5, 8]].sum().backward()
    assert model.value_embedding.weight.grad is not None
    assert model.emitter.weight.grad is not None
    assert torch.isfinite(diagnostics["memory_norm"])
