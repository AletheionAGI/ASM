import torch

from drm_language_emitter.mqar import make_mqar_batch


def test_mqar_targets_match_presented_associations():
    x, y, mask = make_mqar_batch(
        batch_size=4,
        n_pairs=5,
        n_queries=3,
        key_vocab_size=8,
        value_vocab_size=10,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )
    assert x.shape == y.shape == mask.shape
    assert torch.all(mask.sum(dim=1) == 3)
    for row in range(x.shape[0]):
        mapping = {
            int(x[row, index]): int(y[row, index])
            for index in range(0, 10, 2)
        }
        for position in mask[row].nonzero().flatten().tolist():
            assert int(y[row, position]) == mapping[int(x[row, position])]


def test_mqar_generation_is_deterministic():
    kwargs = dict(
        batch_size=2,
        n_pairs=4,
        n_queries=4,
        key_vocab_size=8,
        value_vocab_size=10,
        device=torch.device("cpu"),
    )
    first = make_mqar_batch(generator=torch.Generator().manual_seed(11), **kwargs)
    second = make_mqar_batch(generator=torch.Generator().manual_seed(11), **kwargs)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
