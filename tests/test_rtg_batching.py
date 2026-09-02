import numpy as np
import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_batching import (
    make_batch_plan,
    materialize_batches,
)


def test_backbone_plan_is_one_pcg64_permutation_cycled_without_reshuffle():
    plan = make_batch_plan(5, 29, "backbone", updates=3)
    expected = tuple(int(value) for value in np.random.Generator(np.random.PCG64(40_029)).permutation(5))
    assert plan.permutation == expected
    assert tuple(value for batch in plan.batches for value in batch) == tuple(expected[index % 5] for index in range(12))
    assert materialize_batches(tuple("abcde"), plan)[0] == tuple("abcde"[index] for index in plan.batches[0])


def test_auxiliary_plan_requires_registered_batch_size():
    with pytest.raises(ValueError, match="must be 64"):
        make_batch_plan(100, 43, "auxiliary", batch_size=4)
