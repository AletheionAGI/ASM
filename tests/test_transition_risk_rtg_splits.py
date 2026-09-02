from pathlib import Path

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_seal import (
    TestOpenCapability,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_splits import (
    assert_disjoint_world_ids,
    make_allowed_worlds,
    make_test_worlds,
    split_spec,
)


def test_allowed_registry_and_test_barrier():
    train = make_allowed_worlds("train")
    validation = make_allowed_worlds("validation")
    calibration = make_allowed_worlds("calibration")
    assert (len(train), len(validation), len(calibration)) == (64, 16, 16)
    assert_disjoint_world_ids((train, validation, calibration))
    assert split_spec("test_ood").dynamic_family == "ood"
    with pytest.raises(PermissionError, match="capability"):
        make_allowed_worlds("test_id")
    with pytest.raises(TypeError, match="issued only"):
        TestOpenCapability("0" * 64, Path("receipt"), object())
    with pytest.raises(PermissionError, match="capability"):
        make_test_worlds("test_id", 360101, object(), "0" * 64)
