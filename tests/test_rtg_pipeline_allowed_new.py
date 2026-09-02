import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_pipeline_train import (
    run_training,
)


def test_production_runner_rejects_nonregistered_update_count_before_generation(tmp_path):
    with pytest.raises(ValueError, match="exactly 1000"):
        run_training(tmp_path, tmp_path / "output", updates=2)


def test_allowed_verifier_compares_canonical_bytes_not_tuple_container_types(
    tmp_path, monkeypatch
):
    from aletheion_state_models.benchmarks.transition_risk import rtg_pipeline_train
    from aletheion_state_models.benchmarks.transition_risk.rtg_artifacts import (
        atomic_write_json,
    )

    marker = (object(),)
    monkeypatch.setattr(rtg_pipeline_train, "prepare_allowed_data", lambda: marker)
    monkeypatch.setattr(
        rtg_pipeline_train,
        "build_allowed_manifest",
        lambda root, data: {"schema": "toy", "tuple_value": (1, 2)},
    )
    atomic_write_json(
        tmp_path / "allowed_manifest.json",
        {"schema": "toy", "tuple_value": [1, 2]},
    )
    assert rtg_pipeline_train._verified_allowed_data(tmp_path, tmp_path) is marker
