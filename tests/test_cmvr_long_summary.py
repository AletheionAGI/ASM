from copy import deepcopy
from aletheion_state_models.benchmarks.cmvr_long_curriculum import (
    TEST_LENGTHS,
    arm_passed,
)
from aletheion_state_models.benchmarks.cmvr_long_summary import summarize


def _result(arm, seed=17):
    tests = [
        {
            "length": length,
            "accuracy": 1.0,
            "ce": 0.1,
            "ce_finite": True,
            "mean_rank": 32.0,
        }
        for length in TEST_LENGTHS
    ]
    streams = [
        {"length": length, "tokens_per_second": 100.0, "retained_state_bytes": 66112}
        for length in TEST_LENGTHS[1:]
    ]
    return {
        "arm": arm,
        "seed": seed,
        "test": tests,
        "streaming": streams,
        "streaming_error": 1e-6,
        "finite": True,
        "no_read": {"accuracy": 0.01},
        "no_write": {"accuracy": 0.01},
        "parameters_total": 100,
        "parameters_trainable": 90,
        "controller_gradient_hits": 1,
        "history": [{"rank_std": 1.0}],
    }


def test_nonfinite_test_and_failed_stream_fail_closed():
    row = _result("cm_vr_fixed32")
    row["test"][-1].update(accuracy=0.0, ce=None, ce_finite=False)
    row["streaming"][-1].update(status="failed", error="non-finite")
    assert not arm_passed(row)


def test_summary_preserves_failed_accuracy_and_classifies_legacy_finite_flag():
    results = []
    for arm in ("cm_vr_full64", "cm_vr_fixed32", "cm_vr_adaptive32"):
        for seed in (17, 29, 43):
            row = _result(arm, seed)
            if arm == "cm_vr_fixed32" and seed == 29:
                row["test"][-1].update(accuracy=0.0, ce=None, ce_finite=False)
            results.append(deepcopy(row))
    summary = summarize(results, (17, 29, 43))
    fixed = summary["arms"]["cm_vr_fixed32"]
    assert [row["finite"] for row in fixed["runs"]] == [True, False, True]
    assert fixed["by_length"]["32768"]["accuracy_mean"] == 2 / 3
    assert fixed["by_length"]["32768"]["accuracy_mean_successful"] == 1.0
    assert fixed["by_length"]["32768"]["failed"] == 1
    assert not summary["gates"]["fixed_long_gate_all_seeds"]
