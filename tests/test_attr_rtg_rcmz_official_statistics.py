"""Goldens proving official inference starts from raw hierarchical cells."""

from __future__ import annotations

import numpy as np
import pytest

from attr_rtg_rcmz.constants import ARMS, REGIMES, TRAINING_SEEDS
from attr_rtg_rcmz.official_contrasts import ENDPOINTS, contrast_rows
from attr_rtg_rcmz.official_stats import _fold_origin_tuples


def _rows():
    offsets = {"R": 0.0, "CM": -1.0, "Z": 2.0, "T": 3.0}
    rows = []
    for arm in ARMS:
        for seed in TRAINING_SEEDS:
            for regime in REGIMES:
                sufficient = {
                    name: [{"world": 0, "episode": 0, "value": offsets[arm]}]
                    for name in ENDPOINTS
                }
                # Deliberately contradictory published summaries must never be bootstrapped.
                row = {
                    "status": "VALID",
                    "arm": arm,
                    "seed": seed,
                    "regime": regime,
                    "_sufficient": sufficient,
                }
                row.update({name: -offsets[arm] * 999 for name in ENDPOINTS})
                rows.append(row)
    return rows


def test_official_contrasts_use_raw_cells_and_exact_six_family():
    rows = contrast_rows(_rows())
    assert [row["contrast"] for row in rows] == [
        "CM-R",
        "CM-Z",
        "CM-T",
        "R-Z",
        "R-T",
        "Z-T",
    ]
    cm_r = rows[0]
    assert np.allclose(cm_r["lower"], np.full((3, 5), -1.0))
    assert np.allclose(cm_r["upper"], np.full((3, 5), -1.0))
    assert not cm_r["passed"]  # service/coverage -1 violate the lower gates


def test_calibration_fold_is_origin_then_episode_then_world():
    # World zero has three origins at 0; world one has one origin at 10.
    rows = [(0, 0, 0.0), (0, 0, 0.0), (0, 0, 0.0), (1, 0, 10.0)]
    assert _fold_origin_tuples(rows) == pytest.approx(5.0)


def test_raw_cell_mismatch_fails_closed():
    rows = _rows()
    rows[0]["_sufficient"]["ece"][0]["episode"] = 1
    with pytest.raises(RuntimeError, match="canonical cells differ"):
        contrast_rows(rows)


def test_invalid_arm_only_blocks_dependent_contrasts():
    rows = _rows()
    for row in rows:
        if row["arm"] == "CM" and row["seed"] == 29 and row["regime"] == "ID":
            row.update(status="INVALID", reason="nonfinite")
            row.pop("_sufficient")
    results = contrast_rows(rows)
    dependent = [row for row in results if "CM" in row["contrast"].split("-")]
    autonomous = [row for row in results if "CM" not in row["contrast"].split("-")]
    assert all(row["status"] == "INVALID" and not row["passed"] for row in dependent)
    assert all(row["status"] == "VALID" for row in autonomous)


def test_lp_seed_encoding_is_exact_and_canonical():
    from attr_rtg_rcmz.bootstrap import bootstrap_seed64
    from attr_rtg_rcmz.policy import derive_seed64

    assert bootstrap_seed64() == derive_seed64("bootstrap", 0) == 16043913454067942795
    with pytest.raises(ValueError, match="uint64"):
        derive_seed64("bootstrap", True)
    with pytest.raises(ValueError, match="uint64"):
        derive_seed64("bootstrap", -1)


def test_invalid_scalar_row_is_complete_and_json_safe():
    import json

    from attr_rtg_rcmz.official_stats import invalid_row

    row = invalid_row(
        arm="R", seed=29, regime="ID", reason="nonfinite", peak_bytes=7, elapsed=1.5
    )
    assert row["status"] == "INVALID" and row["invalid_reason"] == "nonfinite"
    for name in (
        "h8_nll",
        "ece",
        "unsafe_selection",
        "safe_service",
        "coverage",
        "abstention",
    ):
        assert name in row and row[name] is None
    assert "NaN" not in json.dumps(row, allow_nan=False)
