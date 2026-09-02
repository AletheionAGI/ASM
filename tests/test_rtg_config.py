from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from aletheion_state_models.benchmarks.transition_risk.rtg_config import (
    POST_RTG_DRM_EXTENSION_FIELDS,
    TRAINING_SEEDS,
    load_registered_config,
    verify_preregistration,
)

ROOT = Path(__file__).parents[1]


def test_frozen_manifest_and_all_literal_configs_validate():
    assert len(verify_preregistration(ROOT)) == 64
    for kind in ("asm", "transformer"):
        for seed in TRAINING_SEEDS:
            config = load_registered_config(ROOT, kind, seed)
            path = ROOT / (f"configs/rtg_asm_30k_seed{seed}.yaml" if kind == "asm" else f"transformer/rtg_transformer_30k_seed{seed}.yaml")
            expected = {item.name for item in fields(config)}
            if kind == "asm":
                expected -= POST_RTG_DRM_EXTENSION_FIELDS
                assert config.asm_z_eta == 0.1
                assert config.asm_z_lambda == 0.01
                assert config.asm_z_metric_d_min == 0.1
                assert config.asm_z_metric_d_max == 2.0
                assert config.asm_z_metric_u_bound == 1.0
            assert set(yaml.safe_load(path.read_text())) == expected


def test_config_rejects_unregistered_seed():
    with pytest.raises(ValueError, match="unregistered"):
        load_registered_config(ROOT, "asm", 1)
