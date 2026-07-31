import os

import torch

from scripts.rescore_drm_scaling_law import fit_power_law, observed_crossovers
from scripts.train_drm_memmap import link_checkpoint, save_checkpoint


def test_scaling_fit_and_observed_crossover():
    rows = []
    values = {
        "A": [3.0, 2.7, 2.4, 2.2],
        "B": [2.8, 2.65, 2.5, 2.4],
    }
    for variant, losses in values.items():
        for tokens, loss in zip((1_000_000, 2_000_000, 5_000_000, 10_000_000), losses):
            rows.append(
                {
                    "variant": variant,
                    "milestone_tokens": tokens,
                    "validation_ce": loss,
                }
            )

    fit = fit_power_law([row for row in rows if row["variant"] == "A"])
    assert fit is not None
    assert fit["alpha"] > 0
    crossovers = observed_crossovers(rows, ["A", "B"])
    assert len(crossovers) == 1
    assert 2_000_000 < crossovers[0]["estimated_crossover_tokens"] < 5_000_000


def test_atomic_checkpoint_and_hardlink_alias(tmp_path):
    checkpoint = tmp_path / "checkpoint_milestone_16.pt"
    alias = tmp_path / "checkpoint_latest.pt"
    save_checkpoint(checkpoint, {"value": torch.tensor([1.0])})
    link_checkpoint(checkpoint, alias)

    assert torch.equal(torch.load(alias, weights_only=True)["value"], torch.tensor([1.0]))
    assert os.stat(checkpoint).st_ino == os.stat(alias).st_ino
