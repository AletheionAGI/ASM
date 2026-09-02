import pytest
import torch

from aletheion_state_models.benchmarks.transition_risk.dataset import encode_frame
from aletheion_state_models.benchmarks.transition_risk.rtg_encoding import (
    encode_all_candidates,
    fixed_encode,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_types import NONSTOP_ACTIONS
from world_model.hazard_world import HazardWorld
from world_model.hazard_world_types import HazardWorldConfig


def toy_config():
    return HazardWorldConfig(
        world_id="toy-encoding",
        seed=11,
        traps=((0, 8), (8, 0), (8, 8)),
        moving_hazards=((7, 7),),
        hazard_velocities=((0, 0),),
        initial_agent=(4, 4),
        goal=(6, 6),
        forcing=0.0,
        sensor_noise=0.0,
    )


def test_fixed_encode_is_lsb_first_signed_float32():
    encoded = fixed_encode((1, 2, 128, 0))
    assert encoded.shape == (32,) and encoded.dtype == torch.float32
    assert encoded[:8].tolist() == [1.0] + [-1.0] * 7
    assert encoded[8:16].tolist() == [-1.0, 1.0] + [-1.0] * 6
    assert encoded[16:24].tolist() == [-1.0] * 7 + [1.0]
    assert encoded[24:].tolist() == [-1.0] * 8


@pytest.mark.parametrize("bad", [(1, 2, 3), (1, 2, 3, 256), (1, 2, 3, 4.0)])
def test_fixed_encode_rejects_non_bytes(bad):
    with pytest.raises(ValueError):
        fixed_encode(bad)


def test_six_candidate_frames_share_observation_and_frozen_order():
    world = HazardWorld(toy_config())
    candidates = encode_all_candidates(world.observe())
    expected = tuple(
        encode_frame(world.observe(), action) for action in NONSTOP_ACTIONS
    )
    assert tuple(item.frame for item in candidates) == expected
    assert len(candidates) == 6
    assert len({item.frame for item in candidates}) == 6
    assert all(item.frame == tuple(item.frame) for item in candidates)
