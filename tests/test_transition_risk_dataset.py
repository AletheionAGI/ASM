from dataclasses import replace
import torch
from aletheion_state_models.benchmarks.transition_risk.dataset import (
    FRAME_WIDTH,
    MODEL_FEATURES,
    collate_episodes,
    encode_frame,
    gather_step_representations,
    make_episodes,
    make_worlds,
    rollout_episode,
)
from aletheion_state_models.benchmarks.transition_risk.leakage import audit_leakage
from world_model.hazard_world import HazardWorld


def test_world_and_episode_generation_is_reproducible():
    worlds = make_worlds(3, 17)
    assert worlds == make_worlds(3, 17)
    assert len({world.world_id for world in worlds}) == 3
    left = rollout_episode(worlds[0], 99)
    right = rollout_episode(worlds[0], 99)
    assert left.actions == right.actions
    torch.testing.assert_close(left.input_ids, right.input_ids)
    torch.testing.assert_close(left.hazard_labels, right.hazard_labels)


def test_fixed_frames_contain_only_observed_causal_features():
    world = HazardWorld(make_worlds(1, 4)[0])
    frame = encode_frame(world.observe(), "BRAKE")
    assert len(frame) == FRAME_WIDTH and all(0 <= token < 256 for token in frame)
    assert (
        "hidden_mode" not in MODEL_FEATURES
        and "failure_countdown" not in MODEL_FEATURES
    )
    assert audit_leakage(MODEL_FEATURES).passed


def test_future_labels_and_targets_align_without_terminal_padding():
    episode = rollout_episode(
        replace(make_worlds(1, 12)[0], traps=((0, 1),), initial_agent=(0, 0)), 7
    )
    steps = len(episode.actions)
    assert episode.input_ids.numel() == steps * FRAME_WIDTH
    assert episode.next_states.shape == (steps, 6)
    assert episode.hazard_labels.shape == (steps, 4)
    assert episode.severity.shape == episode.time_to_hazard.shape == (steps,)
    assert episode.step_positions.tolist() == list(range(3, steps * 4, 4))


def test_collation_and_step_gather_keep_variable_episodes_aligned():
    episodes = make_episodes(make_worlds(2, 21), 2, 31)
    batch = collate_episodes(episodes)
    assert batch["input_ids"].shape[0] == 4
    assert batch["step_mask"].sum().item() == sum(
        len(item.actions) for item in episodes
    )
    representations = torch.randn(4, batch["input_ids"].shape[1], 11)
    gathered = gather_step_representations(representations, batch["step_positions"])
    assert gathered.shape == (*batch["step_positions"].shape, 11)
    for row, episode in enumerate(episodes):
        count = len(episode.actions)
        torch.testing.assert_close(
            gathered[row, :count], representations[row, episode.step_positions]
        )
