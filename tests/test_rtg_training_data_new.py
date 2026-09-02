import torch

from aletheion_state_models.benchmarks.transition_risk.rtg_training_data import (
    BehavioralEpisode,
    backbone_batch_indices,
    collate_ce_episodes,
)


def test_collate_excludes_episode_boundaries_final_bytes_and_padding():
    episodes = (BehavioralEpisode("a", (1, 2, 3)), BehavioralEpisode("b", (7, 8)))
    batch = collate_ce_episodes(episodes)
    assert batch.input_ids.shape == (2, 64)
    assert batch.targets[0, :4].tolist() == [2, 3, -100, -100]
    assert batch.targets[1, :3].tolist() == [8, -100, -100]
    assert torch.all(batch.targets[:, 3:] == -100)
    assert backbone_batch_indices(3, 29, updates=2) == backbone_batch_indices(3, 29, updates=2)
