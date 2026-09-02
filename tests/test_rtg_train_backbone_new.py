import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.rtg_train_backbone import (
    train_backbone,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_training_data import (
    BehavioralEpisode,
)


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 4)
        self.output = nn.Linear(4, 256)

    def forward(self, input_ids):
        return {"logits": self.output(self.embedding(input_ids)), "loss": torch.tensor(float("nan"))}


def test_backbone_uses_explicit_ce_and_small_explicit_updates():
    episodes = (BehavioralEpisode("a", (1, 2, 3, 4)), BehavioralEpisode("b", (4, 3, 2, 1)))
    result = train_backbone(ToyLM(), episodes, 29, updates=2, batch_size=4)
    assert result.terminal_update == 2
    assert len(result.losses) == 2
    assert all(torch.isfinite(torch.tensor(result.losses)))
    assert not result.model.training
