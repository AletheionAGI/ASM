import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.training import evaluate_ce


def test_evaluate_ce_preserves_eval_mode() -> None:
    model = DRMEmitterModel(
        DRMConfig(
            vocab_size=17,
            d_token=8,
            d_state=12,
            n_directions=4,
            metric_rank=2,
            hidden_size=16,
            max_seq_len=4,
        )
    )
    model.eval()

    evaluate_ce(model, list(range(12)), seq_len=4, device=torch.device("cpu"))

    assert not model.training


def test_evaluate_ce_preserves_train_mode() -> None:
    model = DRMEmitterModel(
        DRMConfig(
            vocab_size=17,
            d_token=8,
            d_state=12,
            n_directions=4,
            metric_rank=2,
            hidden_size=16,
            max_seq_len=4,
        )
    )
    model.train()

    evaluate_ce(model, list(range(12)), seq_len=4, device=torch.device("cpu"))

    assert model.training
