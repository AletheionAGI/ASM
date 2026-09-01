import torch
from aletheion_state_models.benchmarks.transition_risk.baselines import (
    KalmanBaseline,
    MarkovRiskBaseline,
    evaluate_controls,
)
from aletheion_state_models.benchmarks.transition_risk.dataset import (
    make_episodes,
    make_worlds,
)


def _episodes(seed):
    return make_episodes(make_worlds(3, seed, max_steps=8), 2, seed)


def test_markov_baseline_fits_training_only_and_predicts_probabilities():
    train = _episodes(3)
    validation = _episodes(4)
    model = MarkovRiskBaseline().fit(train)
    scores = model.predict(validation[0])
    assert scores.shape == (len(validation[0].actions),)
    assert torch.all((scores >= 0) & (scores <= 1))


def test_kalman_shapes_and_control_evaluation():
    train = _episodes(5)
    validation = _episodes(6)
    mean, scale = KalmanBaseline().fit(train).predict(validation)
    assert mean.shape == scale.shape
    assert torch.all(scale > 0)
    result = evaluate_controls(train, validation)
    assert result.persistence_next_state_mse >= 0
    assert result.kalman_next_state_mse >= 0
    assert 0 <= result.markov_h8.auprc <= 1
