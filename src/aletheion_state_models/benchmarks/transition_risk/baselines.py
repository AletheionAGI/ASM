"""Architecture-independent persistence, Markov, and linear-Gaussian ATTR controls."""

from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import torch
from .dataset import FRAME_WIDTH, HazardEpisode
from .metrics import BasicRiskMetrics, basic_risk_metrics


@dataclass(frozen=True)
class ControlResults:
    persistence_next_state_mse: float
    kalman_next_state_mse: float
    markov_h8: BasicRiskMetrics


class MarkovRiskBaseline:
    """Laplace-smoothed tabular hazard risk using only the current observed frame."""

    def __init__(self) -> None:
        self._counts: dict[tuple[int, ...], list[int]] = {}
        self.prevalence = 0.5

    def fit(
        self, episodes: list[HazardEpisode], horizon_index: int = 2
    ) -> "MarkovRiskBaseline":
        counts: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0, 0])
        positives = total = 0
        for episode in episodes:
            frames = episode.input_ids.reshape(-1, FRAME_WIDTH)
            for frame, label in zip(frames, episode.hazard_labels[:, horizon_index]):
                value = int(label.item())
                counts[tuple(frame.tolist())][value] += 1
                positives += value
                total += 1
        if not total:
            raise ValueError("cannot fit Markov baseline without transitions")
        self._counts = dict(counts)
        self.prevalence = positives / total
        return self

    def predict(self, episode: HazardEpisode) -> torch.Tensor:
        values = []
        for frame in episode.input_ids.reshape(-1, FRAME_WIDTH):
            counts = self._counts.get(tuple(frame.tolist()))
            values.append(
                self.prevalence
                if counts is None
                else (counts[1] + 1) / (sum(counts) + 2)
            )
        return torch.tensor(values, dtype=torch.float32)


class KalmanBaseline:
    """One-step linear-Gaussian Kalman prior with fitted process scale."""

    def __init__(self, ridge: float = 1e-3) -> None:
        self.ridge = ridge
        self.weight: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    @staticmethod
    def _pairs(episodes: list[HazardEpisode]) -> tuple[torch.Tensor, torch.Tensor]:
        current = [
            episode.next_states[:-1]
            for episode in episodes
            if len(episode.next_states) > 1
        ]
        following = [
            episode.next_states[1:]
            for episode in episodes
            if len(episode.next_states) > 1
        ]
        if not current:
            raise ValueError("Kalman control needs at least two states per episode")
        x = torch.cat(current)
        return torch.cat([x, torch.ones(x.shape[0], 1)], dim=1), torch.cat(following)

    def fit(self, episodes: list[HazardEpisode]) -> "KalmanBaseline":
        x, y = self._pairs(episodes)
        identity = torch.eye(x.shape[1], dtype=x.dtype)
        identity[-1, -1] = 0
        self.weight = torch.linalg.solve(x.T @ x + self.ridge * identity, x.T @ y)
        residual = y - x @ self.weight
        self.scale = residual.square().mean(0).sqrt().clamp_min(1e-4)
        return self

    def predict(
        self, episodes: list[HazardEpisode]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.weight is None or self.scale is None:
            raise RuntimeError("baseline must be fit first")
        x, _ = self._pairs(episodes)
        mean = x @ self.weight
        return mean, self.scale.expand_as(mean)


def evaluate_controls(
    train: list[HazardEpisode], validation: list[HazardEpisode]
) -> ControlResults:
    persistence_predictions = []
    persistence_targets = []
    for episode in validation:
        if len(episode.next_states) > 1:
            persistence_predictions.append(episode.next_states[:-1])
            persistence_targets.append(episode.next_states[1:])
    persistence_mse = float(
        torch.cat(persistence_predictions)
        .sub(torch.cat(persistence_targets))
        .square()
        .mean()
    )
    kalman = KalmanBaseline().fit(train)
    kalman_mean, _ = kalman.predict(validation)
    kalman_targets = torch.cat(
        [
            episode.next_states[1:]
            for episode in validation
            if len(episode.next_states) > 1
        ]
    )
    kalman_mse = float(kalman_mean.sub(kalman_targets).square().mean())
    markov = MarkovRiskBaseline().fit(train)
    labels = []
    scores = []
    for episode in validation:
        labels.extend(episode.hazard_labels[:, 2].tolist())
        scores.extend(markov.predict(episode).tolist())
    return ControlResults(
        persistence_mse, kalman_mse, basic_risk_metrics(labels, scores)
    )


__all__ = [
    "ControlResults",
    "KalmanBaseline",
    "MarkovRiskBaseline",
    "evaluate_controls",
]
