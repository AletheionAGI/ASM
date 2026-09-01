from aletheion_state_models.benchmarks.transition_risk import (
    HardShield,
    HazardPrediction,
    run_cloned_intervention,
)


class Simulator:
    def __init__(self):
        self.value = 0

    def clone(self):
        other = Simulator()
        other.value = self.value
        return other

    def step(self, action, noise):
        self.value += action + noise
        return self.value


def test_hard_shield_selects_best_safe_action_and_fails_closed():
    shield = HardShield(0.2)
    candidates = [
        HazardPrediction("fast", 0.2, 0.3, 10),
        HazardPrediction("slow", 0.1, 0.15, 2),
        HazardPrediction("brake", 0.05, 0.1, 1),
    ]
    decision = shield.select(candidates)
    assert decision.action == "slow"
    assert decision.rejected_actions == ("fast",)
    assert HardShield(0.01).select(candidates).abstained


def test_cloned_intervention_reuses_noise_and_does_not_mutate_source():
    simulator = Simulator()
    result = run_cloned_intervention(simulator, 2, 0, [1, -1])
    assert result.intervention.trajectory == (3, 2)
    assert result.control.trajectory == (1, 0)
    assert simulator.value == 0
