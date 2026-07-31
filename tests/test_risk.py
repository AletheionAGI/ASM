import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.risk import RiskField


def test_risk_mass_is_clamped_when_enabled():
    config = DRMConfig(
        d_state=4,
        hidden_size=8,
        use_powerlaw_risk=True,
        risk_mass_max=0.05,
        risk_alpha_max=10.0,
    )
    risk = RiskField(config)
    risk.alpha_b.data.fill_(100.0)
    risk.alpha_d.data.fill_(100.0)
    out = risk(torch.randn(3, 4))
    assert torch.all(out["risk_mass"] <= 0.05)
    assert torch.all(out["risk_mass_raw"] >= out["risk_mass"])


def test_disabled_risk_can_omit_all_parameters():
    config = DRMConfig(
        d_state=4,
        hidden_size=8,
        use_powerlaw_risk=False,
        instantiate_disabled_risk=False,
    )
    risk = RiskField(config)
    out = risk(torch.randn(3, 4))
    assert sum(parameter.numel() for parameter in risk.parameters()) == 0
    assert torch.count_nonzero(out["risk_mass"]) == 0
