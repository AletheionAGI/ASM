from __future__ import annotations

from scripts.plot_asm_c2_fw_lm_confirmation import COLORS


def test_confirmation_chart_colors_are_distinct() -> None:
    assert len(COLORS) == len(set(COLORS.values()))
    assert set(COLORS) == {"ASM-C2-FW-LM", "ASM-R", "Transformer"}
