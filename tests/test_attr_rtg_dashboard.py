"""Tests for the registered-summary-only ATTR-RTG dashboard."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.attr_rtg_dashboard.charts import render_all
from scripts.attr_rtg_dashboard.data import dashboard_data, export_tables, load_summary
from scripts.attr_rtg_dashboard.html import render_html

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "docs/benchmarks/attr_rtg/registered_summary.json"


def test_registered_summary_is_flattened_without_recalculation() -> None:
    payload = load_summary(SOURCE)
    tables = dashboard_data(payload)
    assert len(tables["gates"]) == 33
    assert [row["gate"] for row in tables["gates"] if row["passed"]] == ["Transformer.RTG1-Z"]
    id_asm = next(row for row in tables["architecture"] if row["regime"] == "ID" and row["model"] == "ASM-X")
    assert id_asm["nmse"] == payload["summary"]["evidence"]["architecture"]["test_id"]["nmse_asm"]
    assert {row["seed"] for row in tables["seeds"]} == {29, 43, 71, 89, 107}


def test_renderer_emits_offline_assets_and_cautions(tmp_path: Path) -> None:
    payload = load_summary(SOURCE)
    tables = dashboard_data(payload)
    export_tables(tmp_path, tables)
    render_all(tables, tmp_path)
    render_html(payload, tables, tmp_path)
    expected = {"index.html", "dashboard_data.json", "architecture.csv", "governance.csv", "g_vs_c.csv",
                "seeds.csv", "gates.csv", "architecture_quality.png", "architecture_quality.svg",
                "governance.png", "governance.svg", "g_vs_c.png", "g_vs_c.svg",
                "seed_differences.png", "seed_differences.svg"}
    assert {path.name for path in tmp_path.iterdir()} == expected
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "ASM-X vs Transformer" in html
    assert "não demonstra segurança, nem superioridade universal" in html
    assert "C−G unsafe" in html and "G−C safe-service" in html
    assert "1/33" in html and "Transformer.RTG1-Z" in html
    assert "runs/attr_rtg" not in html
    derived = json.loads((tmp_path / "dashboard_data.json").read_text(encoding="utf-8"))
    assert len(derived["governance"]) == 12
