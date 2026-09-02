"""Load and flatten the sealed ATTR-RTG registered summary."""
from __future__ import annotations

import csv
import json
from pathlib import Path

REGIMES = ("test_id", "test_shift", "test_ood")
MODELS = ("asm", "transformer")
HEADS = ("G", "C")
LABELS = {"test_id": "ID", "test_shift": "Shift", "test_ood": "OOD", "asm": "ASM-X", "transformer": "Transformer"}


def load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") != "attr_rtg_registered_summary" or payload.get("schema_version") != 1:
        raise ValueError("expected attr_rtg_registered_summary schema version 1")
    evidence = payload.get("summary", {}).get("evidence", {})
    if set(evidence.get("architecture", {})) != set(REGIMES):
        raise ValueError("registered summary does not contain the three expected regimes")
    return payload


def _aggregate_rows(evidence: dict) -> list[dict]:
    rows = []
    for regime in REGIMES:
        values = evidence["architecture"][regime]
        for model in MODELS:
            suffix = "asm" if model == "asm" else "transformer"
            row = {"regime": LABELS[regime], "model": LABELS[model]}
            for metric in ("nmse", "nll"):
                row[metric] = values[f"{metric}_{suffix}"]
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = values[f"{metric}_{suffix}_ci95"]
            rows.append(row)
    return rows


def _governance_rows(evidence: dict) -> list[dict]:
    rows = []
    for regime in REGIMES:
        for model in MODELS:
            for head in HEADS:
                values = evidence["absolute"][regime][model][head]
                row = {"regime": LABELS[regime], "model": LABELS[model], "head": head}
                for metric in ("unsafe_rate", "safe_service", "coverage"):
                    row[metric] = values[metric]
                    row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = values[f"{metric}_ci95"]
                rows.append(row)
    return rows


def _comparison_rows(evidence: dict) -> list[dict]:
    rows = []
    for model in MODELS:
        for regime in REGIMES:
            values = evidence["versus_c"][model][regime]
            row = {"regime": LABELS[regime], "model": LABELS[model]}
            for metric in ("delta_safety", "delta_safe_service", "coverage_difference"):
                row[metric] = values[metric]
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = values[f"{metric}_ci95"]
            rows.append(row)
    return rows


def _seed_rows(evidence: dict) -> list[dict]:
    rows = []
    for regime in REGIMES:
        values = evidence["architecture"][regime]
        for seed, seed_values in sorted(values["per_seed"].items(), key=lambda item: int(item[0])):
            rows.append({"regime": LABELS[regime], "seed": int(seed), **seed_values})
    return rows


def dashboard_data(payload: dict) -> dict:
    evidence = payload["summary"]["evidence"]
    gates = [{"gate": key, "passed": value} for key, value in sorted(payload["summary"]["gates"].items())]
    return {"architecture": _aggregate_rows(evidence), "governance": _governance_rows(evidence),
            "g_vs_c": _comparison_rows(evidence), "seeds": _seed_rows(evidence), "gates": gates}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_tables(output: Path, tables: dict) -> None:
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", rows)
    (output / "dashboard_data.json").write_text(json.dumps(tables, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
