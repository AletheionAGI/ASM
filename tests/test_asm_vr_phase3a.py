from dataclasses import replace
import xml.etree.ElementTree as ET

import torch

from aletheion_state_models.benchmarks.phase3a_data import (
    load_document_hash_splits,
    sample_byte_windows,
)
from aletheion_state_models.benchmarks.phase3a_plots import render_phase3a_charts
from aletheion_state_models.benchmarks.phase3a_summary import summarize_phase3a
from aletheion_state_models.benchmarks.phase3a_training import Phase3ARunResult
from aletheion_state_models.benchmarks.phase3a_variants import build_phase3a_variant


def test_document_hash_split_and_windows_are_deterministic(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_bytes(b"\n\n".join(f"document {index} ".encode() * 20 for index in range(400)))
    first = load_document_hash_splits(source)
    second = load_document_hash_splits(source)

    assert first.manifest == second.manifest
    assert sum(first.manifest["documents"].values()) == 400
    assert all(first.manifest["documents"][name] > 0 for name in ("train", "validation", "test"))
    batch_a = sample_byte_windows(
        first.train, batch_size=3, sequence_length=16, seed=17, step=2, device="cpu"
    )
    batch_b = sample_byte_windows(
        first.train, batch_size=3, sequence_length=16, seed=17, step=2, device="cpu"
    )
    assert torch.equal(batch_a[0], batch_b[0])
    assert torch.equal(batch_a[1], batch_b[1])
    assert torch.equal(batch_a[0][:, 1:], batch_a[1][:, :-1])


def test_phase3a_fixed_rank_builders_use_exact_prefix_masks():
    for variant, expected_rank in (
        ("vr_fixed_16", 16),
        ("vr_fixed_32", 32),
        ("vr_fixed_48", 48),
        ("vr_full", 64),
    ):
        model, resolved_rank = build_phase3a_variant(variant, 17)
        model.eval()
        output = model(torch.randint(0, 256, (2, 32)), collect_diagnostics=False)
        assert resolved_rank == expected_rank
        assert torch.all(output["variable_rank_ranks"] == expected_rank)
        assert torch.all(output["variable_rank_masks"][..., :expected_rank])
        assert not torch.any(output["variable_rank_masks"][..., expected_rank:])


def _run(variant: str, seed: int) -> Phase3ARunResult:
    rank = 32.0 if variant in {"vr_fixed_32", "vr_adaptive_32"} else 64.0
    return Phase3ARunResult(
        variant=variant,
        seed=seed,
        steps=10,
        tokens_seen=1000,
        best_step=10,
        validation_ce=2.0,
        validation_ppl=7.4,
        test_ce=2.02 if variant == "vr_adaptive_32" else 2.0,
        test_ppl=7.5,
        mean_rank=rank,
        rank_std=3.0 if variant == "vr_adaptive_32" else 0.0,
        rank_min=24.0 if variant == "vr_adaptive_32" else rank,
        rank_max=40.0 if variant == "vr_adaptive_32" else rank,
        rank_ce_correlation=0.2 if variant == "vr_adaptive_32" else 0.0,
        controller_gradient_fraction=1.0 if variant == "vr_adaptive_32" else 0.0,
        tokens_per_second=1000.0,
        peak_memory_mb=100.0,
        parameter_count=1000,
        streaming_error=1e-5,
        finite=True,
        history=[
            {"step": 5.0, "tokens": 500.0, "train_loss": 2.5, "validation_ce": 2.2, "mean_rank": rank},
            {"step": 10.0, "tokens": 1000.0, "train_loss": 2.1, "validation_ce": 2.0, "mean_rank": rank},
        ],
    )


def test_phase3a_summary_and_graphs_cover_every_variant(tmp_path):
    variants = (
        "asm_r",
        "vr_full",
        "vr_fixed_16",
        "vr_fixed_32",
        "vr_fixed_48",
        "vr_adaptive_32",
    )
    summary = summarize_phase3a(
        [_run(variant, seed) for seed in (17, 29, 43) for variant in variants]
    )
    assert summary["passed"]
    assert all(summary["gates"].values())

    render_phase3a_charts(summary, tmp_path)
    chart_names = (
        "validation_ce_by_tokens",
        "final_test_ce_by_variant",
        "quality_vs_mean_rank",
        "rank_distribution",
        "observed_cost",
        "paired_seed_deltas",
    )
    for name in chart_names:
        png = tmp_path / f"{name}.png"
        svg = tmp_path / f"{name}.svg"
        assert png.read_bytes().startswith(b"\x89PNG")
        ET.parse(svg)
    dashboard = (tmp_path / "index.html").read_text()
    assert "no hardware speedup claim" in dashboard
    assert dashboard.count("<img") == len(chart_names)
