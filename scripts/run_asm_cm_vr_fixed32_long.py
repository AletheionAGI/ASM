"""Run gated seed-17 then multiseed long-context ASM-CM-VR validation."""

import hashlib
import json
from pathlib import Path
from aletheion_state_models.benchmarks.cmvr_long_curriculum import (
    ARMS,
    arm_passed,
    run_arm,
)
from aletheion_state_models.benchmarks.cmvr_long_plots import render
from aletheion_state_models.benchmarks.cmvr_long_summary import summarize
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result

OUTPUT = Path("docs/benchmarks/asm_cm_vr_fixed32_long")
RUN_ROOT = "runs/asm_cm_vr_fixed32_long"
SEEDS = (17, 29, 43)


def persist(results, status):
    summary = summarize(results, SEEDS)
    summary["status"] = status
    write_result(OUTPUT / "summary.json", summary)
    render(summary, OUTPUT)
    return summary


def main():
    files = (
        "src/aletheion_state_models/benchmarks/cmvr_long_curriculum.py",
        "src/aletheion_state_models/benchmarks/cmvr_long_summary.py",
        "src/aletheion_state_models/benchmarks/cmvr_long_plots.py",
        "scripts/run_asm_cm_vr_fixed32_long.py",
    )
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update(Path(name).read_bytes())
    manifest = {
        "experiment": "ASM-CM-VR fixed-32 long multiseed with adaptive exploratory arm",
        "gating_order": "seed 17, then seeds 29/43 only if fixed-32 passes",
        "seeds": list(SEEDS),
        "arms": ARMS,
        "curriculum": {
            "40": 1000,
            "80": 500,
            "160": 500,
            "320": 300,
            "512": 200,
            "1024": 100,
            "4096": 25,
        },
        "frozen_test_lengths": [40, 512, 4096, 32768],
        "promotion_threshold": "fixed-32 accuracy >=80% at every test length, finite 32K streaming, parity <=1e-4 in every seed; adaptive is exploratory",
        "optimizer": "AdamW lr=3e-4 weight_decay=0.01, no language replay",
        "test_selection": "terminal checkpoint fixed before held-out test; no test-based selection",
        "implementation": {"files": list(files), "sha256": digest.hexdigest()},
        "hardware": "NVIDIA GeForce RTX 4090",
    }
    write_result(OUTPUT / "manifest.json", manifest)
    results = []
    for arm in ARMS:
        results.append(run_arm(arm, 17, run_root=RUN_ROOT))
        persist(results, "seed17_incomplete" if len(results) < 2 else "seed17_complete")
    fixed17 = next(row for row in results if row["arm"] == "cm_vr_fixed32")
    if arm_passed(fixed17):
        for seed in SEEDS[1:]:
            for arm in ARMS:
                results.append(run_arm(arm, seed, run_root=RUN_ROOT))
                persist(results, f"seed_{seed}_progress")
        status = "completed_multiseed"
    else:
        status = "stopped_after_seed17_gate_failure"
    summary = persist(results, status)
    manifest["status"] = status
    manifest["test_opened_after_terminal_checkpoints"] = True
    write_result(OUTPUT / "manifest.json", manifest)
    print(
        json.dumps(
            {"status": status, "gates": summary["gates"], "passed": summary["passed"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
