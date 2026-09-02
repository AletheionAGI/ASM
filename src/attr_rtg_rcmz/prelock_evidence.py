"""Clean-process, full-topology synthetic CUDA evidence generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .constants import ARMS, TRAINING_SEEDS
from .policy import (
    PEAK_CAP_BYTES,
    configure_torch,
    derive_seed64,
    prepare_deterministic_environment,
)


def run_evidence(output: Path, *, prearmed: bool = False) -> dict[str, Any]:
    """Instantiate all 20 full configs without generating any HazardWorld."""
    prepare_deterministic_environment()
    import torch

    from .config import load_config
    from .contracts import InferenceMessage
    from .models import build_adapter

    if not torch.cuda.is_available():
        raise RuntimeError("synthetic full-shape evidence requires CUDA")
    configure_torch(torch)
    supervisor = _Supervisor(output.with_suffix(".supervisor.json"), prearmed)
    supervisor.arm()
    digest = hashlib.sha256()
    peaks = []
    topology = []
    try:
        for seed in TRAINING_SEEDS:
            for arm in ARMS:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                monitor = _NvmlMonitor()
                monitor.start()
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    path = (
                        Path(__file__).resolve().parents[2]
                        / "configs"
                        / "attr_rtg_rcmz_v1"
                        / f"{arm.lower()}_seed{seed}.yaml"
                    )
                    config = load_config(path)
                    model = build_adapter(config).cuda().float().train()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
                    message, labels = _full_shape_batch(torch, seed)
                    result = model(InferenceMessage.from_mapping(message))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        result.logits, labels
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                stream.synchronize()
                _hash_tensor(digest, result.logits)
                _hash_tensor(digest, result.common24)
                _hash_tensor(digest, result.native_state)
                for name, tensor in sorted(model.state_dict().items()):
                    digest.update(name.encode())
                    _hash_tensor(digest, tensor)
                allocated = int(torch.cuda.max_memory_allocated())
                reserved = int(torch.cuda.max_memory_reserved())
                nvml = monitor.stop()
                explicit = _explicit_bytes(message, labels)
                peak = max(allocated, reserved, nvml) + sum(explicit.values())
                if peak > PEAK_CAP_BYTES:
                    raise RuntimeError("synthetic evidence exceeded 20 GiB peak cap")
                peaks.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "peak_bytes": peak,
                        "allocator_allocated": allocated,
                        "allocator_reserved": reserved,
                        "nvml_process": nvml,
                        "explicit_allocation_classes": explicit,
                    }
                )
                topology.append([seed, arm])
                del result, loss, optimizer, model, message, labels
        statistics = _exercise_statistics(torch, digest)
        receipt = {
            "status": "COMPLETED",
            "kind": "FULL-SHAPE SYNTHETIC NON-OFFICIAL",
            "topology": topology,
            "updates_per_model": 1,
            "context_length": 256,
            "candidate_shape": [64, 6, 4],
            "logical_length_range": [4, 256],
            "head_shapes": {"common24": [64, 24], "logits": [64, 6]},
            "payload_sha256": digest.hexdigest(),
            "peak_cap_bytes": PEAK_CAP_BYTES,
            "peaks": peaks,
            "statistics": statistics,
            "prearmed_supervisor": prearmed,
        }
        _exclusive_json(output, receipt)
        supervisor.completed()
        return receipt
    except BaseException as error:
        supervisor.failed(type(error).__name__)
        raise


def _full_shape_batch(torch: Any, seed: int) -> tuple[dict[str, Any], Any]:
    generator = torch.Generator(device="cuda").manual_seed(
        derive_seed64("synthetic-evidence", seed)
    )
    histories = torch.randint(
        0, 256, (64, 256), dtype=torch.uint8, device="cuda", generator=generator
    )
    candidates = torch.randint(
        0, 256, (64, 6, 4), dtype=torch.uint8, device="cuda", generator=generator
    )
    masks = torch.ones((64, 6), dtype=torch.bool, device="cuda")
    lengths = (
        torch.linspace(4, 256, 64, dtype=torch.float64, device="cuda")
        .round()
        .to(torch.int64)
    )
    labels = torch.randint(
        0, 2, (64, 6), dtype=torch.float32, device="cuda", generator=generator
    )
    return {
        "history_bytes": histories,
        "candidate4s": candidates,
        "masks": masks,
        "logical_lengths": lengths,
    }, labels


def _hash_tensor(digest: Any, tensor: Any) -> None:
    value = tensor.detach().contiguous().cpu()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())


def _explicit_bytes(message: dict[str, Any], labels: Any) -> dict[str, int]:
    staging = sum(item.numel() * item.element_size() for item in message.values())
    staging += labels.numel() * labels.element_size()
    return {
        "gpu_input_staging": staging,
        "pinned_prefetch": staging,
        "conservative_workspace_bound": 32 * 2**20,
    }


def _nvml_process_bytes() -> int:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    total = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == str(os.getpid()):
            total += int(fields[1]) * 2**20
    return total


class _NvmlMonitor:
    """Poll process memory throughout an arm without reading model metrics."""

    def __init__(self) -> None:
        self.peak = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _poll(self) -> None:
        while not self.stop_event.wait(0.02):
            self.peak = max(self.peak, _nvml_process_bytes())

    def stop(self) -> int:
        self.peak = max(self.peak, _nvml_process_bytes())
        self.stop_event.set()
        self.thread.join(timeout=1)
        return self.peak


def _exercise_statistics(torch: Any, digest: Any) -> dict[str, Any]:
    from .bootstrap import paired_bootstrap, simultaneous_bounds
    from .constants import ARMS, CONTRASTS
    from .gates import contrast_gate

    shape = (5, 3, 32, 4, 5)
    base = (
        torch.arange(5 * 3 * 32 * 4 * 5, dtype=torch.float64, device="cuda").reshape(
            shape
        )
        / 1e6
    )
    endpoints = {arm: base + index * 1e-3 for index, arm in enumerate(ARMS)}
    replicates = paired_bootstrap(endpoints)
    gates = {}
    for left, right in CONTRASTS:
        key = f"{left}-{right}"
        lower, upper = simultaneous_bounds(replicates[key])
        marginals = (endpoints[left] - endpoints[right]).mean(dim=(2, 3))
        gate = contrast_gate(lower, upper, marginals)
        _hash_tensor(digest, replicates[key])
        gates[key] = {
            "passed": gate.passed,
            "bounds_pass": gate.bounds_pass,
            "marginals_pass": gate.marginals_pass,
        }
    return {"raw_shape": list(shape), "bootstrap_replicates": 1000, "gates": gates}


class _Supervisor:
    def __init__(self, path: Path, enabled: bool) -> None:
        self.path, self.enabled = path, enabled
        self.deadline = time.monotonic() + 20 * 60 * 60

    def arm(self) -> None:
        if self.enabled:
            _exclusive_json(
                self.path,
                {"status": "ARMED", "clock": "monotonic", "timeout_hours": 20},
            )

    def completed(self) -> None:
        if self.enabled:
            self.path.write_text(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "before_deadline": time.monotonic() <= self.deadline,
                    }
                )
                + "\n"
            )

    def failed(self, reason: str) -> None:
        if self.enabled:
            status = "TIMEOUT" if time.monotonic() > self.deadline else "CRASH"
            self.path.write_text(
                json.dumps({"status": status, "reason": reason}) + "\n"
            )


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prearmed", action="store_true")
    args = parser.parse_args()
    receipt = run_evidence(args.output, prearmed=args.prearmed)
    print(receipt["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
