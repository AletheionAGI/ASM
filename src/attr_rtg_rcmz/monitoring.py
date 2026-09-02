"""Peak-memory sampling implementations for orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class SyntheticPeakSampler:
    def __init__(self, allocation_classes: tuple[str, ...] = ("synthetic",)) -> None:
        self._samples = 0
        self._classes = allocation_classes

    def start(self, arm: str, training_seed: int) -> None:
        self._samples = 0

    def sample(self) -> None:
        self._samples += 1

    def stop(self) -> Mapping[str, Any]:
        return {
            "peak_bytes": 0,
            "samples": self._samples,
            "allocation_classes": self._classes,
        }


class CudaPeakSampler:
    """Samples torch allocator peaks; an injected NVML reader can add process use."""

    def __init__(
        self,
        torch: Any,
        nvml_process_bytes: Any | None,
        allocation_classes: Mapping[str, int],
    ) -> None:
        if not allocation_classes or any(
            int(size) < 0 for size in allocation_classes.values()
        ):
            raise ValueError(
                "all transient allocation classes and nonnegative bytes must be listed"
            )
        self._torch = torch
        self._nvml = nvml_process_bytes
        self._classes = allocation_classes
        self._nvml_peak = 0

    def start(self, arm: str, training_seed: int) -> None:
        self._torch.cuda.synchronize()
        self._torch.cuda.reset_peak_memory_stats()
        self._nvml_peak = 0
        self.sample()

    def sample(self) -> None:
        self._torch.cuda.synchronize()
        if self._nvml is not None:
            self._nvml_peak = max(self._nvml_peak, int(self._nvml()))

    def stop(self) -> Mapping[str, Any]:
        self.sample()
        allocated = int(self._torch.cuda.max_memory_allocated())
        reserved = int(self._torch.cuda.max_memory_reserved())
        explicit = sum(int(size) for size in self._classes.values())
        return {
            "peak_bytes": max(allocated, reserved, self._nvml_peak) + explicit,
            "allocator_allocated": allocated,
            "allocator_reserved": reserved,
            "nvml_process": self._nvml_peak,
            "explicit_bytes": explicit,
            "allocation_classes": dict(self._classes),
        }
