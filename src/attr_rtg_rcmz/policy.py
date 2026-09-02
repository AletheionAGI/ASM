"""Frozen topology and deterministic-runtime policy for the draft engine."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

PROTOCOL = "ATTR-RTG-RCMZ-V1"
TRAINING_SEEDS = (29, 43, 71, 89, 107)
ARMS = ("R", "CM", "Z", "T")
CANDIDATES = ("U", "D", "L", "R", "BRAKE", "RECOVER")
PEAK_CAP_BYTES = 20 * 2**30


def _lp(value: str | bytes) -> bytes:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return len(payload).to_bytes(8, "big") + payload


def derive_seed64(purpose: str, training_seed: int) -> int:
    """Apply the preregistered length-prefixed SHA-256 seed derivation."""
    if (
        not purpose
        or type(training_seed) is not int
        or not 0 <= training_seed <= 2**64 - 1
    ):
        raise ValueError("purpose must be nonempty and training_seed a uint64 integer")
    seed_ascii = format(
        training_seed, "d"
    )  # canonical unsigned decimal: no sign/leading zeros
    material = _lp(PROTOCOL) + _lp(purpose) + _lp(seed_ascii)
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


@dataclass(frozen=True)
class DraftRunConfig:
    updates: int = 2_000
    synthetic: bool = False
    peak_cap_bytes: int = PEAK_CAP_BYTES

    def __post_init__(self) -> None:
        if self.updates <= 0:
            raise ValueError("updates must be positive")
        if not self.synthetic and self.updates != 2_000:
            raise ValueError("non-synthetic runs require exactly 2,000 updates")
        if self.synthetic and self.updates > 16:
            raise ValueError("synthetic dry runs are limited to 16 updates")
        if self.peak_cap_bytes != PEAK_CAP_BYTES:
            raise ValueError("the 20 GiB peak cap is frozen")


def prepare_deterministic_environment() -> None:
    """Set variables that must exist before the lazy torch import."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUDA_MODULE_LOADING"] = "EAGER"


def configure_torch(torch: object) -> None:
    """Configure an already lazily imported torch module."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
