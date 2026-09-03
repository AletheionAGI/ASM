"""Spawn-isolated model scorer with a narrow serialized IPC contract."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MESSAGE_KEYS = frozenset({"history_bytes", "candidate4s", "masks", "logical_lengths"})
FORBIDDEN_MODULE_PREFIXES = ("world_model", "attr_rtg_rcmz.official_data")


@dataclass(frozen=True)
class ScorerRequest:
    arm: str
    config_ref: str
    checkpoint_ref: str
    message: Mapping[str, Any]

    def __post_init__(self) -> None:
        if set(self.message) != MESSAGE_KEYS:
            raise ValueError(
                "scorer message must contain exactly the frozen four fields"
            )
        if self.arm not in {"R", "CM", "Z", "T"}:
            raise ValueError("unregistered arm")
        if not self.config_ref or not self.checkpoint_ref:
            raise ValueError("config and checkpoint references are required")
        for value in self.message.values():
            module = type(value).__module__
            if module.startswith(FORBIDDEN_MODULE_PREFIXES):
                raise TypeError(
                    "world, origin, and truth objects are forbidden in scorer IPC"
                )


@dataclass(frozen=True)
class ScorerResponse:
    logits: tuple[Any, ...]
    common24: tuple[Any, ...]
    native_state: tuple[Any, ...]


def serialize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Detach the four tensors onto CPU before crossing the process boundary."""
    if set(message) != MESSAGE_KEYS:
        raise ValueError("broker may serialize only the exact four model fields")
    result = {}
    for key in ("history_bytes", "candidate4s", "masks", "logical_lengths"):
        value = message[key]
        if not hasattr(value, "detach"):
            raise TypeError("serialized model fields must be tensors")
        result[key] = value.detach().to("cpu").contiguous().clone()
    return result


def score_in_clean_process(
    requests: Sequence[ScorerRequest], *, device: str
) -> tuple[ScorerResponse, ...]:
    """Start one clean scorer for one arm, wait for immutable CPU results, then join."""
    items = tuple(requests)
    if not items:
        return ()
    identity = {(item.arm, item.config_ref, item.checkpoint_ref) for item in items}
    if len(identity) != 1:
        raise ValueError("one scorer process serves exactly one arm/config/checkpoint")
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_score_worker, args=(child, items, device))
    process.start()
    child.close()
    try:
        status, payload = parent.recv()
    finally:
        parent.close()
        process.join()
    if process.exitcode != 0 or status != "ok":
        raise RuntimeError(f"isolated scorer failed: {payload}")
    return payload


def _immutable_floats(tensor: Any) -> tuple[Any, ...]:
    """Preserve every tensor dimension while freezing leaves as Python floats."""
    values = tensor.detach().to(device="cpu", dtype=None).tolist()

    def freeze(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(freeze(item) for item in value)
        return float(value)

    frozen = freeze(values)
    if not isinstance(frozen, tuple):
        raise TypeError("scorer output must have at least one dimension")
    return frozen


def _score_worker(
    connection: Any, requests: tuple[ScorerRequest, ...], device: str
) -> None:
    """Child entry point: imports only contracts/config/model/policy/checkpoint stack."""
    try:
        import sys
        from dataclasses import replace

        import torch

        from .config import ModelConfig, load_config
        from .contracts import InferenceMessage
        from .models import build_adapter
        from .policy import configure_torch

        forbidden = [
            name
            for name in sys.modules
            if name == "world_model"
            or name.startswith("world_model.")
            or name == "attr_rtg_rcmz.official_data"
        ]
        if forbidden:
            raise RuntimeError(f"forbidden scorer modules mapped: {forbidden}")
        first = requests[0]
        if (
            not Path(first.config_ref).is_file()
            or not Path(first.checkpoint_ref).is_file()
        ):
            raise FileNotFoundError("scorer reference does not exist")
        checkpoint = torch.load(
            first.checkpoint_ref, map_location="cpu", weights_only=False
        )
        config = ModelConfig(**checkpoint["config"])
        registered = load_config(first.config_ref)
        authorized = replace(
            registered, synthetic_only=False, official_operations_allowed=True
        )
        if config != authorized or config.arm != first.arm:
            raise ValueError("config/checkpoint/arm references differ")
        configure_torch(torch)
        model = build_adapter(config).to(device=device, dtype=torch.float32)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        stream = torch.cuda.Stream() if device == "cuda" else None
        context = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        responses = []
        with torch.no_grad(), context:
            for request in requests:
                message = {
                    key: value.to(device) for key, value in request.message.items()
                }
                result = model(InferenceMessage.from_mapping(message))
                responses.append(
                    ScorerResponse(
                        _immutable_floats(result.logits),
                        _immutable_floats(result.common24),
                        _immutable_floats(result.native_state),
                    )
                )
        if stream is not None:
            stream.synchronize()
        connection.send(("ok", tuple(responses)))
    except BaseException as error:  # noqa: BLE001 — child must report every terminal failure
        connection.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
