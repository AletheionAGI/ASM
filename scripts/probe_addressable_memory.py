from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from drm_language_emitter.addressable_memory import AddressableMemory
from drm_language_emitter.mqar import make_mqar_batch


KEY_VOCAB_SIZE = 32
VALUE_VOCAB_SIZE = 64
KEY_OFFSET = 2
VALUE_OFFSET = KEY_OFFSET + KEY_VOCAB_SIZE
VOCAB_SIZE = 256


def token_roles(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masks for pair values, queries, and key IDs used by MQAR."""
    previous = F.pad(input_ids[:, :-1], (1, 0))
    previous_two = F.pad(input_ids[:, :-2], (2, 0))
    pair_value = (
        (input_ids >= VALUE_OFFSET)
        & (previous >= KEY_OFFSET)
        & (previous < VALUE_OFFSET)
        & (previous_two != 1)
    )
    query_key = (input_ids >= KEY_OFFSET) & (input_ids < VALUE_OFFSET) & (previous == 1)
    key_ids = (input_ids - KEY_OFFSET).clamp(0, KEY_VOCAB_SIZE - 1)
    return pair_value, query_key, key_ids


class DenseSlotProbe(nn.Module):
    """The existing learned slot memory, retained for a longer learning curve."""

    def __init__(self, slots: int, d_model: int = 64, d_memory: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.token_state = nn.Linear(d_model, d_model)
        self.emitter = nn.Linear(d_model, VOCAB_SIZE)
        config = SimpleNamespace(
            addressable_memory_slots=slots,
            addressable_memory_dim=d_memory,
            addressable_memory_read_scale=1.0,
            addressable_memory_temperature=1.0,
            addressable_memory_usage_decay=0.99,
            addressable_memory_age_bias=1.0,
            addressable_memory_read_enabled=True,
            addressable_memory_write_enabled=True,
            addressable_memory_shuffle_on_eval=False,
            addressable_memory_read_top_k=0,
            addressable_memory_write_top_k=0,
            addressable_memory_use_previous_token_key=False,
            addressable_memory_write_bias=-1.0,
            d_state=d_model,
            d_token=d_model,
        )
        self.memory = AddressableMemory(config)
        nn.init.normal_(self.memory.read_output.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = self.embedding(input_ids)
        memory = self.memory.initial_state(input_ids.shape[0], input_ids.device, embeddings.dtype)
        outputs: list[torch.Tensor] = []
        diagnostic_rows: dict[str, list[torch.Tensor]] = {}
        for position in range(input_ids.shape[1]):
            state = torch.tanh(self.token_state(embeddings[:, position]))
            state, memory, diagnostics = self.memory.step(state, embeddings[:, position], memory)
            outputs.append(self.emitter(state))
            for name, value in diagnostics.items():
                diagnostic_rows.setdefault(name, []).append(value)
        diagnostics = {
            name: torch.stack(values, dim=1).mean()
            for name, values in diagnostic_rows.items()
        }
        return torch.stack(outputs, dim=1), diagnostics


class FastWeightProbe(nn.Module):
    """Fixed-size outer-product memory with deterministic MQAR phase control."""

    def __init__(self, d_value: int = 64):
        super().__init__()
        self.value_embedding = nn.Embedding(VOCAB_SIZE, d_value)
        self.emitter = nn.Linear(d_value, VOCAB_SIZE)
        self.empty_read = nn.Parameter(torch.zeros(d_value))

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length = input_ids.shape
        pair_value, query_key, key_ids = token_roles(input_ids)
        memory = self.empty_read.new_zeros(batch, KEY_VOCAB_SIZE, self.empty_read.numel())
        previous_keys = F.pad(key_ids[:, :-1], (1, 0))
        logits: list[torch.Tensor] = []
        writes = reads = 0
        for position in range(length):
            write_mask = pair_value[:, position]
            value = torch.tanh(self.value_embedding(input_ids[:, position]))
            if write_mask.any():
                row = F.one_hot(previous_keys[:, position], KEY_VOCAB_SIZE).to(value.dtype)
                update = torch.einsum("bk,bd->bkd", row, value)
                replacement = memory * (1.0 - row[:, :, None]) + update
                memory = torch.where(write_mask[:, None, None], replacement, memory)
                writes += int(write_mask.sum())
            row = F.one_hot(key_ids[:, position], KEY_VOCAB_SIZE).to(value.dtype)
            read = torch.einsum("bk,bkd->bd", row, memory)
            read = torch.where(query_key[:, position, None], read, self.empty_read)
            reads += int(query_key[:, position].sum())
            logits.append(self.emitter(read))
        diagnostics = {
            "writes_per_example": self.empty_read.new_tensor(writes / batch),
            "reads_per_example": self.empty_read.new_tensor(reads / batch),
            "memory_norm": memory.float().norm(dim=(-2, -1)).mean(),
        }
        return torch.stack(logits, dim=1), diagnostics


class OracleSlotProbe(nn.Module):
    """Non-learned end-to-end sanity check for MQAR construction and masking."""

    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.empty(0), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length = input_ids.shape
        pair_value, query_key, key_ids = token_roles(input_ids)
        memory = torch.zeros(batch, KEY_VOCAB_SIZE, dtype=torch.long, device=input_ids.device)
        previous_keys = F.pad(key_ids[:, :-1], (1, 0))
        logits = torch.full((batch, length, VOCAB_SIZE), -20.0, device=input_ids.device)
        writes = reads = 0
        for position in range(length):
            write_mask = pair_value[:, position]
            if write_mask.any():
                updated = memory.scatter(1, previous_keys[:, position, None], input_ids[:, position, None])
                memory = torch.where(write_mask[:, None], updated, memory)
                writes += int(write_mask.sum())
            recalled = memory.gather(1, key_ids[:, position, None]).squeeze(1)
            selected = torch.where(query_key[:, position], recalled, torch.zeros_like(recalled))
            logits[:, position].scatter_(1, selected[:, None], 20.0)
            reads += int(query_key[:, position].sum())
        diagnostics = {
            "writes_per_example": logits.new_tensor(writes / batch),
            "reads_per_example": logits.new_tensor(reads / batch),
        }
        return logits, diagnostics


@torch.inference_mode()
def evaluate(model: nn.Module, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 100_000)
    correct = total = 0
    losses: list[float] = []
    diagnostics: dict[str, list[float]] = {}
    for _ in range(args.eval_batches):
        x, y, mask = make_mqar_batch(
            args.batch_size, 8, 8, KEY_VOCAB_SIZE, VALUE_VOCAB_SIZE, generator, device
        )
        logits, values = model(x)
        selected, targets = logits[mask], y[mask]
        losses.append(float(F.cross_entropy(selected, targets)))
        correct += int((selected.argmax(-1) == targets).sum())
        total += targets.numel()
        for name, value in values.items():
            diagnostics.setdefault(name, []).append(float(value))
    return {
        "validation_ce": sum(losses) / len(losses),
        "validation_accuracy": correct / total,
        "validation_targets": total,
        "diagnostics": {name: sum(values) / len(values) for name, values in diagnostics.items()},
    }


def build_model(variant: str, slots: int) -> nn.Module:
    if variant == "ORACLE_SLOT":
        return OracleSlotProbe()
    if variant == "DENSE_SLOT_LONG":
        return DenseSlotProbe(slots)
    if variant == "FAST_WEIGHT":
        return FastWeightProbe()
    raise ValueError(f"unknown variant: {variant}")


def train_variant(
    variant: str, args: argparse.Namespace, milestones: set[int], device: torch.device
) -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed)
        model = build_model(variant, args.slots).to(device)
    rows = [{"step": 0, **evaluate(model, args, device)}]
    if variant == "ORACLE_SLOT":
        return {"variant": variant, "parameter_count": 0, "rows": rows}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        x, y, mask = make_mqar_batch(
            args.batch_size, 8, 8, KEY_VOCAB_SIZE, VALUE_VOCAB_SIZE, generator, device
        )
        logits, _ = model(x)
        loss = F.cross_entropy(logits[mask], y[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in milestones:
            row = {
                "step": step,
                "train_ce": float(loss.detach()),
                "elapsed_sec": time.perf_counter() - started,
                **evaluate(model, args, device),
            }
            rows.append(row)
            print(json.dumps({"variant": variant, **row}), flush=True)
    return {
        "variant": variant,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "rows": rows,
    }


def write_report(payload: dict[str, object], path: Path) -> None:
    lines = [
        "# Isolated associative-memory diagnostic",
        "",
        "| Variant | Steps | Validation CE | Accuracy | Parameters |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        final = result["rows"][-1]
        lines.append(
            f"| {result['variant']} | {final['step']:,} | {final['validation_ce']:.6f} "
            f"| {100 * final['validation_accuracy']:.2f}% | {result['parameter_count']:,} |"
        )
    gate = payload["gate"]
    lines.extend([
        "",
        "## Decision gate",
        "",
        f"- Oracle plumbing check: **{'passed' if gate['oracle_accuracy_at_least_99_9pct'] else 'failed'}**.",
        f"- Fast-weight learnability at 95%: **{'passed' if gate['fast_weight_accuracy_at_least_95pct'] else 'failed'}**.",
        f"- ASM-C2 reintegration: **{'authorized' if payload['passed'] else 'blocked'}**.",
        "",
        "The oracle is diagnostic only. It cannot authorize reintegration by itself.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle, dense-slot, and fast-weight MQAR diagnostic.")
    parser.add_argument("--variants", default="ORACLE_SLOT,DENSE_SLOT_LONG,FAST_WEIGHT")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--milestones", default="100,250,500,1000,2000,5000,10000")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    milestones = {int(value) for value in args.milestones.split(",")}
    milestones.add(args.steps)
    variants = [value.strip().upper() for value in args.variants.split(",") if value.strip()]
    results = [train_variant(variant, args, milestones, device) for variant in variants]
    final = {result["variant"]: result["rows"][-1] for result in results}
    gate = {
        "oracle_accuracy_at_least_99_9pct": final["ORACLE_SLOT"]["validation_accuracy"] >= 0.999,
        "fast_weight_accuracy_at_least_95pct": final["FAST_WEIGHT"]["validation_accuracy"] >= 0.95,
    }
    payload = {
        "protocol": vars(args) | {"output": str(args.output)},
        "results": results,
        "gate": gate,
        "passed": all(gate.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(payload, args.output.with_name("report.md"))
    print(json.dumps({"gate": gate, "passed": payload["passed"]}, indent=2))


if __name__ == "__main__":
    main()
