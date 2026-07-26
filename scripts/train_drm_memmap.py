from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from drm_language_emitter.config import DRMConfig
from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.training import count_parameters
from drm_language_emitter.utils import load_yaml_or_json, save_json


def distributed_state() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled:
        torch.distributed.init_process_group(backend="nccl")
    return enabled, rank, local_rank, world_size


def resolve_device(requested: str, local_rank: int) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested, but torch.cuda.is_available() is false")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device(requested)


def count_tokens_per_step(batch_size: int, seq_len: int, grad_accum_steps: int, world_size: int) -> int:
    return batch_size * seq_len * grad_accum_steps * world_size


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DRMConfig,
    args: argparse.Namespace,
    step: int,
    tokens_seen: int,
    parameter_count: int,
    best_val_ce: float,
    world_size: int,
) -> dict[str, Any]:
    module = model.module if hasattr(model, "module") else model
    return {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "parameter_count": parameter_count,
        "best_val_ce": best_val_ce,
        "world_size": world_size,
        "precision": args.precision,
        "dataset_manifest": str(args.dataset_manifest),
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


class BatchPrefetcher:
    def __init__(
        self,
        dataset: MemmapTokenDataset,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator,
        rank: int,
        world_size: int,
        pin_memory: bool,
        depth: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.generator = generator
        self.rank = rank
        self.world_size = world_size
        self.pin_memory = pin_memory
        self.depth = max(depth, 0)
        self.executor = ThreadPoolExecutor(max_workers=1) if self.depth > 0 else None
        self.pending: list[Future[tuple[torch.Tensor, torch.Tensor]]] = []
        for _ in range(self.depth):
            self._submit()

    def _submit(self) -> None:
        if self.executor is None:
            return
        self.pending.append(
            self.executor.submit(
                self.dataset.make_batch_cpu,
                self.batch_size,
                self.seq_len,
                self.generator,
                self.rank,
                self.world_size,
                self.pin_memory,
            )
        )

    def next_cpu(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.executor is None:
            return self.dataset.make_batch_cpu(
                self.batch_size,
                self.seq_len,
                self.generator,
                self.rank,
                self.world_size,
                self.pin_memory,
            )
        future = self.pending.pop(0)
        batch = future.result()
        self._submit()
        return batch

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    return int(payload["step"]), int(payload["tokens_seen"]), float(payload.get("best_val_ce", math.inf))


def resolve_resume_path(output_root: Path, resume: str) -> Path | None:
    if not resume:
        return None
    if resume == "latest":
        path = output_root / "checkpoint_latest.pt"
        return path if path.exists() else None
    return Path(resume)


def autocast_context(device: torch.device, precision: str):
    enabled = precision == "bf16" and device.type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def cuda_memory_mb(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"max_memory_mb": None, "memory_allocated_mb": None, "memory_reserved_mb": None}
    return {
        "max_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
        "memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
    }


def make_adamw(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    fused: bool,
) -> torch.optim.Optimizer:
    if fused:
        try:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
        except TypeError:
            pass
        except RuntimeError:
            pass
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def evaluate_ce(
    model: torch.nn.Module,
    dataset: MemmapTokenDataset,
    batch_size: int,
    seq_len: int,
    batches: int,
    device: torch.device,
    precision: str,
    global_step: int,
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(100_000 + global_step)
    losses: list[float] = []
    for _ in range(max(batches, 1)):
        x, y = dataset.make_batch(batch_size, seq_len, device, generator=generator, pin_memory=device.type == "cuda")
        with autocast_context(device, precision):
            out = model(x, y, global_step=global_step, collect_diagnostics=False)
        losses.append(float(out["aux_losses"].get("ce", out["loss"]).detach().cpu()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DRM on uint8 token shards from a manifest.")
    parser.add_argument("--config", default="configs/drm_500m.yaml")
    parser.add_argument("--dataset-manifest", default="data/tokens_5b/manifest.json")
    parser.add_argument("--output-root", default="runs/drm_500m_5b")
    parser.add_argument("--target-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--steps", type=int, default=0, help="Override target-tokens with a fixed step count when > 0.")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-process batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-tokens-interval", type=int, default=50_000_000)
    parser.add_argument("--checkpoint-tokens-interval", type=int, default=250_000_000)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", default="", help="Path to checkpoint, or 'latest'.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-forward", action="store_true")
    parser.add_argument("--geometry-update-interval", type=int, default=None, help="Override config geometry_update_interval.")
    parser.add_argument("--aux-loss-interval", type=int, default=None, help="Compute geometry auxiliary losses every N flow ticks.")
    parser.add_argument("--naturalization-interval", type=int, default=None, help="Apply metric naturalization every N flow ticks.")
    parser.add_argument("--forward-chunk-size", type=int, default=None, help="Reset geometry cache at chunk boundaries when > 0.")
    parser.add_argument("--compile-drm-step", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--shared-geometry-trunk", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-batches", type=int, default=0, help="Prefetch N CPU batches ahead with a background thread.")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=0, help="Collect coarse timing for the first N steps.")
    parser.add_argument("--metrics-format", choices=["json", "jsonl", "both"], default="json")
    parser.add_argument("--lambda-metric-diversity", type=float, default=None, help="Override config lambda_metric_diversity.")
    parser.add_argument("--metric-rank", type=int, default=None, help="Override config metric_rank for phased pretraining experiments.")
    parser.add_argument("--save-final-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ddp, rank, local_rank, world_size = distributed_state()
    rank_zero = rank == 0
    device = resolve_device(args.device, local_rank)
    output_root = Path(args.output_root)
    dataset_manifest = Path(args.dataset_manifest)
    torch.manual_seed(args.seed + rank)

    config = DRMConfig.from_dict(load_yaml_or_json(args.config))
    config.max_seq_len = args.seq_len
    config.vocab_size = 256
    if args.metric_rank is not None:
        config.metric_rank = args.metric_rank
    if args.lambda_metric_diversity is not None:
        config.lambda_metric_diversity = args.lambda_metric_diversity
    if args.geometry_update_interval is not None:
        config.geometry_update_interval = args.geometry_update_interval
    if args.aux_loss_interval is not None:
        config.aux_loss_interval = args.aux_loss_interval
    if args.naturalization_interval is not None:
        config.naturalization_interval = args.naturalization_interval
    if args.forward_chunk_size is not None:
        config.forward_chunk_size = args.forward_chunk_size
    if args.compile_drm_step:
        config.compile_drm_step = True
    if args.torch_compile:
        config.use_torch_compile = True
    if args.shared_geometry_trunk:
        config.use_shared_geometry_trunk = True
    config._validate()
    model = DRMEmitterModel(config).to(device)
    parameter_count = count_parameters(model)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if rank_zero:
        output_root.mkdir(parents=True, exist_ok=True)
        save_json(
            output_root / "run_config.json",
            {
                "config": config.to_dict(),
                "parameter_count": parameter_count,
                "dataset_manifest": str(dataset_manifest),
                "target_tokens": args.target_tokens,
                "tokens_per_step": count_tokens_per_step(args.batch_size, args.seq_len, args.grad_accum_steps, world_size),
                "world_size": world_size,
                "args": vars(args),
            },
        )
        print(f"parameter_count={parameter_count}", flush=True)
        print(f"dataset_manifest={dataset_manifest}", flush=True)
        print(f"world_size={world_size}", flush=True)

    train_dataset = MemmapTokenDataset(dataset_manifest, split="train")
    val_dataset = MemmapTokenDataset(dataset_manifest, split="val")
    if rank_zero:
        print(f"train_tokens_available={len(train_dataset)}", flush=True)
        print(f"val_tokens_available={len(val_dataset)}", flush=True)

    if args.dry_run:
        if args.dry_run_forward:
            x, y = train_dataset.make_batch(
                1,
                min(args.seq_len, 16),
                device,
                generator=torch.Generator().manual_seed(args.seed),
                pin_memory=args.pin_memory,
            )
            with autocast_context(device, args.precision):
                out = model(x, y, global_step=1)
            if rank_zero:
                print(f"dry_run_loss={float(out['loss'].detach().cpu()):.6f}", flush=True)
        train_dataset.close()
        val_dataset.close()
        if ddp:
            torch.distributed.destroy_process_group()
        return

    optimizer = make_adamw(model, args.lr, args.weight_decay, args.fused_adamw)
    start_step = 0
    tokens_seen = 0
    best_val_ce = math.inf
    resume_path = resolve_resume_path(output_root, args.resume)
    if resume_path is not None:
        start_step, tokens_seen, best_val_ce = load_checkpoint(resume_path, model, optimizer, device)
        if rank_zero:
            print(f"resumed={resume_path} step={start_step} tokens_seen={tokens_seen}", flush=True)

    tokens_per_step = count_tokens_per_step(args.batch_size, args.seq_len, args.grad_accum_steps, world_size)
    total_steps = args.steps if args.steps > 0 else math.ceil(max(args.target_tokens - tokens_seen, 0) / tokens_per_step)
    final_step = start_step + total_steps
    next_eval_tokens = ((tokens_seen // args.eval_tokens_interval) + 1) * args.eval_tokens_interval
    next_checkpoint_tokens = ((tokens_seen // args.checkpoint_tokens_interval) + 1) * args.checkpoint_tokens_interval
    generator = torch.Generator().manual_seed(args.seed + rank * 9973 + start_step)
    history: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    rolling_step_times: list[float] = []
    prefetcher = BatchPrefetcher(
        train_dataset,
        args.batch_size,
        args.seq_len,
        generator,
        rank,
        world_size,
        args.pin_memory and device.type == "cuda",
        args.prefetch_batches,
    )
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, final_step + 1):
        step_started = time.perf_counter()
        data_elapsed = 0.0
        train_elapsed = 0.0
        step_loss_tensors: list[torch.Tensor] = []
        for accum_index in range(args.grad_accum_steps):
            data_started = time.perf_counter()
            x_cpu, y_cpu = prefetcher.next_cpu()
            x, y = MemmapTokenDataset.move_batch_to_device(x_cpu, y_cpu, device)
            data_elapsed += time.perf_counter() - data_started
            train_started = time.perf_counter()
            with autocast_context(device, args.precision):
                out = model(x, y, global_step=step, collect_diagnostics=False)
                loss = out["loss"] / args.grad_accum_steps
            loss.backward()
            step_loss_tensors.append(out["aux_losses"].get("ce", out["loss"]).detach())
            train_elapsed += time.perf_counter() - train_started
        opt_started = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_elapsed = time.perf_counter() - opt_started
        tokens_seen += tokens_per_step
        step_elapsed = time.perf_counter() - step_started
        rolling_step_times.append(step_elapsed)
        if len(rolling_step_times) > 50:
            rolling_step_times.pop(0)

        train_ce_tensor = torch.stack(step_loss_tensors).mean()
        if ddp:
            loss_tensor = train_ce_tensor.clone()
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            train_ce_tensor = loss_tensor
        if step - start_step <= args.profile_steps:
            profile_rows.append(
                {
                    "step": step,
                    "data_elapsed_sec": data_elapsed,
                    "forward_backward_elapsed_sec": train_elapsed,
                    "optimizer_elapsed_sec": optimizer_elapsed,
                    "step_elapsed_sec": step_elapsed,
                    "instant_tokens_per_sec": tokens_per_step / max(step_elapsed, 1e-8),
                    **cuda_memory_mb(device),
                }
            )
        should_log = rank_zero and args.log_interval > 0 and (step == 1 or step % args.log_interval == 0)
        if should_log:
            train_ce = float(train_ce_tensor.detach().cpu())
        else:
            train_ce = None

        eval_due = tokens_seen >= next_eval_tokens
        checkpoint_due = tokens_seen >= next_checkpoint_tokens

        if eval_due and rank_zero:
            val_ce = evaluate_ce(model, val_dataset, args.batch_size, args.seq_len, args.eval_batches, device, args.precision, step)
            best_val_ce = min(best_val_ce, val_ce)
            if train_ce is None:
                train_ce = float(train_ce_tensor.detach().cpu())
        else:
            val_ce = None
        if eval_due:
            next_eval_tokens += args.eval_tokens_interval
            if ddp:
                torch.distributed.barrier()

        if should_log or (eval_due and rank_zero):
            elapsed = time.perf_counter() - started
            rolling_elapsed = sum(rolling_step_times)
            row = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_ce": train_ce,
                "val_ce": val_ce,
                "best_val_ce": best_val_ce if math.isfinite(best_val_ce) else None,
                "tokens_per_sec": (tokens_seen - (start_step * tokens_per_step)) / max(elapsed, 1e-8),
                "instant_tokens_per_sec": tokens_per_step / max(step_elapsed, 1e-8),
                "rolling_tokens_per_sec": (tokens_per_step * len(rolling_step_times)) / max(rolling_elapsed, 1e-8),
                "step_elapsed_sec": step_elapsed,
                "data_elapsed_sec": data_elapsed,
                "forward_backward_elapsed_sec": train_elapsed,
                "optimizer_elapsed_sec": optimizer_elapsed,
                "elapsed_sec": elapsed,
                **cuda_memory_mb(device),
            }
            history.append(row)
            if args.metrics_format in {"json", "both"}:
                save_json(output_root / "metrics_latest.json", {"history": history, "latest": row, "profile": profile_rows})
            if args.metrics_format in {"jsonl", "both"}:
                append_jsonl(output_root / "metrics_history.jsonl", row)
                save_json(output_root / "metrics_latest.json", {"latest": row, "profile": profile_rows})
            print(json.dumps(row), flush=True)

        if checkpoint_due and rank_zero:
            payload = checkpoint_payload(model, optimizer, config, args, step, tokens_seen, parameter_count, best_val_ce, world_size)
            save_checkpoint(output_root / "checkpoint_latest.pt", payload)
            save_checkpoint(output_root / f"checkpoint_tokens_{tokens_seen}.pt", payload)
        if checkpoint_due:
            next_checkpoint_tokens += args.checkpoint_tokens_interval
            if ddp:
                torch.distributed.barrier()

        if tokens_seen >= args.target_tokens:
            break

    if rank_zero:
        if args.save_final_checkpoint:
            payload = checkpoint_payload(model, optimizer, config, args, step, tokens_seen, parameter_count, best_val_ce, world_size)
            save_checkpoint(output_root / "checkpoint_last.pt", payload)
        save_json(
            output_root / "summary.json",
            {
                "final_step": step,
                "tokens_seen": tokens_seen,
                "target_tokens": args.target_tokens,
                "parameter_count": parameter_count,
                "best_val_ce": best_val_ce if math.isfinite(best_val_ce) else None,
                "world_size": world_size,
                "dataset_manifest": str(dataset_manifest),
                "profile": profile_rows,
            },
        )

    prefetcher.close()
    train_dataset.close()
    val_dataset.close()
    if ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
