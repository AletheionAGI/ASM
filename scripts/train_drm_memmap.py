from __future__ import annotations

import argparse
import json
import math
import os
import time
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
        "dataset_manifest": str(args.train_manifest),
        "train_manifest": str(args.train_manifest),
        "validation_manifest": str(args.validation_manifest),
        "args": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def link_checkpoint(source: Path, target: Path) -> None:
    """Point a checkpoint alias at an existing payload without duplicating it."""
    temporary = target.with_name(f".{target.name}.tmp-link")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, target)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location=device, weights_only=True)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_state_all") is not None:
        cuda_rng_state_all = [
            state.detach().cpu().to(dtype=torch.uint8)
            for state in payload["cuda_rng_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
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
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(100_000 + global_step)
    losses: list[float] = []
    for _ in range(max(batches, 1)):
        x, y = dataset.make_batch(batch_size, seq_len, device, generator=generator)
        with autocast_context(device, precision):
            out = model(x, y, global_step=global_step, collect_diagnostics=False)
        losses.append(float(out["aux_losses"].get("ce", out["loss"]).detach().cpu()))
    model.train(was_training)
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DRM on uint8 token shards from a manifest.")
    parser.add_argument("--config", default="configs/drm_500m.yaml")
    parser.add_argument(
        "--dataset-manifest",
        default=None,
        help="Legacy combined manifest containing train/val splits.",
    )
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--validation-manifest", default=None)
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
    parser.add_argument(
        "--checkpoint-token-milestones",
        default="",
        help="Comma-separated token milestones; overrides the checkpoint interval.",
    )
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", default="", help="Path to checkpoint, or 'latest'.")
    parser.add_argument("--save-best-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-forward", action="store_true")
    parser.add_argument(
        "--sequence-mode",
        choices=[
            "local_step",
            "geodesic_step",
            "directional_candidates",
            "directional_cumsum",
            "directional_block_cumsum",
            "directional_superblock_cumsum",
        ],
        default=None,
    )
    parser.add_argument("--geodesic-solver-steps", type=int, default=None)
    parser.add_argument("--geodesic-lr", type=float, default=None)
    parser.add_argument("--geodesic-anchor-weight", type=float, default=None)
    parser.add_argument("--geodesic-metric-weight", type=float, default=None)
    parser.add_argument("--geodesic-risk-weight", type=float, default=None)
    parser.add_argument("--directional-candidate-temperature", type=float, default=None)
    parser.add_argument("--directional-candidate-scale", type=float, default=None)
    parser.add_argument("--directional-cumsum-step-mode", choices=["candidate", "velocity"], default=None)
    parser.add_argument("--directional-cumsum-block-size", type=int, default=None)
    parser.add_argument("--directional-superblock-size", type=int, default=None)
    parser.add_argument("--directional-superblock-local-size", type=int, default=None)
    parser.add_argument("--directional-endpoint-correction-weight", type=float, default=None)
    parser.add_argument("--directional-endpoint-correction-power", type=float, default=None)
    parser.add_argument("--directional-cumsum-inner-block-size", type=int, default=None)
    parser.add_argument("--directional-anderson-iterations", type=int, default=None)
    parser.add_argument("--directional-anderson-history-size", type=int, default=None)
    parser.add_argument("--directional-anderson-ridge", type=float, default=None)
    parser.add_argument("--directional-anderson-relaxation", type=float, default=None)
    parser.add_argument("--directional-anderson-transition-mode", choices=["candidate", "velocity"], default=None)
    parser.add_argument("--directional-anderson-block-stride", type=int, default=None)
    parser.add_argument("--directional-anderson-scope", choices=["trajectory", "endpoint"], default=None)
    parser.add_argument("--directional-fixed-point-iterations", type=int, default=None)
    parser.add_argument("--directional-fixed-point-relaxation", type=float, default=None)
    parser.add_argument("--directional-local-mixer", choices=["none", "causal_conv"], default=None)
    parser.add_argument("--directional-local-mixer-hidden-size", type=int, default=None)
    parser.add_argument("--directional-local-mixer-kernel-size", type=int, default=None)
    parser.add_argument("--directional-local-mixer-layers", type=int, default=None)
    parser.add_argument("--directional-local-mixer-scale", type=float, default=None)
    parser.add_argument("--lambda-block-consistency", type=float, default=None)
    parser.add_argument("--block-consistency-weight", type=float, default=None)
    parser.add_argument("--lambda-sampled-block-consistency", type=float, default=None)
    parser.add_argument("--sampled-block-consistency-interval", type=int, default=None)
    parser.add_argument("--sampled-block-consistency-local-size", type=int, default=None)
    parser.add_argument("--sampled-block-consistency-teacher-mode", choices=["candidate", "velocity"], default=None)
    args = parser.parse_args()
    checkpoint_milestones = sorted(
        {
            int(item.strip())
            for item in args.checkpoint_token_milestones.split(",")
            if item.strip()
        }
    )
    if any(value <= 0 for value in checkpoint_milestones):
        raise ValueError("checkpoint token milestones must be positive")
    if args.dataset_manifest:
        if args.train_manifest or args.validation_manifest:
            parser.error("--dataset-manifest cannot be combined with --train-manifest/--validation-manifest")
        args.train_manifest = args.dataset_manifest
        args.validation_manifest = args.dataset_manifest
        train_split = "train"
        validation_split = "val"
    else:
        args.train_manifest = args.train_manifest or "data/tokens_5b/manifest.json"
        args.validation_manifest = args.validation_manifest or args.train_manifest
        train_split = "train"
        validation_split = "val" if args.validation_manifest == args.train_manifest else "validation"

    ddp, rank, local_rank, world_size = distributed_state()
    rank_zero = rank == 0
    device = resolve_device(args.device, local_rank)
    output_root = Path(args.output_root)
    train_manifest = Path(args.train_manifest)
    validation_manifest = Path(args.validation_manifest)
    torch.manual_seed(args.seed + rank)

    config = DRMConfig.from_dict(load_yaml_or_json(args.config))
    config.max_seq_len = args.seq_len
    config.vocab_size = 256
    config.seed = args.seed
    if args.sequence_mode is not None:
        config.sequence_mode = args.sequence_mode
    if args.geodesic_solver_steps is not None:
        config.geodesic_solver_steps = args.geodesic_solver_steps
    if args.geodesic_lr is not None:
        config.geodesic_lr = args.geodesic_lr
    if args.geodesic_anchor_weight is not None:
        config.geodesic_anchor_weight = args.geodesic_anchor_weight
    if args.geodesic_metric_weight is not None:
        config.geodesic_metric_weight = args.geodesic_metric_weight
    if args.geodesic_risk_weight is not None:
        config.geodesic_risk_weight = args.geodesic_risk_weight
    if args.directional_candidate_temperature is not None:
        config.directional_candidate_temperature = args.directional_candidate_temperature
    if args.directional_candidate_scale is not None:
        config.directional_candidate_scale = args.directional_candidate_scale
    if args.directional_cumsum_step_mode is not None:
        config.directional_cumsum_step_mode = args.directional_cumsum_step_mode
    if args.directional_cumsum_block_size is not None:
        config.directional_cumsum_block_size = args.directional_cumsum_block_size
    if args.directional_superblock_size is not None:
        config.directional_superblock_size = args.directional_superblock_size
    if args.directional_superblock_local_size is not None:
        config.directional_superblock_local_size = args.directional_superblock_local_size
    if args.directional_endpoint_correction_weight is not None:
        config.directional_endpoint_correction_weight = args.directional_endpoint_correction_weight
    if args.directional_endpoint_correction_power is not None:
        config.directional_endpoint_correction_power = args.directional_endpoint_correction_power
    if args.directional_cumsum_inner_block_size is not None:
        config.directional_cumsum_inner_block_size = args.directional_cumsum_inner_block_size
    if args.directional_anderson_iterations is not None:
        config.directional_anderson_iterations = args.directional_anderson_iterations
    if args.directional_anderson_history_size is not None:
        config.directional_anderson_history_size = args.directional_anderson_history_size
    if args.directional_anderson_ridge is not None:
        config.directional_anderson_ridge = args.directional_anderson_ridge
    if args.directional_anderson_relaxation is not None:
        config.directional_anderson_relaxation = args.directional_anderson_relaxation
    if args.directional_anderson_transition_mode is not None:
        config.directional_anderson_transition_mode = args.directional_anderson_transition_mode
    if args.directional_anderson_block_stride is not None:
        config.directional_anderson_block_stride = args.directional_anderson_block_stride
    if args.directional_anderson_scope is not None:
        config.directional_anderson_scope = args.directional_anderson_scope
    if args.directional_fixed_point_iterations is not None:
        config.directional_fixed_point_iterations = args.directional_fixed_point_iterations
    if args.directional_fixed_point_relaxation is not None:
        config.directional_fixed_point_relaxation = args.directional_fixed_point_relaxation
    if args.directional_local_mixer is not None:
        config.directional_local_mixer = args.directional_local_mixer
    if args.directional_local_mixer_hidden_size is not None:
        config.directional_local_mixer_hidden_size = args.directional_local_mixer_hidden_size
    if args.directional_local_mixer_kernel_size is not None:
        config.directional_local_mixer_kernel_size = args.directional_local_mixer_kernel_size
    if args.directional_local_mixer_layers is not None:
        config.directional_local_mixer_layers = args.directional_local_mixer_layers
    if args.directional_local_mixer_scale is not None:
        config.directional_local_mixer_scale = args.directional_local_mixer_scale
    if args.lambda_block_consistency is not None:
        config.lambda_block_consistency = args.lambda_block_consistency
    if args.block_consistency_weight is not None:
        config.block_consistency_weight = args.block_consistency_weight
    if args.lambda_sampled_block_consistency is not None:
        config.lambda_sampled_block_consistency = args.lambda_sampled_block_consistency
    if args.sampled_block_consistency_interval is not None:
        config.sampled_block_consistency_interval = args.sampled_block_consistency_interval
    if args.sampled_block_consistency_local_size is not None:
        config.sampled_block_consistency_local_size = args.sampled_block_consistency_local_size
    if args.sampled_block_consistency_teacher_mode is not None:
        config.sampled_block_consistency_teacher_mode = args.sampled_block_consistency_teacher_mode
    config = config.validated_copy()
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
                "dataset_manifest": str(train_manifest),
                "train_manifest": str(train_manifest),
                "validation_manifest": str(validation_manifest),
                "target_tokens": args.target_tokens,
                "tokens_per_step": count_tokens_per_step(args.batch_size, args.seq_len, args.grad_accum_steps, world_size),
                "world_size": world_size,
                "args": vars(args),
            },
        )
        print(f"parameter_count={parameter_count}", flush=True)
        print(f"train_manifest={train_manifest}", flush=True)
        print(f"validation_manifest={validation_manifest}", flush=True)
        print(f"world_size={world_size}", flush=True)

    train_dataset = MemmapTokenDataset(train_manifest, split=train_split)
    val_dataset = MemmapTokenDataset(validation_manifest, split=validation_split)
    if rank_zero:
        print(f"train_tokens_available={len(train_dataset)}", flush=True)
        print(f"val_tokens_available={len(val_dataset)}", flush=True)

    if args.dry_run:
        if args.dry_run_forward:
            x, y = train_dataset.make_batch(1, min(args.seq_len, 16), device, generator=torch.Generator().manual_seed(args.seed))
            with autocast_context(device, args.precision):
                out = model(x, y, global_step=1)
            if rank_zero:
                print(f"dry_run_loss={float(out['loss'].detach().cpu()):.6f}", flush=True)
        train_dataset.close()
        val_dataset.close()
        if ddp:
            torch.distributed.destroy_process_group()
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    pending_milestones = [value for value in checkpoint_milestones if value > tokens_seen]
    next_checkpoint_tokens = (
        pending_milestones[0]
        if pending_milestones
        else ((tokens_seen // args.checkpoint_tokens_interval) + 1) * args.checkpoint_tokens_interval
    )
    generator = torch.Generator().manual_seed(args.seed + rank * 9973 + start_step)
    metrics_path = output_root / "metrics_latest.json"
    history: list[dict[str, Any]] = []
    elapsed_offset = 0.0
    if resume_path is not None and metrics_path.is_file():
        previous_metrics = load_yaml_or_json(metrics_path)
        history = list(previous_metrics.get("history", []))
        if history:
            elapsed_offset = float(history[-1].get("elapsed_sec", 0.0))
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = start_step

    for step in range(start_step + 1, final_step + 1):
        step_loss = 0.0
        for accum_index in range(args.grad_accum_steps):
            x, y = train_dataset.make_batch(
                args.batch_size,
                args.seq_len,
                device,
                generator=generator,
                rank=rank,
                world_size=world_size,
            )
            with autocast_context(device, args.precision):
                out = model(x, y, global_step=step, collect_diagnostics=False)
                loss = out["loss"] / args.grad_accum_steps
            loss.backward()
            step_loss += float(out["aux_losses"].get("ce", out["loss"]).detach().cpu())
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.max_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step

        if ddp:
            loss_tensor = torch.tensor(step_loss / args.grad_accum_steps, device=device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            train_ce = float(loss_tensor.detach().cpu())
        else:
            train_ce = step_loss / args.grad_accum_steps

        eval_due = tokens_seen >= next_eval_tokens
        checkpoint_due = tokens_seen >= next_checkpoint_tokens
        should_log = rank_zero and args.log_interval > 0 and (step == 1 or step % args.log_interval == 0)

        if eval_due and rank_zero:
            val_ce = evaluate_ce(model, val_dataset, args.batch_size, args.seq_len, args.eval_batches, device, args.precision, step)
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                if args.save_best_checkpoint:
                    payload = checkpoint_payload(model, optimizer, config, args, step, tokens_seen, parameter_count, best_val_ce, world_size)
                    save_checkpoint(output_root / "checkpoint_best.pt", payload)
        else:
            val_ce = None
        if eval_due:
            next_eval_tokens += args.eval_tokens_interval
            if ddp:
                torch.distributed.barrier()

        if should_log or (eval_due and rank_zero):
            elapsed = elapsed_offset + time.perf_counter() - started
            row = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_ce": train_ce,
                "val_ce": val_ce,
                "best_val_ce": best_val_ce if math.isfinite(best_val_ce) else None,
                "tokens_per_sec": tokens_seen / max(elapsed, 1e-8),
                "elapsed_sec": elapsed,
            }
            history.append(row)
            save_json(output_root / "metrics_latest.json", {"history": history, "latest": row})
            print(json.dumps(row), flush=True)

        if checkpoint_due and rank_zero:
            payload = checkpoint_payload(model, optimizer, config, args, step, tokens_seen, parameter_count, best_val_ce, world_size)
            if pending_milestones:
                reached = [value for value in pending_milestones if value <= tokens_seen]
                for milestone in reached:
                    milestone_path = output_root / f"checkpoint_milestone_{milestone}.pt"
                    save_checkpoint(milestone_path, payload)
                    link_checkpoint(milestone_path, output_root / "checkpoint_latest.pt")
                pending_milestones = [value for value in pending_milestones if value > tokens_seen]
            else:
                checkpoint_path = output_root / f"checkpoint_tokens_{tokens_seen}.pt"
                save_checkpoint(checkpoint_path, payload)
                link_checkpoint(checkpoint_path, output_root / "checkpoint_latest.pt")
        if checkpoint_due:
            next_checkpoint_tokens = (
                pending_milestones[0]
                if pending_milestones
                else next_checkpoint_tokens + args.checkpoint_tokens_interval
            )
            if ddp:
                torch.distributed.barrier()

        if tokens_seen >= args.target_tokens:
            break

    if rank_zero:
        payload = checkpoint_payload(model, optimizer, config, args, step, tokens_seen, parameter_count, best_val_ce, world_size)
        latest_checkpoint = output_root / "checkpoint_latest.pt"
        if latest_checkpoint.is_file() and tokens_seen >= args.target_tokens:
            link_checkpoint(latest_checkpoint, output_root / "checkpoint_last.pt")
        else:
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
                "dataset_manifest": str(train_manifest),
                "train_manifest": str(train_manifest),
                "validation_manifest": str(validation_manifest),
            },
        )

    train_dataset.close()
    val_dataset.close()
    if ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
