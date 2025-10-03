"""Distributed training script for voxel autoregressive model."""
import os
import json
import time
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxel_ode.dataset import VoxelARIndexDataset, pad_collate_fn
from voxel_ode.model import VoxelAutoRegressor
from voxel_ode.schedulers import WarmupCosine
from voxel_ode.utils import set_seed, ddp_print


def init_distributed():
    """Initialize DDP if running with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import datetime
        # Disable P2P for Docker environments where it may not work
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1800))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False


def load_stats(prepped_root):
    """Load statistics JSON."""
    with open(os.path.join(prepped_root, "stats.json"), "r") as f:
        stats = json.load(f)
    return stats


def evaluate(model, loader, device, stats, scalar_log1p_flags, max_batches=None):
    """
    Evaluate model on test set.

    Returns:
        Dictionary with per-output MSE and ±5% accuracy metrics
    """
    model.eval()

    # Accumulators
    mse_accum = {
        "grid_p": 0.0,
        "grid_T": 0.0,
        "scalar": torch.zeros(5, device=device)
    }
    acc_accum = {
        "grid_p": 0.0,
        "grid_T": 0.0,
        "scalar": torch.zeros(5, device=device)
    }
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if (max_batches is not None) and (i >= max_batches):
                break

            x_grid = batch["x_grid"].to(device, non_blocking=True)
            params = batch["params"].to(device, non_blocking=True)
            y_grid = batch["y_grid"].to(device, non_blocking=True)
            y_scalar = batch["y_scalar"].to(device, non_blocking=True)

            grid_pred, scalar_pred = model(x_grid, params)

            # MSE in normalized space
            mse_p = torch.mean((grid_pred[:, 0] - y_grid[:, 0]) ** 2).detach()
            mse_T = torch.mean((grid_pred[:, 1] - y_grid[:, 1]) ** 2).detach()

            # De-standardize for ±5% accuracy
            g_mean = torch.tensor(stats["mean"]["grid"], device=device).view(1, 2, 1, 1, 1)
            g_std = torch.tensor(stats["std"]["grid"], device=device).view(1, 2, 1, 1, 1).clamp(min=1e-6)
            grid_true = y_grid * g_std + g_mean
            grid_hat = grid_pred * g_std + g_mean

            def rel_ok(a, b, tol=0.05, eps=1e-8):
                return ((a - b).abs() <= tol * b.abs().clamp(min=eps)).float().mean()

            acc_p = rel_ok(grid_hat[:, 0], grid_true[:, 0])
            acc_T = rel_ok(grid_hat[:, 1], grid_true[:, 1])

            # Scalars
            s_mean = torch.tensor(stats["mean"]["scalar"], device=device).view(1, 5)
            s_std = torch.tensor(stats["std"]["scalar"], device=device).view(1, 5).clamp(min=1e-6)
            s_hat = scalar_pred * s_std + s_mean
            s_true = y_scalar * s_std + s_mean

            # Inverse log1p if needed
            for k, flag in enumerate(scalar_log1p_flags):
                if flag:
                    s_hat[:, k] = torch.expm1(s_hat[:, k])
                    s_true[:, k] = torch.expm1(s_true[:, k])

            mse_s = torch.mean((scalar_pred - y_scalar) ** 2, dim=0).detach()
            acc_s = torch.mean(
                ((s_hat - s_true).abs() <= 0.05 * s_true.abs().clamp(min=1e-8)).float(),
                dim=0
            ).detach()

            # Accumulate
            mse_accum["grid_p"] += mse_p.item()
            mse_accum["grid_T"] += mse_T.item()
            mse_accum["scalar"] += mse_s
            acc_accum["grid_p"] += acc_p.item()
            acc_accum["grid_T"] += acc_T.item()
            acc_accum["scalar"] += acc_s
            count += 1

    # Reduce across GPUs if using DDP
    if dist.is_initialized():
        for k in ["grid_p", "grid_T"]:
            t = torch.tensor([mse_accum[k], acc_accum[k], count], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            mse_accum[k] = (t[0] / t[2]).item()
            acc_accum[k] = (t[1] / t[2]).item()

        t_mse = torch.cat([mse_accum["scalar"], torch.tensor([count], device=device)])
        t_acc = torch.cat([acc_accum["scalar"], torch.tensor([count], device=device)])
        dist.all_reduce(t_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_acc, op=dist.ReduceOp.SUM)
        count = int(t_mse[-1].item())
        mse_accum["scalar"] = (t_mse[:-1] / max(1, count)).cpu().tolist()
        acc_accum["scalar"] = (t_acc[:-1] / max(1, count)).cpu().tolist()
    else:
        mse_accum["scalar"] = (mse_accum["scalar"] / max(1, count)).cpu().tolist()
        acc_accum["scalar"] = (acc_accum["scalar"] / max(1, count)).cpu().tolist()
        for k in ["grid_p", "grid_T"]:
            mse_accum[k] = mse_accum[k] / max(1, count)
            acc_accum[k] = acc_accum[k] / max(1, count)

    # Format results
    metrics = {
        "mse/grid_pressure": mse_accum["grid_p"],
        "mse/grid_temperature": mse_accum["grid_T"],
        "acc5/grid_pressure": acc_accum["grid_p"],
        "acc5/grid_temperature": acc_accum["grid_T"],
    }
    for i, name in enumerate(stats["scalar_channels"]):
        metrics[f"mse/{name}"] = mse_accum["scalar"][i]
        metrics[f"acc5/{name}"] = acc_accum["scalar"][i]

    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train voxel autoregressive model")

    # Data
    ap.add_argument("--prepped_root", type=str, required=True, help="Root with stats.json and indices")
    ap.add_argument("--train_index", type=str, default=None)
    ap.add_argument("--test_index", type=str, default=None)

    # Dataloader
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")

    # Model
    ap.add_argument("--receptive_field_radius", type=int, default=2, help="Receptive field radius (kernel size = 2*r+1)")
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--use_param_broadcast", action="store_true")

    # Training
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--seed", type=int, default=42)

    # Augmentations
    ap.add_argument("--aug_xy_rot", action="store_true")
    ap.add_argument("--aug_flip", action="store_true")
    ap.add_argument("--noise_std", type=float, default=0.0)

    # Training options
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="voxel-ode")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Initialize DDP
    use_ddp = init_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    # Load stats
    stats = load_stats(args.prepped_root)
    train_index = args.train_index or os.path.join(args.prepped_root, "index_train.jsonl")
    test_index = args.test_index or os.path.join(args.prepped_root, "index_test.jsonl")

    # Datasets
    train_ds = VoxelARIndexDataset(
        index_path=train_index,
        stats_path=os.path.join(args.prepped_root, "stats.json"),
        augment=True,
        aug_xy_rot=args.aug_xy_rot,
        aug_flip=args.aug_flip,
        noise_std=args.noise_std,
        use_param_broadcast=args.use_param_broadcast,
        use_params_as_condition=True,
    )
    test_ds = VoxelARIndexDataset(
        index_path=test_index,
        stats_path=os.path.join(args.prepped_root, "stats.json"),
        augment=False,
        aug_xy_rot=False,
        aug_flip=False,
        noise_std=0.0,
        use_param_broadcast=args.use_param_broadcast,
        use_params_as_condition=True,
    )

    # Samplers
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        test_sampler = None

    # Loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=pad_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, args.batch_size // 2),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        sampler=test_sampler,
        shuffle=False,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        collate_fn=pad_collate_fn,
    )

    # Model
    C_static = len(stats["static_channels"])
    in_channels = C_static + 2  # static + P + T
    if args.use_param_broadcast:
        in_channels += 26

    ddp_print(f"Creating model with in_channels={in_channels}, base_channels={args.base_channels}, depth={args.depth}")

    try:
        model = VoxelAutoRegressor(
            in_channels=in_channels,
            base_channels=args.base_channels,
            depth=args.depth,
            r=args.receptive_field_radius,
            cond_params_dim=26,
            use_param_broadcast=args.use_param_broadcast,
            grid_out_channels=2,
            scalar_out_dim=5,
        )
        ddp_print(f"Model created successfully with {sum(p.numel() for p in model.parameters())} parameters")

        model = model.to(device)
        ddp_print(f"Model moved to device {device}")
    except Exception as e:
        print(f"[RANK {dist.get_rank() if dist.is_initialized() else 0}] ERROR creating model: {e}")
        raise

    # Synchronize before wrapping in DDP
    if use_ddp:
        dist.barrier()
        ddp_print("All ranks synchronized before DDP wrapping")
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=False)
        ddp_print("Model wrapped in DDP successfully")

    # Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = WarmupCosine(opt, warmup_steps=args.warmup_steps, max_steps=args.max_steps)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # W&B
    run = None
    if args.use_wandb:
        if not dist.is_initialized() or dist.get_rank() == 0:
            import wandb
            run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Evaluate at step 0
    if use_ddp and isinstance(test_loader.sampler, DistributedSampler):
        test_loader.sampler.set_epoch(0)

    ddp_print("Evaluating at step 0...")
    metrics0 = evaluate(model, test_loader, device, stats, stats["log1p_flags"]["scalar"], max_batches=10)
    if run and (not dist.is_initialized() or dist.get_rank() == 0):
        for k, v in metrics0.items():
            run.log({f"test/{k}": v, "step": 0})
    if not dist.is_initialized() or dist.get_rank() == 0:
        ddp_print("Initial test metrics:", metrics0)

    # Training loop
    global_step = 0
    best_val = float("inf")
    t0 = time.time()

    ddp_print(f"Starting training for {args.max_steps} steps...")

    while global_step < args.max_steps:
        if use_ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(global_step // max(1, len(train_loader)) + 1)

        for it, batch in enumerate(train_loader):
            model.train()
            x_grid = batch["x_grid"].to(device, non_blocking=True)
            params = batch["params"].to(device, non_blocking=True)
            y_grid = batch["y_grid"].to(device, non_blocking=True)
            y_scalar = batch["y_scalar"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                grid_pred, scalar_pred = model(x_grid, params)
                loss_grid_p = torch.mean((grid_pred[:, 0] - y_grid[:, 0]) ** 2)
                loss_grid_T = torch.mean((grid_pred[:, 1] - y_grid[:, 1]) ** 2)
                loss_scalar = torch.mean((scalar_pred - y_scalar) ** 2)
                loss = loss_grid_p + loss_grid_T + loss_scalar

            scaler.scale(loss / args.accum_steps).backward()

            if ((it + 1) % args.accum_steps) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                # Logging
                if (global_step % args.log_every == 0):
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        lr = opt.param_groups[0]["lr"]
                        ddp_print(f"step {global_step:07d} | loss {loss.item():.6e} | lr {lr:.3e}")
                        if run:
                            run.log({
                                "train/loss_total": loss.item(),
                                "train/loss_grid_p": loss_grid_p.item(),
                                "train/loss_grid_T": loss_grid_T.item(),
                                "train/loss_scalar": loss_scalar.item(),
                                "lr": lr,
                                "step": global_step,
                                "time_per_step": (time.time() - t0) / max(1, global_step),
                            })

                # Evaluation
                if (global_step % args.eval_every == 0):
                    if use_ddp and isinstance(test_loader.sampler, DistributedSampler):
                        test_loader.sampler.set_epoch(global_step)

                    metrics = evaluate(model, test_loader, device, stats, stats["log1p_flags"]["scalar"], max_batches=50)
                    val_score = metrics["mse/grid_pressure"] + metrics["mse/grid_temperature"] + sum(
                        metrics[f"mse/{name}"] for name in stats["scalar_channels"]
                    ) / len(stats["scalar_channels"])

                    if not dist.is_initialized() or dist.get_rank() == 0:
                        ddp_print(f"[eval {global_step}] val_score={val_score:.6e}")
                        if run:
                            for k, v in metrics.items():
                                run.log({f"test/{k}": v, "step": global_step})

                        # Save best
                        if val_score < best_val:
                            best_val = val_score
                            ckpt_path = os.path.join(args.save_dir, f"best_step{global_step}.pt")
                            state = {
                                "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                "step": global_step,
                                "best_val": best_val,
                                "args": vars(args)
                            }
                            torch.save(state, ckpt_path)
                            ddp_print(f"Saved best checkpoint: {ckpt_path}")

                # Periodic checkpoint
                if (global_step % args.ckpt_every == 0):
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        ckpt_path = os.path.join(args.save_dir, f"step{global_step}.pt")
                        state = {
                            "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                            "step": global_step,
                            "best_val": best_val,
                            "args": vars(args)
                        }
                        torch.save(state, ckpt_path)
                        ddp_print(f"Saved checkpoint: {ckpt_path}")

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

    ddp_print("Training complete!")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
