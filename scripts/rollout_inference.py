"""Full autoregressive rollout inference on a single simulation."""
import os
import json
import argparse
import h5py
import torch
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxel_ode.model import VoxelAutoRegressor


def load_stats(stats_path, device):
    """Load and convert statistics to tensors."""
    with open(stats_path, "r") as f:
        stats = json.load(f)

    stats_t = {
        "static_mean": torch.tensor(stats["mean"]["static"], device=device).view(-1, 1, 1, 1),
        "static_std": torch.tensor(stats["std"]["static"], device=device).view(-1, 1, 1, 1).clamp(min=1e-6),
        "grid_mean": torch.tensor(stats["mean"]["grid"], device=device).view(1, 2, 1, 1, 1),
        "grid_std": torch.tensor(stats["std"]["grid"], device=device).view(1, 2, 1, 1, 1).clamp(min=1e-6),
        "scalar_mean": torch.tensor(stats["mean"]["scalar"], device=device).view(1, 5),
        "scalar_std": torch.tensor(stats["std"]["scalar"], device=device).view(1, 5).clamp(min=1e-6),
        "scalar_log1p_flags": stats["log1p_flags"]["scalar"],
        "static_channels": stats["static_channels"],
    }
    return stats_t


@torch.no_grad()
def autoregressive_rollout(
    sim_path,
    stats_path,
    ckpt_path,
    save_path,
    r=2,
    base_channels=64,
    depth=8,
    use_param_broadcast=False,
    device='cuda'
):
    """
    Perform full autoregressive rollout on a single simulation.

    Args:
        sim_path: Path to prepped .h5 file
        stats_path: Path to stats.json
        ckpt_path: Path to model checkpoint
        save_path: Where to save predictions
        r, base_channels, depth: Model architecture params
        use_param_broadcast: Whether params are broadcast as channels
        device: Device to run on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    stats = load_stats(stats_path, device=device)

    # Load simulation data
    with h5py.File(sim_path, "r") as f:
        static = torch.tensor(f["static"][...], dtype=torch.float32, device=device)
        params = torch.tensor(f["params_scalar"][...], dtype=torch.float32, device=device).unsqueeze(0)
        outputs_grid_true = f["outputs_grid"][...]  # (T, 2, Z, Y, X)
        outputs_scalar_true = f["outputs_scalar"][...]  # (T, 5)

    T, _, Z, Y, X = outputs_grid_true.shape
    C_static = static.shape[0]
    in_channels = C_static + 2 + (26 if use_param_broadcast else 0)

    # Create model
    model = VoxelAutoRegressor(
        in_channels=in_channels,
        base_channels=base_channels,
        depth=depth,
        r=r,
        cond_params_dim=26,
        use_param_broadcast=use_param_broadcast,
        grid_out_channels=2,
        scalar_out_dim=5,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded model from {ckpt_path}")
    print(f"Rolling out {T} timesteps for grid shape ({Z}, {Y}, {X})")

    # Standardize static once
    static_norm = (static - stats["static_mean"]) / stats["static_std"]

    # Prediction arrays
    pred_grid = np.zeros_like(outputs_grid_true, dtype=np.float32)
    pred_scalar = np.zeros_like(outputs_scalar_true, dtype=np.float32)

    # Initialize with true t=0
    pred_grid[0] = outputs_grid_true[0]
    pred_scalar[0] = outputs_scalar_true[0]

    # Autoregressive rollout
    for t in range(T - 1):
        # Use ground truth at time t as input
        p_t = torch.tensor(outputs_grid_true[t, 0], dtype=torch.float32, device=device)
        T_t = torch.tensor(outputs_grid_true[t, 1], dtype=torch.float32, device=device)

        # Standardize
        p_t = (p_t - stats["grid_mean"][0, 0]) / stats["grid_std"][0, 0]
        T_t = (T_t - stats["grid_mean"][0, 1]) / stats["grid_std"][0, 1]

        # Build input
        x_grid = torch.cat([static_norm, p_t.unsqueeze(0), T_t.unsqueeze(0)], dim=0)

        if use_param_broadcast:
            Bc = params[0].view(-1, 1, 1, 1).expand(-1, Z, Y, X)
            x_grid = torch.cat([x_grid, Bc], dim=0)

        x_grid = x_grid.unsqueeze(0)  # [1, C_in, Z, Y, X]

        # Predict t+1
        grid_pred_norm, scalar_pred_norm = model(x_grid, params)

        # De-normalize grid
        grid_pred = (grid_pred_norm * stats["grid_std"] + stats["grid_mean"]).squeeze(0).cpu().numpy()

        # De-normalize scalars
        scalar_pred = (scalar_pred_norm * stats["scalar_std"] + stats["scalar_mean"]).squeeze(0)
        for k, flag in enumerate(stats["scalar_log1p_flags"]):
            if flag:
                scalar_pred[k] = torch.expm1(scalar_pred[k])

        pred_grid[t + 1] = grid_pred
        pred_scalar[t + 1] = scalar_pred.cpu().numpy()

        if (t + 1) % 10 == 0:
            print(f"  Completed timestep {t+1}/{T-1}")

    # Save predictions
    with h5py.File(save_path, "w") as g:
        g.create_dataset("predicted/outputs_grid", data=pred_grid, compression="gzip", compression_opts=4)
        g.create_dataset("predicted/outputs_scalar", data=pred_scalar, compression="gzip", compression_opts=4)
        g.create_dataset("copied/params_scalar", data=params.squeeze(0).cpu().numpy())
        g.create_dataset("copied/static_shape", data=np.array([C_static, Z, Y, X], dtype=np.int32))
        g.attrs["source_sim"] = sim_path
        g.attrs["ckpt_path"] = ckpt_path

    print(f"\nSaved rollout predictions to: {save_path}")


def main():
    ap = argparse.ArgumentParser(description="Autoregressive rollout inference")
    ap.add_argument("--prepped_sim_path", type=str, required=True)
    ap.add_argument("--stats_path", type=str, required=True)
    ap.add_argument("--ckpt_path", type=str, required=True)
    ap.add_argument("--save_path", type=str, required=True)
    ap.add_argument("--receptive_field_radius", type=int, default=2, help="Receptive field radius")
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--use_param_broadcast", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    autoregressive_rollout(
        sim_path=args.prepped_sim_path,
        stats_path=args.stats_path,
        ckpt_path=args.ckpt_path,
        save_path=args.save_path,
        r=args.receptive_field_radius,
        base_channels=args.base_channels,
        depth=args.depth,
        use_param_broadcast=args.use_param_broadcast,
        device=args.device,
    )


if __name__ == "__main__":
    main()
