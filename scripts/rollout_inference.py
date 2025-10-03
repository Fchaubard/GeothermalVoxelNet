"""
Full autoregressive rollout inference on original HDF5 simulations.

Runs the model autoregressively using its own predictions (no ground truth reset).
This simulates real deployment where we only have initial conditions.
"""
import os
import json
import argparse
import h5py
import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voxel_ode.model import VoxelAutoRegressor
from voxel_ode.data_prep import STATIC_KEYS


def load_model(ckpt_path, r, base_channels, depth, in_channels, device):
    """Load trained model from checkpoint."""
    model = VoxelAutoRegressor(
        in_channels=in_channels,
        base_channels=base_channels,
        depth=depth,
        r=r,
        use_param_broadcast=False,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def load_stats(stats_path):
    """Load normalization statistics."""
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return stats


def normalize_static(static, stats, device):
    """Normalize static channels: [C, Z, Y, X] -> tensor."""
    x = torch.from_numpy(static).float().to(device)
    mean = torch.tensor(stats["mean"]["static"], device=device).view(-1, 1, 1, 1)
    std = torch.tensor(stats["std"]["static"], device=device).view(-1, 1, 1, 1).clamp(min=1e-6)
    return (x - mean) / std


def normalize_grid(grid, ch, stats, device):
    """Normalize grid (Pressure or Temperature): [Z, Y, X] -> tensor."""
    x = torch.from_numpy(grid).float().to(device)
    mean = stats["mean"]["grid"][ch]
    std = max(stats["std"]["grid"][ch], 1e-6)
    return (x - mean) / std


def denormalize_grid(grid_norm, ch, stats):
    """Denormalize grid from normalized space to physical units."""
    mean = stats["mean"]["grid"][ch]
    std = max(stats["std"]["grid"][ch], 1e-6)
    return grid_norm * std + mean


def denormalize_scalar(scalar_norm, ch, stats):
    """Denormalize scalar from normalized space to physical units."""
    mean = stats["mean"]["scalar"][ch]
    std = max(stats["std"]["scalar"][ch], 1e-6)
    val = scalar_norm * std + mean
    # Inverse log1p if needed
    if stats["log1p_flags"]["scalar"][ch]:
        val = np.expm1(val)
    return val


@torch.no_grad()
def rollout(
    sim_path,
    stats_path,
    ckpt_path,
    save_path,
    r=3,
    base_channels=128,
    depth=12,
    device='cuda'
):
    """
    Autoregressive rollout on full simulation.

    Uses model's own predictions (no ground truth reset).
    Evaluates error accumulation over time.
    """
    print(f"Loading simulation: {sim_path}")
    print(f"Loading stats: {stats_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    # Load stats
    stats = load_stats(stats_path)

    # Load simulation
    with h5py.File(sim_path, "r") as f:
        # Static inputs
        static_list = []
        for k in STATIC_KEYS:
            static_list.append(f[f"Input/{k}"][...])
        static = np.stack(static_list, axis=0)  # [C_static, Z, Y, X]

        params = f["Input/ParamsScalar"][...].astype(np.float32)

        # Initial conditions (t=0)
        P0 = f["Input/Pressure0"][...]
        T0 = f["Input/Temperature0"][...]

        # Ground truth outputs for evaluation
        P_true = f["Output/Pressure"][...]  # [T, Z, Y, X]
        T_true = f["Output/Temperature"][...]  # [T, Z, Y, X]

        # Scalar outputs
        scalar_keys = [
            "FieldEnergyInjectionRate",
            "FieldEnergyProductionRate",
            "FieldEnergyProductionTotal",
            "FieldWaterInjectionRate",
            "FieldWaterProductionRate"
        ]
        scalars_true = np.stack([f[f"Output/{k}"][...] for k in scalar_keys], axis=-1)  # [T, 5]

    T, Z, Y, X = P_true.shape
    C_static = len(STATIC_KEYS)
    in_channels = C_static + 2  # static + P + T

    print(f"\nSimulation shape: {T} timesteps, {Z}×{Y}×{X} grid")
    print(f"Input channels: {in_channels} ({C_static} static + 2 lagged)")

    # Load model
    model = load_model(ckpt_path, r, base_channels, depth, in_channels, device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Normalize static inputs (stays constant)
    static_norm = normalize_static(static, stats, device).unsqueeze(0)  # [1, C_static, Z, Y, X]
    params_t = torch.from_numpy(params).float().to(device).unsqueeze(0)  # [1, 26]

    # Storage for predictions
    P_pred = np.zeros((T, Z, Y, X), dtype=np.float32)
    T_pred = np.zeros((T, Z, Y, X), dtype=np.float32)
    scalars_pred = np.zeros((T, 5), dtype=np.float32)

    # Initialize with ground truth at t=0
    P_pred[0] = P0
    T_pred[0] = T0
    # Note: scalars at t=0 are outputs, not initial conditions
    # We'll predict them in the first step

    # Autoregressive rollout
    print(f"\nRunning autoregressive rollout ({T-1} predictions)...")

    # Current state (starts with initial conditions)
    P_current = P0
    T_current = T0

    metrics_per_timestep = []

    for t in tqdm(range(T)):
        # Normalize current state
        P_norm = normalize_grid(P_current, 0, stats, device).unsqueeze(0).unsqueeze(0)  # [1, 1, Z, Y, X]
        T_norm = normalize_grid(T_current, 1, stats, device).unsqueeze(0).unsqueeze(0)  # [1, 1, Z, Y, X]

        # Concatenate inputs: [1, C_in, Z, Y, X]
        x_grid = torch.cat([static_norm, P_norm, T_norm], dim=1)

        # Predict next state
        grid_pred_norm, scalar_pred_norm = model(x_grid, params_t)

        # Denormalize predictions
        P_next = denormalize_grid(grid_pred_norm[0, 0].cpu().numpy(), 0, stats)
        T_next = denormalize_grid(grid_pred_norm[0, 1].cpu().numpy(), 1, stats)
        scalars_next = np.array([
            denormalize_scalar(scalar_pred_norm[0, i].cpu().item(), i, stats)
            for i in range(5)
        ])

        # Store predictions
        if t < T - 1:  # We predict t+1, so for t=T-1, we predict beyond available data
            P_pred[t + 1] = P_next
            T_pred[t + 1] = T_next
            scalars_pred[t] = scalars_next

            # Compute metrics vs ground truth
            mse_p = np.mean((P_next - P_true[t + 1]) ** 2)
            mse_t = np.mean((T_next - T_true[t + 1]) ** 2)
            mse_scalars = np.mean((scalars_next - scalars_true[t]) ** 2)

            # Relative error (mean absolute percentage error)
            mape_p = np.mean(np.abs((P_next - P_true[t + 1]) / (np.abs(P_true[t + 1]) + 1e-8))) * 100
            mape_t = np.mean(np.abs((T_next - T_true[t + 1]) / (np.abs(T_true[t + 1]) + 1e-8))) * 100

            metrics_per_timestep.append({
                't': t,
                'mse_pressure': float(mse_p),
                'mse_temperature': float(mse_t),
                'mse_scalars': float(mse_scalars),
                'mape_pressure': float(mape_p),
                'mape_temperature': float(mape_t),
            })

            # Use prediction as input for next timestep (autoregressive)
            P_current = P_next
            T_current = T_next
        else:
            # Last timestep: predict scalars only
            scalars_pred[t] = scalars_next

    # Compute overall metrics
    print("\n=== Overall Metrics ===")
    mse_p_all = np.mean((P_pred[1:] - P_true[1:]) ** 2)
    mse_t_all = np.mean((T_pred[1:] - T_true[1:]) ** 2)
    mse_scalars_all = np.mean((scalars_pred - scalars_true) ** 2)

    print(f"Pressure MSE:     {mse_p_all:.6e}")
    print(f"Temperature MSE:  {mse_t_all:.6e}")
    print(f"Scalars MSE:      {mse_scalars_all:.6e}")

    # Show error growth over time
    print("\n=== Error Accumulation ===")
    print("Timestep | Pressure MSE | Temperature MSE | Pressure MAPE | Temperature MAPE")
    print("-" * 80)
    for i in [0, 4, 9, 14, 19, 24, 28]:  # Sample timesteps
        if i < len(metrics_per_timestep):
            m = metrics_per_timestep[i]
            print(f"t={m['t']:2d}      | {m['mse_pressure']:12.6e} | {m['mse_temperature']:15.6e} | {m['mape_pressure']:13.2f}% | {m['mape_temperature']:16.2f}%")

    # Save predictions
    print(f"\nSaving predictions to: {save_path}")
    with h5py.File(save_path, "w") as f:
        f.create_dataset("predicted/Pressure", data=P_pred, compression="gzip")
        f.create_dataset("predicted/Temperature", data=T_pred, compression="gzip")
        f.create_dataset("predicted/scalars", data=scalars_pred, compression="gzip")

        # Also save ground truth for comparison
        f.create_dataset("ground_truth/Pressure", data=P_true, compression="gzip")
        f.create_dataset("ground_truth/Temperature", data=T_true, compression="gzip")
        f.create_dataset("ground_truth/scalars", data=scalars_true, compression="gzip")

        # Save metrics
        import json
        f.attrs["metrics_overall"] = json.dumps({
            "mse_pressure": float(mse_p_all),
            "mse_temperature": float(mse_t_all),
            "mse_scalars": float(mse_scalars_all),
        })
        f.attrs["metrics_per_timestep"] = json.dumps(metrics_per_timestep)

    print(f"Done! Predictions saved with ground truth for comparison.")
    return metrics_per_timestep


def main():
    parser = argparse.ArgumentParser(description="Autoregressive rollout inference")
    parser.add_argument("--sim_path", type=str, required=True, help="Path to original v2.4 HDF5 file")
    parser.add_argument("--stats_path", type=str, required=True, help="Path to stats.json from prep")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--save_path", type=str, required=True, help="Output path for predictions")
    parser.add_argument("--receptive_field_radius", type=int, default=3, help="Receptive field radius")
    parser.add_argument("--base_channels", type=int, default=128, help="Base channels")
    parser.add_argument("--depth", type=int, default=12, help="Model depth")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    rollout(
        args.sim_path,
        args.stats_path,
        args.ckpt_path,
        args.save_path,
        r=args.receptive_field_radius,
        base_channels=args.base_channels,
        depth=args.depth,
        device=args.device,
    )


if __name__ == "__main__":
    main()
