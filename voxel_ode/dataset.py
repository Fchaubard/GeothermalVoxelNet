"""PyTorch dataset for voxel grid patches with augmentations."""
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


def pad_collate_fn(batch):
    """
    Custom collate function that pads tensors to handle variable shapes from rotation augmentations.

    When 90° or 270° rotations are applied to non-square grids, Y and X dimensions swap.
    This function pads all tensors to the max dimensions in the batch.
    """
    # Find max dimensions across batch
    max_z = max(item["x_grid"].shape[1] for item in batch)
    max_y = max(item["x_grid"].shape[2] for item in batch)
    max_x = max(item["x_grid"].shape[3] for item in batch)

    # Pad each item
    x_grids = []
    y_grids = []
    params = []
    y_scalars = []

    for item in batch:
        x = item["x_grid"]
        y = item["y_grid"]

        # Compute padding needed (pad only on right/bottom)
        pad_z = max_z - x.shape[1]
        pad_y = max_y - x.shape[2]
        pad_x = max_x - x.shape[3]

        # Pad format: (left, right, top, bottom, front, back)
        padding = (0, pad_x, 0, pad_y, 0, pad_z)

        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=0)
        y_padded = torch.nn.functional.pad(y, padding, mode='constant', value=0)

        x_grids.append(x_padded)
        y_grids.append(y_padded)
        params.append(item["params"])
        y_scalars.append(item["y_scalar"])

    return {
        "x_grid": torch.stack(x_grids),
        "params": torch.stack(params),
        "y_grid": torch.stack(y_grids),
        "y_scalar": torch.stack(y_scalars),
    }


class H5Cache:
    """Per-process cache of open h5py.File handles to avoid repeated opening."""

    def __init__(self):
        self.handles = {}

    def get(self, path):
        h = self.handles.get(path)
        if h is None or not h.id.valid:
            h = h5py.File(path, "r")
            self.handles[path] = h
        return h

    def __del__(self):
        for p, h in self.handles.items():
            try:
                h.close()
            except Exception:
                pass
        self.handles.clear()


class VoxelARIndexDataset(Dataset):
    """
    Dataset for autoregressive voxel prediction from patch indices.

    Loads:
        - Static features (geology, wells, etc.)
        - Grid state at time t (Pressure, Temperature)
        - Targets at time t+1 (grid + scalars)

    Applies:
        - Per-channel standardization
        - Optional log1p for scalars
        - XY augmentations (rotation, flip) - never Z axis
        - Gaussian noise on inputs
    """

    def __init__(
        self,
        index_path: str,
        stats_path: str,
        augment: bool = True,
        aug_xy_rot: bool = True,
        aug_flip: bool = True,
        noise_std: float = 0.0,
        noise_exclude_static_idx=(0, 2, 3),  # FaultId, IsActive, IsWell
        use_param_broadcast: bool = False,
        use_params_as_condition: bool = True,
    ):
        super().__init__()

        # Load index
        self.index = []
        with open(index_path, "r") as f:
            for line in f:
                self.index.append(json.loads(line))

        # Load statistics
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        self.augment = augment
        self.aug_xy_rot = aug_xy_rot
        self.aug_flip = aug_flip
        self.noise_std = noise_std
        self.noise_exclude_static_idx = set(noise_exclude_static_idx)
        self.use_param_broadcast = use_param_broadcast
        self.use_params_as_condition = use_params_as_condition
        self.h5cache = H5Cache()

        # Convert stats to tensors
        self.static_mean = torch.tensor(self.stats["mean"]["static"], dtype=torch.float32)
        self.static_std = torch.tensor(self.stats["std"]["static"], dtype=torch.float32)
        self.grid_mean = torch.tensor(self.stats["mean"]["grid"], dtype=torch.float32)
        self.grid_std = torch.tensor(self.stats["std"]["grid"], dtype=torch.float32)
        self.scalar_mean = torch.tensor(self.stats["mean"]["scalar"], dtype=torch.float32)
        self.scalar_std = torch.tensor(self.stats["std"]["scalar"], dtype=torch.float32)
        self.scalar_log1p_flags = self.stats["log1p_flags"]["scalar"]

    def __len__(self):
        return len(self.index)

    def _standardize_static(self, static):
        """Standardize static channels: [C_static, Z, Y, X]"""
        x = torch.from_numpy(static).float()
        m = self.static_mean.view(-1, 1, 1, 1)
        s = self.static_std.view(-1, 1, 1, 1).clamp(min=1e-6)
        return (x - m) / s

    def _standardize_grid(self, grid, ch):
        """Standardize grid channel (0=Pressure, 1=Temperature): [Z, Y, X]"""
        x = torch.from_numpy(grid).float()
        m = self.grid_mean[ch]
        s = self.grid_std[ch].clamp(min=1e-6)
        return (x - m) / s

    def _standardize_scalar(self, val, ch):
        """Standardize scalar (with optional log1p): scalar value"""
        x = torch.tensor(val, dtype=torch.float32)
        if self.scalar_log1p_flags[ch]:
            x = torch.log1p(torch.clamp(x, min=0))
        m = self.scalar_mean[ch]
        s = self.scalar_std[ch].clamp(min=1e-6)
        return (x - m) / s

    def __getitem__(self, idx):
        rec = self.index[idx]
        f = self.h5cache.get(rec["sim_path"])
        t = rec["t"]
        z0, z1, y0, y1, x0, x1 = rec["z0"], rec["z1"], rec["y0"], rec["y1"], rec["x0"], rec["x1"]

        # Load data
        static = f["static"][:, z0:z1, y0:y1, x0:x1]  # [C_static, pz, py, px]
        params = f["params_scalar"][...]  # [26]

        # Lagged outputs at t
        p_t = f["outputs_grid"][t, 0, z0:z1, y0:y1, x0:x1]
        T_t = f["outputs_grid"][t, 1, z0:z1, y0:y1, x0:x1]

        # Targets at t+1
        p_tp1 = f["outputs_grid"][t + 1, 0, z0:z1, y0:y1, x0:x1]
        T_tp1 = f["outputs_grid"][t + 1, 1, z0:z1, y0:y1, x0:x1]
        scalars_tp1 = f["outputs_scalar"][t + 1, :]  # [5]

        # Standardize inputs
        static = self._standardize_static(static)  # [C_static, ...]
        p_t = self._standardize_grid(p_t, 0).unsqueeze(0)
        T_t = self._standardize_grid(T_t, 1).unsqueeze(0)
        x_grid = torch.cat([static, p_t, T_t], dim=0).contiguous()

        # Scalar parameters
        params_t = torch.from_numpy(params).float()
        params_condition = params_t.clone()

        if self.use_param_broadcast:
            # Broadcast params as additional channels
            Bc = params_t.view(-1, 1, 1, 1).expand(-1, x_grid.shape[1], x_grid.shape[2], x_grid.shape[3])
            x_grid = torch.cat([x_grid, Bc], dim=0)

        # Standardize targets
        y_grid = torch.stack([
            self._standardize_grid(p_tp1, 0),
            self._standardize_grid(T_tp1, 1),
        ], dim=0)
        y_scalar = torch.stack([
            self._standardize_scalar(scalars_tp1[i], i) for i in range(5)
        ], dim=0)

        # Augmentations (XY only, never Z)
        if self.augment:
            # Add batch dim for rotation
            x_aug = x_grid.unsqueeze(0)
            y_aug = y_grid.unsqueeze(0)

            # Random 90° rotation in XY plane (dims -2, -1 are Y, X)
            k = np.random.randint(0, 4) if self.aug_xy_rot else 0
            if k > 0:
                x_aug = torch.rot90(x_aug, k, dims=(-2, -1))
                y_aug = torch.rot90(y_aug, k, dims=(-2, -1))

            # Random flips in X and Y (never Z)
            if self.aug_flip and (np.random.rand() < 0.5):
                x_aug = torch.flip(x_aug, dims=[-1])  # flip X
                y_aug = torch.flip(y_aug, dims=[-1])
            if self.aug_flip and (np.random.rand() < 0.5):
                x_aug = torch.flip(x_aug, dims=[-2])  # flip Y
                y_aug = torch.flip(y_aug, dims=[-2])

            x_grid = x_aug.squeeze(0)
            y_grid = y_aug.squeeze(0)

            # Gaussian noise on inputs (exclude categorical channels)
            if self.noise_std > 0:
                noisy = x_grid.clone()
                for i in range(x_grid.shape[0]):
                    if i < static.shape[0] and i in self.noise_exclude_static_idx:
                        continue
                    noisy[i] = noisy[i] + torch.randn_like(noisy[i]) * self.noise_std
                x_grid = noisy

        return {
            "x_grid": x_grid.float(),  # [C_in, Z, Y, X]
            "params": params_condition.float(),  # [26]
            "y_grid": y_grid.float(),  # [2, Z, Y, X]
            "y_scalar": y_scalar.float(),  # [5]
        }
