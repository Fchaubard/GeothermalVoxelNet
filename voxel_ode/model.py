import torch
import torch.nn as nn


class LayerNorm3d(nn.Module):
    """3D LayerNorm using GroupNorm with 1 group."""

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps, affine=True)

    def forward(self, x):
        # x: [B, C, Z, Y, X]
        return self.gn(x)


class ResBlock3d(nn.Module):
    """3D Residual block with two conv layers."""

    def __init__(self, c, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv3d(c, c, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm1 = LayerNorm3d(c)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(c, c, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = LayerNorm3d(c)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(x + h)


class VoxelAutoRegressor(nn.Module):
    """
    3D CNN for autoregressive prediction of voxel grids.

    Predicts:
    - Grid outputs (Pressure, Temperature) at t+1
    - Scalar field outputs (5 values) at t+1

    From:
    - Static 3D inputs (geology, wells, etc.)
    - Dynamic grid at t (Pressure, Temperature)
    - Optional scalar parameters (26 values)
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        depth: int = 8,
        r: int = 2,
        cond_params_dim: int = 26,
        use_param_broadcast: bool = False,
        grid_out_channels: int = 2,  # Pressure, Temperature
        scalar_out_dim: int = 5,
    ):
        super().__init__()
        k = 2 * r + 1  # receptive field kernel size
        self.use_param_broadcast = use_param_broadcast
        self.cond_params_dim = cond_params_dim

        # Stem: large kernel to capture receptive field
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=k, padding=r, bias=False),
            LayerNorm3d(base_channels),
            nn.SiLU(inplace=True),
        )

        # Residual trunk
        blocks = []
        for i in range(depth):
            # Vary dilation to extend receptive field
            dilation = 1 #if i < depth // 2 else 2
            blocks.append(ResBlock3d(base_channels, dilation=dilation))
        self.trunk = nn.Sequential(*blocks)

        # FiLM-like conditioning from scalar parameters
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_params_dim, base_channels),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels, base_channels),
        )

        # Grid prediction head
        self.grid_head = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm3d(base_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(base_channels, grid_out_channels, kernel_size=1),
        )

        # Scalar prediction head (global pooling)
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_channels, base_channels),
            nn.SiLU(inplace=True),
            nn.Linear(base_channels, scalar_out_dim),
        )

    def forward(self, x_grid, params_scalar=None):
        """
        Args:
            x_grid: [B, C_in, Z, Y, X] - concatenated static + dynamic inputs
            params_scalar: [B, 26] - optional scalar parameters

        Returns:
            grid_out: [B, 2, Z, Y, X] - predicted Pressure & Temperature
            scalar_out: [B, 5] - predicted field scalars
        """
        h = self.stem(x_grid)

        # Apply conditioning from scalar parameters
        if params_scalar is not None:
            bias = self.cond_mlp(params_scalar).view(-1, h.size(1), 1, 1, 1)
            h = h + bias

        h = self.trunk(h)

        grid_out = self.grid_head(h)
        scalar_out = self.scalar_head(h)

        return grid_out, scalar_out
