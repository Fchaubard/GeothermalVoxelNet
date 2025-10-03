# Voxel ODE

3D autoregressive CNN for geothermal reservoir simulations. Predicts next-timestep Pressure/Temperature fields (3D grids) and 5 field-level scalars from current state.

## Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install h5py numpy tqdm wandb
```

## Data Format

### Input HDF5 Structure (v2.4 schema)

**Grid dimensions:** 326 (Z) × 70 (Y) × 76 (X)
**Time steps:** 30
**Each file:** ~760MB (full res) or ~20MB (downsampled)

```
Input/
  Static 3D grids (all 326×70×76):
    FaultId           - Fault zone IDs (categorical, int)
    IsActive          - Active/inactive cells (binary, 0/1)
    IsWell            - Well locations (binary, 0/1)
    InjRate           - Injection rate distribution (continuous)
    PermX             - X-permeability (continuous, log-scale)
    PermY             - Y-permeability (continuous, log-scale)
    PermZ             - Z-permeability (continuous, log-scale)
    Porosity          - Rock porosity (continuous, 0-1)
    Pressure0         - Initial pressure (Pa)
    Temperature0      - Initial temperature (K)

  Global parameters:
    ParamsScalar      - 26 simulation parameters (rock properties, boundary conditions, etc.)

Output/
  Time-evolving grids (30 × 326×70×76):
    Pressure          - Pressure field evolution (Pa)
    Temperature       - Temperature field evolution (K)

  Field-level scalars (30 timesteps each):
    FieldEnergyInjectionRate    - Total energy injection rate (W)
    FieldEnergyProductionRate   - Total energy production rate (W)
    FieldEnergyProductionTotal  - Cumulative energy produced (J)
    FieldWaterInjectionRate     - Total water injection rate (kg/s)
    FieldWaterProductionRate    - Total water production rate (kg/s)
```

**Total input channels:** 11 static grids + 2 lagged grids (Pressure, Temperature at t) = 13 channels
**Output channels:** 2 grids (Pressure, Temperature at t+1) + 5 scalars

## Data Prep

Your raw data is in `/workspace/omv/data/` (not `./data/valid/` - that's test data).

```bash
python scripts/prep_data.py \
  --input_root /workspace/omv/data \
  --out_root ./data/prepped \
  --patch_size 32 32 32 \
  --stride 8 8 8
```

**What this does:**
- Extracts overlapping 32³ patches with stride 8 (4× overlap per dimension)
- From grid 326×70×76, this creates ~37 × ~9 × ~10 = ~3,330 patches per simulation per timestep
- With 45 simulations × 29 training timesteps = ~4.3M training patches
- Computes global statistics (mean/std) for normalization
- Splits: first 5 indices → test, rest → train
- Packs into efficient HDF5

**Why dense patches with overlap:**
- Full 326×70×76 grid is too big for GPU memory
- Small patches (32³) fit easily, allow larger batch sizes
- Overlap (stride < patch_size) increases training data
- Model sees local physics in each patch
- At inference, can stitch patches or run on full grid (fully convolutional)

**Output structure:**
```
./data/prepped/
  train/*.prepped.h5       # Packed training simulations
  test/*.prepped.h5        # Packed test simulations
  stats.json               # Global normalization statistics
  index_train.jsonl        # List of all training patches (sim_path, t, z0:z1, y0:y1, x0:x1)
  index_test.jsonl         # List of all test patches
```

Each `.prepped.h5` file contains:
- `/static` (11, Z, Y, X) - stacked static input grids
- `/params_scalar` (26,) - global simulation parameters
- `/outputs_grid` (T, 2, Z, Y, X) - Pressure and Temperature evolution
- `/outputs_scalar` (T, 5) - field-level scalar outputs

## Model Architecture

**VoxelAutoRegressor:** 3D ResNet with FiLM conditioning

```
Input:  x_grid  [B, C_in, D, H, W]  where C_in = 11 static + 2 lagged = 13
        params  [B, 26]

Architecture:
  1. Stem: Conv3d(k=2r+1, padding=r) → C_in to base_channels
  2. ResNet blocks (depth × ResBlock):
       Conv3d(k=2r+1, padding=r) → LayerNorm → SiLU
       + FiLM conditioning (params → affine per channel)
       + residual connection
  3. Two output heads:
       Grid head:   Conv3d → 2 channels (Pressure, Temperature)
       Scalar head: AdaptiveAvgPool → Linear → 5 scalars

Output: grid_pred  [B, 2, D, H, W]
        scalar_pred [B, 5]
```

**Key design choices:**

1. **Local receptive field (kernel size = 2r+1):**
   - Default r=3 → 7×7×7 kernel
   - Physics is local (PDEs, diffusion, pressure propagation)
   - Larger r captures more spatial context but increases compute
   - Fully convolutional: works on any grid size

2. **FiLM conditioning (Feature-wise Linear Modulation):**
   - Global params (26 scalars) affect entire field
   - Each ResBlock: params → Linear → [scale, bias] per channel
   - Modulates features: `out = scale * features + bias`
   - Avoids broadcasting 26 channels across entire 3D grid (wasteful)

3. **ResNet blocks with LayerNorm:**
   - Residual connections stabilize deep networks
   - LayerNorm over spatial dims (better than BatchNorm for variable batch sizes)
   - SiLU activation (smoother than ReLU)

4. **Separate grid/scalar heads:**
   - Grid prediction needs spatial structure → Conv head
   - Scalars are field-level aggregates → Global pool + Linear
   - Different loss scales (grids ~1e7 Pa, scalars vary)

5. **Channel dimensions:**
   - `base_channels=128`: feature width (128 → 256 → 512 in deeper layers)
   - `depth=12`: number of ResBlocks (more = more capacity, slower)
   - Total params with r=3, base=128, depth=12: ~10-15M

## Training

**Multi-GPU (recommended):**
```bash
export WANDB_API_KEY=your_key_here
export WANDB_PROJECT=voxel-ode

torchrun --standalone --nproc_per_node=10 scripts/train_ddp.py \
  --prepped_root ./data/prepped \
  --batch_size 32 \
  --receptive_field_radius 3 \
  --base_channels 128 \
  --depth 12 \
  --max_steps 200000 \
  --eval_every 1000 \
  --log_every 100 \
  --ckpt_every 10000 \
  --aug_xy_rot \
  --aug_flip \
  --noise_std 0.01 \
  --use_amp \
  --use_wandb \
  --num_workers 0
```

**Single GPU:**
```bash
python scripts/train_ddp.py \
  --prepped_root ./data/prepped \
  --batch_size 8 \
  --receptive_field_radius 3 \
  --base_channels 128 \
  --depth 12 \
  --max_steps 200000 \
  --eval_every 100 \
  --log_every 20 \
  --ckpt_every 1000 \
  --aug_xy_rot \
  --aug_flip \
  --noise_std 0.01 \
  --use_amp
```

**Hyperparameters explained:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--receptive_field_radius` | 3 | Kernel = 2r+1 (7×7×7 default) |
| `--base_channels` | 128 | Feature width (128, 256, 512 typical) |
| `--depth` | 12 | Number of ResBlocks (8-16 typical) |
| `--batch_size` | 16 | Per-GPU batch size |
| `--lr` | 3e-4 | Learning rate (AdamW) |
| `--weight_decay` | 0.01 | AdamW weight decay |
| `--warmup_steps` | 1000 | Linear LR warmup |
| `--aug_xy_rot` | flag | Random 90° XY rotations (never Z/gravity) |
| `--aug_flip` | flag | Random XY flips |
| `--noise_std` | 0.01 | Gaussian noise std on inputs |
| `--use_amp` | flag | Automatic mixed precision (fp16) |
| `--num_workers` | 0 | DataLoader workers (keep 0 for HDF5) |

**Augmentations:**
- **XY rotations:** 0°, 90°, 180°, 270° in horizontal plane
- **XY flips:** horizontal/vertical
- **Z axis never augmented:** represents depth/gravity, must preserve
- **Gaussian noise:** adds regularization, simulates measurement uncertainty
- Custom collate function pads variable shapes from rotations

**Training details:**
- Optimizer: AdamW with linear warmup + cosine decay
- Loss: MSE for grids + MSE for scalars (weighted equally)
- Normalization: per-channel z-score using global stats
- Scalars: log1p transform for skewed distributions (energy/water rates)
- Evaluation: MSE and ±5% accuracy per output variable

Checkpoints save to `./checkpoints/`.

## Inference

Autoregressive rollout on full simulation:

```bash
python scripts/rollout_inference.py \
  --prepped_sim_path ./data/prepped/test/v2.4_0001.prepped.h5 \
  --stats_path ./data/prepped/stats.json \
  --ckpt_path ./checkpoints/best_step10000.pt \
  --save_path ./predictions.h5 \
  --receptive_field_radius 3 \
  --base_channels 128 \
  --depth 12
```

**What this does:**
- Loads initial conditions (t=0) from prepped file
- Autoregressively predicts t=1, t=2, ..., t=29
- Each step: use predicted Pressure/Temperature as input for next step
- Saves predictions to HDF5

**Output file structure:**
```
predictions.h5:
  predicted/outputs_grid   (30, 2, Z, Y, X)  - Pressure & Temperature predictions
  predicted/outputs_scalar (30, 5)           - Field scalar predictions
```

**Note:** Model is fully convolutional, so it can run on:
- Small patches (32×32×32) during training
- Full grid (326×70×76) during inference
- Any grid size in between

## Metrics

Reports per-output metrics:
- **MSE:** Mean squared error in normalized space
- **±5% accuracy:** Fraction of voxels/timesteps within 5% of ground truth (in physical units)

Tracked for:
- Grid outputs: Pressure, Temperature
- Scalar outputs: FieldEnergyInjectionRate, FieldEnergyProductionRate, FieldEnergyProductionTotal, FieldWaterInjectionRate, FieldWaterProductionRate

## Troubleshooting

**OOM (out of memory):**
- Reduce `--batch_size` (try 8, 4, 2)
- Reduce `--base_channels` (try 64)
- Reduce `--depth` (try 8)
- Use smaller patches in prep (--patch_size 24 24 24)

**Training too slow:**
- Check GPU utilization (should be 90%+)
- Set `--num_workers 0` for HDF5 data
- Increase `--batch_size` if GPU not saturated
- Use `--use_amp` for mixed precision

**Poor performance:**
- Check initial metrics at step 0 (should be reasonable, not NaN)
- Try without augmentations first (remove --aug_xy_rot --aug_flip)
- Reduce `--noise_std` or set to 0.0
- Check W&B curves for learning rate, loss trends
- Verify data prep used correct input_root

**Multi-GPU hanging:**
- NCCL P2P is disabled automatically in code
- Keep `--num_workers 0` for HDF5
- Check all GPUs visible: `nvidia-smi`

## Files

```
voxel_ode/
  data_prep.py      - Patch extraction, statistics computation
  dataset.py        - PyTorch dataset with augmentations, custom collate
  model.py          - VoxelAutoRegressor (3D ResNet + FiLM)
  schedulers.py     - Warmup + cosine LR schedule
  utils.py          - Utilities (seeding, stats, etc.)

scripts/
  prep_data.py              - Preprocess raw HDF5 to patches
  train_ddp.py              - DDP training script
  rollout_inference.py      - Autoregressive rollout

/workspace/omv/data/      - Raw HDF5 files (YOUR DATA HERE)
./data/prepped/           - Preprocessed patches + stats
./checkpoints/            - Model checkpoints
```

## Design Rationale

**Why patches instead of full grids?**
- 326×70×76 full grid = 1.7M voxels, too big for GPU memory with batch training
- 32³ patches = 32K voxels, fits easily, allows batch_size=16+
- Overlapping patches (stride < patch_size) increases training data from limited simulations
- Model is fully convolutional, so trains on patches but infers on any size

**Why local receptive field instead of global?**
- Reservoir physics governed by local PDEs (diffusion, Darcy flow, heat transfer)
- 7×7×7 kernel captures nearest-neighbor interactions
- Global context not needed for single-timestep prediction
- Fully convolutional architecture inherently captures multi-scale through depth

**Why XY-only augmentations?**
- Z axis represents depth/gravity (fundamental physical asymmetry)
- Rotating in Z would flip geological layers, break stratigraphy
- XY rotations preserve vertical structure, valid for horizontal reservoir symmetry

**Why log1p for scalars?**
- Energy/water rates span 5+ orders of magnitude
- Log transform stabilizes training, prevents large values from dominating loss
- log1p handles zeros gracefully (common in production/injection rates)

**Why FiLM conditioning instead of concatenation?**
- 26 global params broadcast to 326×70×76 grid = 26× memory waste
- FiLM learns affine transform per channel (lightweight, expressive)
- Allows global params to modulate features at every layer

**Why separate heads for grids vs scalars?**
- Grids need spatial resolution → convolutional head
- Scalars are field-level aggregates → global pooling head
- Different loss scales (Pressure ~1e7, scalars vary)
- Allows independent tuning of each output type

## Expected Performance

**Dataset size:**
- 45 simulations in `/workspace/omv/data/`
- First 5 → test, remaining 40 → train
- ~4M training patches (40 sims × 29 timesteps × 3,330 patches/sim)

**Training time (10× A40 GPUs, batch_size=16):**
- ~5-10 steps/sec
- 200K steps in ~5-10 hours
- Checkpoints every 1000 steps
- Eval every 100 steps (~30 sec eval)

**Memory usage:**
- Model: ~1-2 GB (r=3, base=128, depth=12)
- Batch: ~2-4 GB (batch_size=16, patch=32³)
- Total per GPU: ~3-6 GB (well within 46GB A40)
- Can increase batch_size to 32-64 if desired

**Inference:**
- Single simulation rollout (30 timesteps): ~5-10 seconds
- Runs on full 326×70×76 grid (no patching needed)
