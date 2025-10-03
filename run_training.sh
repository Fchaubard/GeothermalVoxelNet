#!/bin/bash
# Production training script for 10 GPUs on full 326×70×76 data

# FIRST: Prep the data
# python scripts/prep_data.py \
#   --input_root /workspace/omv/data \
#   --out_root ./data/prepped \
#   --patch_size 32 32 32 \
#   --stride 8 8 8

# THEN: Run training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export WANDB_PROJECT=voxel-ode

torchrun \
  --standalone \
  --nproc_per_node=10 \
  scripts/train_ddp.py \
    --prepped_root ./data/prepped \
    --batch_size 16 \
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
    --use_amp \
    --use_wandb \
    --num_workers 0
