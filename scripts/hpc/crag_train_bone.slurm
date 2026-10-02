#!/bin/bash
# CRAG bone training: Stage 1 assembly-only on thick fracture chunks
# (265 train / 82 val), w/o img. 40k steps in chained segments.
# Usage: sbatch crag_train_bone.slurm <max_steps> <ckpt|null> <tag>
#SBATCH --job-name=crag_train_bone
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/gpfs/projects/punim2657/CRAG/crag_train_bone_%j.log
set -euo pipefail

MAX_STEPS="${1:?usage: sbatch crag_train_bone.slurm <max_steps> <ckpt|null> <tag>}"
CKPT="${2:?usage: sbatch crag_train_bone.slurm <max_steps> <ckpt|null> <tag>}"
TAG="${3:?usage: sbatch crag_train_bone.slurm <max_steps> <ckpt|null> <tag>}"

cd /data/gpfs/projects/punim2657/CRAG
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== BONE SEGMENT ${TAG}: target ${MAX_STEPS} steps, ckpt ${CKPT} ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

HYDRA_FULL_ERROR=1 python train.py \
    experiment=stg_1_partnext \
    experiment_name=bone \
    project_name="CRAG-bone" \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.strategy=auto \
    trainer.max_steps=${MAX_STEPS} \
    trainer.val_check_interval=1000 \
    ckpt_path=${CKPT} \
    +data.data_root.bone=data/bone_crag.hdf5 \
    "data.categories=[bone]" \
    +data.up_axies.bone=Z \
    data.num_workers=4 \
    data.batch_size=2 \
    data.num_points_per_object=2048 \
    data.num_points_per_part=256 \
    data.min_points_per_part=20 \
    model.image_drop_rate=1.0 \
    loggers=csv \
    tags='["bone"]'

echo "=== SEGMENT ${TAG} COMPLETE ==="
