#!/bin/bash
# CRAG harness run: long dummy-scale Stage-1 training in chained segments.
# Proves multi-hour stability + checkpoint resume (each segment resumes from
# the previous one's last.ckpt). Synthetic identity-pose batches, so this
# calibrates throughput and robustness, NOT learning — real-data signal needs
# CRAG-schema HDF5s (see spec).
# Usage: sbatch crag_harness.slurm <max_steps> <ckpt_path-or-null> <run_tag>
#SBATCH --job-name=crag_harness
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/gpfs/projects/punim2657/CRAG/crag_harness_%j.log
set -euo pipefail

MAX_STEPS="${1:?usage: sbatch crag_harness.slurm <max_steps> <ckpt|null> <tag>}"
CKPT="${2:?usage: sbatch crag_harness.slurm <max_steps> <ckpt|null> <tag>}"
TAG="${3:?usage: sbatch crag_harness.slurm <max_steps> <ckpt|null> <tag>}"

cd /data/gpfs/projects/punim2657/CRAG
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== HARNESS SEGMENT ${TAG}: target ${MAX_STEPS} steps, ckpt ${CKPT} ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

HYDRA_FULL_ERROR=1 python train.py \
    experiment=stg_1_partnext \
    data=crag_dummy \
    experiment_name=harness \
    project_name="CRAG-harness" \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.strategy=auto \
    trainer.max_steps=${MAX_STEPS} \
    trainer.val_check_interval=1000 \
    ckpt_path=${CKPT} \
    data.num_workers=4 \
    data.batch_size=2 \
    data.num_points_per_object=2048 \
    data.num_points_per_part=256 \
    data.min_points_per_part=20 \
    loggers=csv \
    tags='["harness"]' \
    '~callbacks.visualization_saving'

echo "=== SEGMENT ${TAG} COMPLETE ==="
