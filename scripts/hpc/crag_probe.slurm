#!/bin/bash
# CRAG zero-shot probe: evaluate a checkpoint on converted held-out data.
# Usage: sbatch crag_probe.slurm <run_name> <hdf5> <category> <ckpt>
#SBATCH --job-name=crag_probe
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/gpfs/projects/punim2657/CRAG/crag_probe_%j.log
set -euo pipefail

NAME="${1:?usage: sbatch crag_probe.slurm <name> <hdf5> <category> <ckpt>}"
HDF5="${2:?usage: sbatch crag_probe.slurm <name> <hdf5> <category> <ckpt>}"
CAT="${3:?usage: sbatch crag_probe.slurm <name> <hdf5> <category> <ckpt>}"
CKPT="${4:?usage: sbatch crag_probe.slurm <name> <hdf5> <category> <ckpt>}"

cd /data/gpfs/projects/punim2657/CRAG
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

echo "=== PROBE ${NAME}: ${CAT} @ ${CKPT} ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

HYDRA_FULL_ERROR=1 python test.py \
    experiment=stg_1_partnext \
    experiment_name=${NAME} \
    project_name="CRAG-probe" \
    ckpt_path=${CKPT} \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.strategy=auto \
    "+data.data_root.${CAT}=${HDF5}" \
    "data.categories=[${CAT}]" \
    "+data.up_axies.${CAT}=Z" \
    data.num_workers=4 \
    data.batch_size=2 \
    data.num_points_per_object=2048 \
    data.num_points_per_part=256 \
    data.min_points_per_part=20 \
    model.image_drop_rate=1.0 \
    loggers=csv \
    tags='["probe"]'

echo "=== PROBE ${NAME} COMPLETE ==="
