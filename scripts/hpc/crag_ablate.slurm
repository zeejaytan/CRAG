#!/bin/bash
# CRAG diagnosis runs: short single-GPU tests, no chains.
# Usage: sbatch crag_ablate.slurm <preset> <max_steps>
#   preset fullscale: paper-scale points, batch 2
#   preset nocanon:   dummy-scale points + canonicalize=false
#SBATCH --job-name=crag_ablate
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/gpfs/projects/punim2657/CRAG/crag_ablate_%j.log
set -euo pipefail

PRESET="${1:?usage: sbatch crag_ablate.slurm <fullscale|nocanon> <max_steps>}"
MAX_STEPS="${2:?usage: sbatch crag_ablate.slurm <fullscale|nocanon> <max_steps>}"

case "$PRESET" in
  fullscale)
    EXP=abl_fullscale
    EXTRA="data.num_points_per_object=20000 data.num_points_per_part=2048 data.min_points_per_part=512" ;;
  nocanon)
    EXP=abl_nocanon
    EXTRA="data.num_points_per_object=2048 data.num_points_per_part=256 data.min_points_per_part=20 +data.canonicalize=false" ;;" 
  *) echo "unknown preset: $PRESET" >&2; exit 2 ;;
esac

cd /data/gpfs/projects/punim2657/CRAG
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== ABLATE ${PRESET}: ${MAX_STEPS} steps ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# shellcheck disable=SC2086
HYDRA_FULL_ERROR=1 python train.py \
    experiment=stg_1_partnext \
    experiment_name=${EXP} \
    project_name="CRAG-ablate" \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.strategy=auto \
    trainer.max_steps=${MAX_STEPS} \
    trainer.val_check_interval=1000 \
    ckpt_path=null \
    +data.data_root.bbad_vessels=data/bbad_vessels_full_crag.hdf5 \
    "data.categories=[bbad_vessels]" \
    +data.up_axies.bbad_vessels=Z \
    data.num_workers=4 \
    data.batch_size=2 \
    model.image_drop_rate=1.0 \
    loggers=csv \
    tags='["ablate"]' \
    ${EXTRA}

echo "=== ABLATE ${PRESET} COMPLETE ==="
