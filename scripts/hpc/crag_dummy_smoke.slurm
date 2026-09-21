#!/bin/bash
# CRAG Stage-1 viability smoke: real CragUnifiedModel + synthetic dummy batches.
# No dataset download needed; proves the train loop, loss flow and checkpoint
# save/load on one GPU. TripoSG weights must exist at ./pretrained/TripoSG
# (DinoV2 + shape VAE load even with enable_generation=false).
#SBATCH --job-name=crag_dummy_smoke
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/gpfs/projects/punim2657/CRAG/crag_dummy_smoke_%j.log
set -euo pipefail

cd /data/gpfs/projects/punim2657/CRAG
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== SMOKE TEST: CRAG Stage 1 under shared env (dummy data) ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import torch, pytorch3d; print('torch', torch.__version__, '| pt3d', pytorch3d.__version__)"
echo ""

HYDRA_FULL_ERROR=1 python train.py \
    experiment=stg_1_partnext \
    data=crag_dummy \
    experiment_name=dummy_smoke_$(date +%Y%m%d_%H%M%S) \
    project_name="CRAG-dummy-smoke" \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.strategy=auto \
    trainer.max_steps=150 \
    trainer.val_check_interval=50 \
    data.num_workers=4 \
    data.batch_size=2 \
    data.num_points_per_object=2048 \
    data.num_points_per_part=256 \
    data.min_points_per_part=20 \
    loggers=csv \
    tags='["dummy_smoke"]' \
    '~callbacks.visualization_saving'
    # ^ drop the viz callback: it needs real renderings/meshes per batch
    # (raw_renderings, part meshes), which synthetic dummy batches lack.
    # NOTE 2026-09-21: stg_1_partnext scales data to 20000 pts/object, which
    # OOMs an 80 GB A100 at batch 8 (job 30822155: first step loss 1.29, then
    # OOM in the VAE/norm forward). Dummy-scale overrides below prove the loop;
    # full-scale memory sizing belongs to the real-data ticket, not this smoke.

echo ""
echo "=== SMOKE TEST COMPLETE ==="
