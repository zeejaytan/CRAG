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
    data.batch_size=8 \
    loggers=csv \
    tags='["dummy_smoke"]'

echo ""
echo "=== SMOKE TEST COMPLETE ==="
