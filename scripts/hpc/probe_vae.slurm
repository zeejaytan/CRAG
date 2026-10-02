#!/bin/bash
# VAE feature probe (ticket 06): needs CUDA (flash_attn varlen kernels).
#SBATCH --job-name=probe_vae
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/gpfs/projects/punim2657/CRAG/probe_vae_%j.log
set -euo pipefail

cd /data/gpfs/projects/punim2657/CRAG
source .venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

python scripts/probe_vae_features.py --hdf5 data/bbad_vessels_full_crag.hdf5 \
    --category bbad_vessels --items 12 2>&1 | grep -vE "fallback|registered at|UserWarning|warnings.warn"

echo "=== PROBE VAE COMPLETE ==="
