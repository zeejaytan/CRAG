#!/bin/bash
# GARF 5-epoch smoke test, run with the SHARED (CRAG-topped-up) env to prove
# one env serves both projects. Port of GARF slurm/smoke_test.slurm; the only
# difference is which .venv is activated.
# Run inside a holder step: srun --jobid=<HOLDER> bash scripts/hpc/garf_smoke_crag_env.slurm
# (launched from the CRAG checkout; it cds to GARF itself).
#SBATCH --job-name=garf_smoke_shared_env
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/gpfs/projects/punim2657/CRAG/garf_smoke_shared_env_%j.log
set -euo pipefail

cd /data/gpfs/projects/punim2657/GARF
source /data/gpfs/projects/punim2657/CRAG/.venv/bin/activate

export CUDA_HOME=/apps/easybuild-2022/easybuild/software/Core/CUDA/12.5.1
export PATH=$CUDA_HOME/bin:$PATH

echo "=== SMOKE TEST: GARF Flow Matching under shared CRAG env ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -c "import torch, pytorch3d; print('torch', torch.__version__, '| pt3d', pytorch3d.__version__)"
echo ""

HYDRA_FULL_ERROR=1 python train.py \
    experiment=denoiser_flow_matching \
    experiment_name=smoke_shared_env_$(date +%Y%m%d_%H%M%S) \
    project_name="GARF-smoke-shared-env" \
    data.data_root=/data/gpfs/projects/punim2657/GARF/input/Fractura/bone_synthetic.hdf5 \
    data.categories="['pig']" \
    data.num_workers=4 \
    data.batch_size=8 \
    data.multi_ref=True \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.max_epochs=5 \
    trainer.check_val_every_n_epoch=5 \
    trainer.precision=bf16-mixed \
    model.feature_extractor_ckpt=output/feature_extractor.ckpt \
    loggers=csv \
    tags='["smoke_test_shared_env"]'

echo ""
echo "=== SMOKE TEST COMPLETE ==="
