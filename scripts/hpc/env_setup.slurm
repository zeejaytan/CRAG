#!/bin/bash
#SBATCH --job-name=crag-env-setup
#SBATCH --partition=gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/projects/punim2657/CRAG/env_setup_%j.log

# Top up the CRAG env (cloned from GARF .venv) with CRAG-only deps,
# upgrade pytorch3d to the CRAG-pinned 0.7.8 wheel, then import-test
# everything incl. GARF model imports (shared-env check).
set -euo pipefail

CRAG_ROOT=/data/gpfs/projects/punim2657/CRAG
PY="$CRAG_ROOT/.venv/bin/python"
UV="$HOME/.local/bin/uv"

module purge
module load CUDA/12.8.0
module load GCC/13.3.0
export CUDA_HOME="${CUDA_HOME:-$EBROOTCUDA}"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

echo "=== top-up: jaxtyping tyro pygltflib diso ==="
"$UV" pip install --python "$PY" jaxtyping tyro pygltflib diso

echo "=== upgrade pytorch3d -> CRAG-pinned 0.7.8 wheel ==="
"$UV" pip install --python "$PY" --upgrade \
  "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8%2B5043d15/pytorch3d-0.7.8+5043d15pt2.8.0cu128-cp312-cp312-linux_x86_64.whl"

echo "=== import tests ==="
"$PY" - <<'PY'
import sys
print("python:", sys.version.split()[0])
import torch
print("torch:", torch.__version__, "cuda build:", torch.version.cuda,
      "available:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import flash_attn
print("flash_attn:", flash_attn.__version__)
import pytorch3d
print("pytorch3d:", pytorch3d.__version__)
from pytorch3d import transforms
import torch as t
m = transforms.quaternion_to_matrix(t.tensor([[1., 0., 0., 0.]], device="cuda"))
print("pytorch3d cuda transform: OK", tuple(m.shape))
import spconv.pytorch as spconv
import torch_scatter, torch_cluster
print("spconv + scatter + cluster: OK")
import transformers, diffusers, accelerate
print("transformers/diffusers/accelerate: OK")
import diso, jaxtyping, tyro, pygltflib
print("diso/jaxtyping/tyro/pygltflib: OK")
PY

echo "=== GARF model imports (shared-env check) ==="
cd /data/gpfs/projects/punim2657/GARF
"$PY" - <<'PY'
from assembly.models.denoiser.denoiser_flow_matching import DenoiserFlowMatching
from assembly.models.denoiser.denoiser_diffusion import DenoiserDiffusion
from assembly.data.breaking_bad.module import BreakingBadDataModule
print("GARF model imports: OK")
PY

echo "=== CRAG source import ==="
cd "$CRAG_ROOT"
"$PY" -c "import source.models.crag_unified; print('CRAG source import: OK')"

echo "=== ALL ENV CHECKS PASSED ==="
