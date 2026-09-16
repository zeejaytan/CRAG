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

echo "=== top-up: jaxtyping tyro pygltflib ==="
"$UV" pip install --python "$PY" jaxtyping tyro pygltflib

echo "=== top-up: diso (no build isolation: torch+setuptools already in env) ==="
"$UV" pip install --python "$PY" --no-build-isolation diso

echo "=== pytorch3d: sync GARF's working 0.7.7 build ==="
# The CRAG-pinned 0.7.8 wheel needs glibc 2.35; Spartan GPU nodes have 2.34,
# so it installs but cannot import. GARF's 0.7.7 was built on Spartan and
# works, and CRAG only uses stable APIs (transforms, chamfer_distance).
GARF_SP=/data/gpfs/projects/punim2657/GARF/.venv/lib/python3.12/site-packages
CRAG_SP="$CRAG_ROOT/.venv/lib/python3.12/site-packages"
rm -rf "$CRAG_SP/pytorch3d" "$CRAG_SP"/pytorch3d-*.dist-info "$CRAG_SP"/pytorch3d-*.egg-info
cp -r "$GARF_SP/pytorch3d" "$GARF_SP/pytorch3d-0.7.7-py3.12.egg-info" "$CRAG_SP/"

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
