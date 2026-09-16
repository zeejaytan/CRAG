# CRAG / GARF shared-env setup notes

Shared env: `/data/gpfs/projects/punim2657/CRAG/.venv` (2026-09-16) — cloned from
GARF `.venv` (9.3 GB copy; next time prefer `cp -al`), topped up with
`jaxtyping`, `tyro`, `pygltflib`, `diso`. Proven by `scripts/hpc/env_setup.slurm`
(`ALL ENV CHECKS PASSED`) + GARF 5-epoch smoke (`scripts/hpc/garf_smoke_crag_env.slurm`).

## 1. Do NOT upgrade pytorch3d to the CRAG-pinned 0.7.8 wheel on Spartan

The wheel (`pytorch3d-0.7.8+5043d15...cu128`, MiroPsota builder) links against
**glibc 2.35**. Spartan GPU nodes (checked spartan-gpgpu127, A100) have **glibc
2.34** — the wheel installs cleanly but fails at import with
`ImportError: /lib64/libm.so.6: version GLIBC_2.35 not found (required by _C...so)`.
Fix used: copy GARF's Spartan-built **0.7.7** (`pytorch3d/` + egg-info from
GARF `.venv`) over the 0.7.8 install. Safe because CRAG only calls stable APIs
(`transforms`, `chamfer_distance` — grep `source/` to re-verify after upstream pulls).

## 2. Install `diso` with `--no-build-isolation`

`diso==0.1.4` needs `torch` at build time but does not declare it, so an isolated
build fails with `ModuleNotFoundError: No module named 'torch'`. Spartan's uv is
**0.7.15**: it does not understand CRAG `pyproject.toml`'s
`[tool.uv.extra-build-dependencies]` (warns, harmless) — so do **not** `uv sync`
the CRAG pyproject here. Instead: `uv pip install --python <venv> --no-build-isolation diso`
(torch 2.8 + setuptools 80.9 already in env satisfy the build).
