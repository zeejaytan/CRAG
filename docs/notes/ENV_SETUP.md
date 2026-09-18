# CRAG / GARF shared-env setup notes

Shared env: `/data/gpfs/projects/punim2657/CRAG/.venv` (2026-09-16) — cloned from
GARF `.venv` (9.3 GB copy; next time prefer `cp -al`), topped up with
`jaxtyping`, `tyro`, `pygltflib`, `diso`. Proven by `scripts/hpc/env_setup.slurm`
(`ALL ENV CHECKS PASSED`) + GARF 5-epoch smoke (`scripts/hpc/garf_smoke_crag_env.slurm`,
job 30694968, 2026-09-18 — genuinely under this env; see §3 for why the 2026-09-16
run of the same script did not count).

## 1. Do NOT upgrade pytorch3d to the CRAG-pinned 0.7.8 wheel on Spartan

The wheel (`pytorch3d-0.7.8+5043d15...cu128`, MiroPsota builder) links against
**glibc 2.35**. Spartan GPU nodes (checked spartan-gpgpu127, A100) have **glibc
2.34** — the wheel installs cleanly but fails at import with
`ImportError: /lib64/libm.so.6: version GLIBC_2.35 not found (required by _C...so)`.
Fix used: copy GARF's Spartan-built **0.7.7** (`pytorch3d/` + egg-info from
GARF `.venv`) over the 0.7.8 install. Safe because CRAG only calls stable APIs
(`transforms`, `chamfer_distance` — grep `source/` to re-verify after upstream pulls).

## 3. A `cp`-cloned venv keeps the OLD absolute path — repair it before trusting `activate`

The env was made with a plain 9.3 GB copy of GARF's `.venv`, not
`virtualenv-clone`. So `bin/activate` (`VIRTUAL_ENV=.../GARF/.venv`) and every
console-script shebang (`torchrun`, `hf`, …) kept pointing at the GARF checkout.
Two consequences, both silent until 2026-09-18:

- The 2026-09-16 "GARF smoke under shared CRAG env" actually ran **GARF's**
  binaries (its log warns from `.../GARF/.venv/...` paths) — the shared-env
  claim was unproven. Only the `env_setup` import checks (which used
  `$CRAG/.venv/bin/python` directly) genuinely tested the CRAG env.
- Deleting GARF's `.venv` (2026-09-18, single-env switch) broke `activate`
  entirely: PATH led with a dead directory and fell through to the wrong python
  (`No module named 'hydra'`).

Fix (no reinstall needed — package bodies are path-independent; only `bin/`
text files carry the prefix):

```bash
V=/data/gpfs/projects/punim2657/CRAG/.venv
grep -rl "/data/gpfs/projects/punim2657/GARF/.venv" $V/bin $V/pyvenv.cfg \
  | while read f; do sed -i "s|/data/gpfs/projects/punim2657/GARF/.venv|/data/gpfs/projects/punim2657/CRAG/.venv|g" "$f"; done
source $V/bin/activate && which python  # must print the CRAG venv python
```

Verify any future clone the same way: `source bin/activate && which python`,
and check a job log's `.../.venv/...` warning paths name the env you think ran.
Next time prefer `cp -al` (hardlinks) plus this fixup, or `uv sync` from scratch.

## 2. Install `diso` with `--no-build-isolation`

`diso==0.1.4` needs `torch` at build time but does not declare it, so an isolated
build fails with `ModuleNotFoundError: No module named 'torch'`. Spartan's uv is
**0.7.15**: it does not understand CRAG `pyproject.toml`'s
`[tool.uv.extra-build-dependencies]` (warns, harmless) — so do **not** `uv sync`
the CRAG pyproject here. Instead: `uv pip install --python <venv> --no-build-isolation diso`
(torch 2.8 + setuptools 80.9 already in env satisfy the build).
