# 01 — Restore missing `source/utils/data.py` (upstream gap)

Status: ready-for-agent · Answers: none (routine unblock for the viability ladder)

## Problem

`source/datasets/crag.py` imports `make_seed`, `pytorch_worker_seed` and
`worker_init_fn` from `source.utils.data`, but that module is absent from the
release — upstream `main` has the same 5 files under `source/utils/`, so this
is not checkout staleness. Nothing CRAG-side can train: both the real and the
dummy datamodules fail at Hydra instantiate
(`ModuleNotFoundError: No module named 'source.utils.data'`, dummy smoke
job 30711469, 2026-09-18).

## Fix

Add `source/utils/data.py` with the three worker-seeding helpers, semantics
taken from the call sites (`crag.py:64,162,173,184,590`):

- `pytorch_worker_seed()` — per-worker int from `torch.initial_seed()` (+ worker id).
- `make_seed(*args)` — hash arbitrary entropy (pid, time_ns, urandom) to a
  32-bit seed for the iterable-dataset shuffling RNG.
- `worker_init_fn(worker_id)` — standard DataLoader hook seeding numpy/random.

No behavior change to any existing file; purely additive. If upstream later
ships its own `data.py`, diff and prefer theirs.

## Verify (all on Spartan, cheap first)

1. Login node: `import source.datasets.crag_dummy`, one dummy batch via
   `CragDataset.collate_fn`, CPU model build.
2. Resubmit `scripts/hpc/crag_dummy_smoke.slurm`, expect `SMOKE TEST COMPLETE`.
