# 02 — Restore missing `source/utils/lr_scheduler.py` (upstream gap)

Status: ready-for-agent · Answers: none (routine unblock for the viability ladder)

## Problem

`configs/model/crag_unified.yaml` points `lr_scheduler._target_` at
`source.utils.lr_scheduler.LambdaWarmUpCosineFactorScheduler`, but that module
is absent from the release — verified against upstream `main` (`FETCH_HEAD`,
2026-09-19): same 5 files under `source/utils/`, only the yaml names the
class. Dummy smoke job 30752842 (2026-09-19) failed at scheduler instantiate,
after 01 fixed the datamodule import.

## Contract (from `crag_unified.py:1141-1152`)

Instantiated by Hydra with the yaml params
(`warm_up_steps: 1000, f_start: 1e-6, f_min: 1e-3, f_max: 1.0`) plus
`optimizer=` and `max_decay_steps=<trainer.max_steps>` kwargs; the instance is
passed as `lr_lambda` to `torch.optim.lr_scheduler.LambdaLR` stepped every
step. So it must be a callable `step -> multiplicative factor`: warm up
`f_start -> f_max`, then cosine-decay `f_max -> f_min`.

## Verify

1. Login node: import every `source.*` target referenced by configs (sweep —
   stop the one-job-per-missing-file bleed).
2. Resubmit `scripts/hpc/crag_dummy_smoke.slurm`, expect `SMOKE TEST COMPLETE`.
