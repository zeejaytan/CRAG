# 04 — Medium run: full vessel split, 100k steps

Status: ready-for-agent · Answers: none (routine; serves CR1 without settling it)

Blocked by: 01 (converter). Unblocks: 03 (probe re-run).

## Spec

`data/bbad_vessels_full_crag.hdf5` (1108 train / 107 val, unscreened,
2–20 parts filter), same Stage-1 w/o-img recipe as 02, 100k steps in
5 × 20k chained segments (`scripts/hpc/crag_train_medium.slurm`,
`experiment_name=medium_vessels`), single A100, `val_check_interval=5000`.

## The one question

Does val part_acc move off ~7%? Move → scaling justified, talk full-scale.
Flat → stop and diagnose structural biases (full-mesh concat suspect #1)
before further compute. Either outcome is recorded as the finding it is —
flat here constrains the setup, never the method.
