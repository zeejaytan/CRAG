# 04 — Medium run: full vessel split, 100k steps

Status: done · Answers: none (routine; serves CR1 without settling it)

## Result (2026-09-26)

All 5 segments COMPLETED (100k steps / 180 epochs, ~5.5 GPU-hours). Val
part_acc flat: 0.093 → 0.091 → 0.092 → 0.092 → 0.093; rmse_r ~79.4° (≈
chance); chamfer 0.0216 → 0.0194 (negligible). Train loss noisy 0.42–0.78.
Per the ticket's stop rule this is the FLAT branch: do not scale further.
Flat constrains the setup (data volume? full-mesh concat bias? 2k-point
sampling of thin walls per TORA O1?), never the method. Checkpoints kept:
`output/medium_vessels/last.ckpt`.

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
