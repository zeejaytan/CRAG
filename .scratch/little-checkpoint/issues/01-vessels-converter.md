# 01 — Convert TORA vessels (+ Juglet) to CRAG HDF5 schema

Status: done · Answers: none (routine; serves CR1 without settling it)

## Result (2026-09-24)

Converted `data/bbad_vessels_crag.hdf5` (410 train / 17 val, 229 MB) and read
back through the true `CragDataset` path: batches collate (17–19 parts,
12 views @224px). One converter bug caught by the verify step (scalar-void
renderings unreadable → uint8 arrays). `up_axis: Z` assumed (BreakingBad
convention); low stakes — `random_rotation` washes global orientation.

## Problem

`CragDataset.get_data` needs per item: `pieces/{p}/vertices+faces`,
`pieces_names`, `{item}/full_mesh/vertices+faces`, `renderings/{k}` image
bytes, and `data_split/{category}/{split}` lists. TORA's
`bbad_vessels_v3.hdf5` (and GARF's `juglet_deploy.hdf5`) have pieces only.

## Fix

`scripts/convert_vessels_to_crag.py` (committed, runs on Spartan login node):
source HDF5 + split names + category in, CRAG-schema HDF5 out under `data/`
(gitignored). Per item: copy pieces + names, concat pieces to `full_mesh`
(faces offset), 12 deterministic gray PNGs as `renderings`, rebuilt
`data_split`. Runs: (a) vessels `train_screened`/`val_screened` → category
`bbad_vessels`; (b) Juglet sample → category `artifact` (same script, probe
input for 03).

## Verify (login node, CPU, no GPU)

`CragDataset.get_data` on 2 converted vessel items + the Juglet item:
meshes non-degenerate, `full_pointcloud` sampled, batch collates. Then
`test.py` config composition for the probe (03) while here.
