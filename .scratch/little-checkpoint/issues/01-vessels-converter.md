# 01 — Convert TORA vessels (+ Juglet) to CRAG HDF5 schema

Status: ready-for-agent · Answers: none (routine; serves CR1 without settling it)

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
