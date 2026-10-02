# 07 — Bone training run (features are informative — learn?)

Status: ready-for-agent · Answers: none (routine; serves CR1 without settling it)

Blocked by: 01 (converter). Follows 06 (bone AUC 0.627 — encoder sees
thick chunks, so training has something to work with).

## Spec

`scripts/hpc/crag_train_bone.slurm`: Stage-1 w/o-img on
`data/bone_crag.hdf5` (265 train / 82 val), batch 2, dummy-scale points,
`experiment_name=bone`, 40k steps in 5 × 8k chained segments, single A100,
`val_check_interval=1000`, csv.

## Done when

Val part_acc moves off chance → working assembly prior (wrong domain for
the Juglet, real nonetheless — then discuss what it unlocks). Flat again
despite informative features → recipe itself suspect (LR-at-scale,
multi_ref/randomization behavior); stop and say so.
