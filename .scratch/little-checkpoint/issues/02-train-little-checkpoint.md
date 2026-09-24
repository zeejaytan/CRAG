# 02 — Train little checkpoint (Stage 1, vessels, w/o img)

Status: needs-info · Answers: none (routine; serves CR1 without settling it)

Blocked by: 01. Unblocks: 03.

## Spec

`experiment=stg_1_partnext`, data override to converted vessels
(`data_root.bbad_vessels`, `categories=[bbad_vessels]`, `up_axies` Z),
`model.image_drop_rate=1.0`, dummy-smoke scale first
(batch 2, 2048 pts/object) then full scale only if memory allows. 20k steps,
single A100, chained resume segments per `crag_harness.slurm` pattern (~2–3
days). `loggers=csv`. Viz callback stays ON for real data (fields exist now).

## Done when

Loss falls on train, val `part_acc` reads above chance on vessels, `last.ckpt`
+ step checkpoints saved. A flat or diverged curve stops the ticket: report,
don't extend.

Needs-info: confirm step budget + GPU count at submit time (queue reality).
