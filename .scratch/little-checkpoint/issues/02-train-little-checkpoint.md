# 02 — Train little checkpoint (Stage 1, vessels, w/o img)

Status: done · Answers: none (routine; serves CR1 without settling it)

Blocked by: 01. Unblocks: 03.

## Result (2026-09-25)

Chain `31207108→…→31207112`: all COMPLETED, 20k steps (100 epochs), ~65 min.
Checkpoints: `output/little_vessels/last.ckpt` + `steps-step=020000.ckpt`
(4.9 GB each). Train loss 0.59→0.57→0.42→0.53→0.78 (noisy, batch 2);
val part_acc flat at 0.065–0.074, rmse_r ~78–80° (≈ chance). Verdict:
pipeline fully proven (resume ×4, checkpoints, val + viz outputs); the model
learned nothing generalizable — expected at 1/640th of the paper's data
volume, and NOT evidence about CRAG either way.

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
