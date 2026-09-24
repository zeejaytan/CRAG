# 03 — Probe: FRACTURA ceramics, then Juglet (zero-shot)

Status: needs-info · Answers: none (routine; first CRAG ceramics number, not a CR1 verdict)

Blocked by: 01, 02.

## Spec

Convert `fractura_real.hdf5` ceramics (8 objects) + `juglet_deploy.hdf5`
with the 01 converter (never trained on — held out). `test.py` with the 02
checkpoint, single GPU. Score ceramics with standard assembly metrics;
score the Juglet against TORA's hand-built `juglet_gt.hdf5` (fraction seated
+ rotation error on misplaced sherds) AND render a break-face close-up
before/after for the conservator's verdict (geometry rule — numbers alone
decide nothing here).

## Done when

Per-CR1 read-out at small scale: what seated, what didn't, what it means.
A weak-checkpoint failure is inconclusive about CRAG, recorded as such —
it must never be written up as "the method failed".
