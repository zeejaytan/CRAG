# 03 — Probe: FRACTURA ceramics, then Juglet (zero-shot)

Status: done · Answers: none (routine; first CRAG ceramics number, not a CR1 verdict)

## Result (2026-09-28) — probes ran, outputs degenerate, no eye spent

- Ceramics (little ckpt): part_acc 0.215, rmse_r 69°. Juglet (medium ckpt):
  part_acc 0.111 (= 1/9), rmse_r 68°.
- Render-before-reporting caught what the numbers hide: proposal meshes are
  BIT-IDENTICAL to the stored arrangement (max abs diff 0.0, Juglet all three
  scenes incl. input; ceramics narrow_bottle4 same). The model outputs
  ~identity flow — nothing moves; the metric is reference-part credit only.
  The "21%" is a measurement reading, not 21% seated.
- A staged visual-qa pair of identical meshes was built then removed: a null
  look must never reach the conservator. No `Needs-eye` spent; nothing to
  witness until a checkpoint actually moves sherds.
- Coherent with flat val (04): the setup never learned, so all probes show
  the prior mean. Next: structural diagnosis per 04 (sampling density,
  full-mesh bias), not more steps.

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
