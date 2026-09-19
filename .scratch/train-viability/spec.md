# Train viability — prove CRAG Stage 1 can actually train here

Goal: a real `CragUnifiedModel` training step on Spartan, starting with zero
data dependencies (synthetic `crag_dummy` batches), before any decision about
real datasets or a Juglet probe.

Ladder:

1. Dummy smoke (150 steps, 1 GPU, `scripts/hpc/crag_dummy_smoke.slurm`) —
   proves forward, loss, checkpoint save/load. Needs `pretrained/TripoSG`
   (DinoV2 + shape VAE load even with generation off) — downloaded 2026-09-18.
2. Real-data decision (BreakingBad HDF5 via OneDrive, or GARF→CRAG conversion,
   or wait for upstream release). Separate ticket when step 1 passes.
3. Juglet zero-shot probe only after step 1–2. Never train *on* the Juglet
   (9 fragments, one incomplete vessel — memorizes, proves nothing).

Related: `intent/CR1-does-generation-help-the-juglet.md` (this ladder is the
precondition, not the answer).
