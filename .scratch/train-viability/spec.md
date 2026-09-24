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

## Result — harness complete (2026-09-21)

Chain `30886021→22→23→24` (4 × ~4k-step segments, `afterok` cascade): all
COMPLETED, ~65 min total. Resume worked 3/3 (Lightning picked up the global
step each time; one benign warning about non-resumable dataloader mid-epoch).
Final loss 0.0025 — the identity-pose dummy task memorized, exactly as
predicted: this proves stability + resume + checkpointing
(`output/harness/last.ckpt` 4.9 GB, plus `steps-step=010000.ckpt`), not
learning. Throughput at dummy scale: ~4 steps/s single A100.
Next gate (separate ticket): real CRAG-schema HDF5s or the upstream checkpoint
release — that is what CR1 waits on.
