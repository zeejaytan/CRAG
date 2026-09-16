# CR1 — Does CRAG's generative shape prior help on the Juglet where pure assembly fails?

**Status:** open — blocked on upstream checkpoint/data release · **Blocked by:** none · **Effort:** ~days once a Stage 1 or Stage 2 checkpoint exists

## Why it matters

TORA genuinely fails on the 9-piece Juglet and GARF does not close the shape either.
CRAG claims its generation branch resolves assembly ambiguities and can synthesise
missing geometry — the Juglet is incomplete (a piece was never recovered), so a method
that fills gaps is either the right tool or a convincing way to be wrong. This decides
whether CRAG earns a full Juglet campaign or stays a synthetic-data reference.

## Done when

- [ ] A CRAG assembly of the 9-piece Juglet scan, stated as fraction of sherds seated
      plus rotation error in degrees on misplaced sherds, scored against the
      hand-built `juglet_gt.hdf5` reference (TORA project, 2026-08-10)
- [ ] A rendered before/after at a view that shows whether the break faces meet — not a
      whole-pot thumbnail — with the conservator's verdict on whether the proposal is
      worth acting on
- [ ] An explicit call on the generated fill: does it mark the genuinely missing region,
      or invent geometry where bone-dry evidence says stop

## Gate / stop condition

If no upstream checkpoint appears and Stage 1 training from scratch is not funded, stop:
record the costed training plan and retire this question until the release lands. Do not
burn a two-stage training run to answer a question a zero-shot probe could settle.

## Source

CRAG release (`ai4ce/CRAG`, 2026-05-25; checkpoints unreleased as of 2026-09-16) vs the
TORA/GARF Juglet findings (`../../tora/intent/`, `../../GARF/intent/`).
