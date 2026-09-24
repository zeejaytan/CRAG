# Little checkpoint — thin-walled vessels → Juglet probe

Goal: the smallest CRAG checkpoint that can honestly attempt the Juglet:
trained on thin-walled BreakingBad vessels (TORA's `bbad_vessels_v3`, the
closest domain on disk), probed zero-shot on FRACTURA ceramics, then the
Juglet. Explicitly NOT a CR1 verdict — a pipeline proof plus the first real
CRAG number on ceramics.

Why vessels: a vessel fracture is a ribbon through a wall. TORA O1 measured
the median training vessel at 0.78 sampling cells through-wall (mates at
83–96°); the `*_screened` splits keep the quality-filtered subset
(`screen_vessel_corpus.py`). 414 train / 20 val is the right "little" scale.

Standing choices (reversible, recorded here):
- Renderings are gray placeholders + `image_drop_rate=1.0` = the paper's own
  "no reference image" mode (`crag_unified.py:685-687` zeroes dropped views;
  paper §3.3 sets `c_I` to zero). Real 12-view renders deferred to any future
  image-conditioned work — the datamodule crashes on zero renderings
  (`torch.stack([])`), so placeholders are load-bearing, not cosmetic.
- `full_mesh` = concatenated piece meshes (TORA stores pieces only). Interior
  fracture faces get sampled as full-shape surface: a known bias, disclosed,
  acceptable for a pipeline checkpoint.
- `up_axis: Z` (BreakingBad convention). Revisit if probe renders look rotated.

Related: `intent/CR1-does-generation-help-the-juglet.md` (precondition).

## Result — pending
