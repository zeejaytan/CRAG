# 06 — Probe VAE part-feature quality (no training)

Status: done · Answers: none (routine; serves CR1 without settling it)

## Result (2026-10-02) — features carry no object identity

198 vessel parts → 1024-dim frozen-VAE features (0 dead channels — alive,
varying). Same-object retrieval AUC **0.512** (chance 0.5); thin half 0.528
vs bulky half 0.531 — no difference. So not the paper's thin-specific
failure mode: pooled features discriminate nothing at all.
Caveat (held honestly): the probe mean-pools tokens; the assembly
transformer attends over full token sequences, a richer signal — but val
flat at ~9% across six runs says it exploits nothing either.
Verdict: mush. Stage-1 as released cannot learn thin shells here; further
training spend needs VAE adaptation (Stage-2 surgery) or upstream release.

## Follow-up (2026-10-02) — thick bone chunks: PARTLY SEES

Same probe on `bone_crag.hdf5` (job 32091029): 82 parts, 0 dead channels,
same-object AUC **0.627** (thin half 0.717, bulky 0.573 — inverted vs the
paper's failure mode; small-n, don't over-read the split). Verdict:
the encoder discriminates thick fractures moderately — the blindness is
thin-shell-specific, not general. Reopens bone training as a working path.

## Why

04 flat + 05 both spared: density and canonicalization change nothing, yet
val sits at chance. Leading structural suspect, and the paper's own admitted
failure mode (§Representative Failures): the TSDF VAE underrepresents slender
thin-shell components — opposite surfaces merge — which would make assembly
features uninformative no matter the steps. Note the VAE is frozen
pretrained TripoSG; nothing we trained touches it, so probe it directly.

## Method

`scripts/probe_vae_features.py` (new, login node or gpu-short, CPU is fine —
model builds on CPU against `pretrained/TripoSG`): load ~200 vessel parts
via `CragDataset`, run the model's `encode_parts` path, collect features.
Tests, cheapest first:

1. Dead dims: fraction of feature channels with ~zero variance across parts.
2. Part identity k-NN: given a part's feature, retrieve its source object
   among K candidates by feature distance. Chance = 1/K. Near chance =
   features carry no part identity → explains flat val at any scale.
3. Thin-vs-bulky split: same test on high- vs low-thickness parts (thickness
   from part bbox smallest side). If bulky passes and thin fails, it's the
   paper's failure mode exactly.

## Done when

One number per test + verdict: features informative (then the flat val is a
training-recipe problem — LR, volume, schedule) or mush (then Stage-1 as
released cannot learn thin shells here; escalate to VAE adaptation = Stage-2
surgery, or upstream watch). No GPU training spend either way.
