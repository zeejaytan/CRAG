# 06 — Probe VAE part-feature quality (no training)

Status: ready-for-agent · Answers: none (routine; serves CR1 without settling it)

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
