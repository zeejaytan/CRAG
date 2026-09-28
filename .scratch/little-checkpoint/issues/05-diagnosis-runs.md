# 05 — Diagnosis runs: density vs canonicalization

Status: done · Answers: none (routine; serves CR1 without settling it)

Follows 04 (flat val → stop scaling, diagnose). Two short single-GPU runs
(~4k steps each) on the full vessel split, everything else identical:

- `fullscale`: paper-scale points (`num_points_per_object=20000`,
  `num_points_per_part=2048`, `min_points_per_part=512`), batch 2.
  Decides the sampling-density suspect (TORA O1). An OOM here is also a
  finding: it prices memory engineering.
- `nocanon`: dummy-scale points + `data.canonicalize=false`. Decides the
  PCA-frame suspect for thin sherds.

Done when: val part_acc moves in either → that suspect wins and earns
follow-up; both flat → volume/structural reassessment (upstream watch,
held allocation, or close the ladder).

## Result (2026-09-28) — both suspects spared, ladder reassessed

- `fullscale` (31444820): val 0.0925, rmse 79.2° — density changes nothing.
- `nocanon` (31451182): val 0.0899, rmse 77.8° — canonicalization changes
  nothing. (Two submit slips on the way: missing `+` prefix, stray quote;
  both caught, `bash -n` added to the habit.)
- Both flat at ~9% ≈ chance. Per the ticket rule this is reassessment, not
  scaling: remaining candidates are data volume (1108 vs 15k objects),
  LR-at-scale, and — sharpest — the paper's own admitted failure mode
  (TSDF VAE underrepresents slender thin-shell components), which would make
  part features uninformative no matter the steps. Next: probe VAE feature
  quality directly (cheap), or wait on upstream. No further training
  without a positive signal.
