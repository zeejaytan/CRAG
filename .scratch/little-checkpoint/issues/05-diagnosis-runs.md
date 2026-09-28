# 05 — Diagnosis runs: density vs canonicalization

Status: ready-for-agent · Answers: none (routine; serves CR1 without settling it)

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
