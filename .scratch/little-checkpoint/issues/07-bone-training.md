# 07 — Bone training run (features are informative — learn?)

Status: ready-for-agent · Answers: none (routine; serves CR1 without settling it)

Blocked by: 01 (converter). Follows 06 (bone AUC 0.627 — encoder sees
thick chunks, so training has something to work with).

## Spec

`scripts/hpc/crag_train_bone.slurm`: Stage-1 w/o-img on
`data/bone_crag.hdf5` (265 train / 82 val), batch 2, dummy-scale points,
`experiment_name=bone`, 40k steps in 5 × 8k chained segments, single A100,
`val_check_interval=1000`, csv.

## Done when (updated with numbers, 2026-10-02)

Chain `32099155→…→32099159`: all COMPLETED, 40k steps. Val part_acc
0.243→0.243→0.245→0.237→0.244; rmse_r 62–66°; train loss 0.33→0.64
(noisy, batch 2). Chance baseline at mean 6.8 parts ≈ 0.15: val sits
~1.6× chance, plateaued — same ratio as vessels (9% vs 5.6%), higher
absolute. Verdict: WEAK-POSITIVE, not flat-at-chance. The stack learns
coarse poses on thick fractures and stalls; scheduler-resume checked
(LambdaLR state restores last_epoch, per-segment max_decay extends the
cosine continuously — no freeze artifact). Bone ckpt now probing ceramics
+ Juglet (jobs 32112553/54).
