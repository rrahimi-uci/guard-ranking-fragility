# Overnight autonomous run — STATUS

**Date:** 2026-07-17. **State:** 🔧 recovering from an eval bug (no data lost); autonomous.

## What happened
The 10-VM adaptation fleet **trained all cells correctly** (LoRA adapters are safe in GCS,
`gs://jazztest-bucket/sta/results/adapters_<key>/`, 10 per checkpoint), but a bug in the VM startup
made the **eval** step score only the unmodified base cell:

- `sta_startup.sh` trained into `/root/staout/runs` but the eval call omitted `--out-root`, so it
  looked in the empty default dir → all 10 trained cells `missing_adapter`, parquets had 3308 rows
  (1 cell) instead of the full 11-cell grid. shieldgemma additionally OOM'd its base cell (gemma's
  256k vocab × batch 32 → 16 GB logits tensor on a 40 GB A100).

**Fixed** (commit 7607b72): eval now passes `--out-root`, batch 8 (OOM-safe), and a REEVAL mode pulls
the trained adapters from GCS and re-scores — **no retraining needed**.

## Recovery plan (autonomous)
1. Let the 5 still-training VMs finish (their adapters upload to GCS on exit).
2. Launch a **re-eval fleet** (`REEVAL=1 bash experiments/deploy/sta_launch.sh`) — `-re` VMs that
   download adapters + re-score correctly → corrected `sta_scores_<key>.parquet` (full 11-cell grid).
3. Collect + save in-repo → analyze → write adaptation section + append guidelines row → rebuild PDF
   → reviewer critique (`report.md`) + fixes → teardown ALL VMs → merge to `main`.

## Done + committed
- Phase-0, pipeline, guard_research/hashing fix, deploy scripts.
- KL-SFT 4/4 (data in-repo, Act I filled: KL recovers transfer +0.061 vs SFT).
- Paper guidelines summary table (Table 16, styled).

## Adaptation study status
- Training: 5/10 finished, 5 running (granite, smollm3, qwen3-4b, qwen3guard-4b, wildguard-7b).
- Adapters in GCS: for all finished checkpoints (recoverable).
- Corrected scores: pending re-eval fleet.
