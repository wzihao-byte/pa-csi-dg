# Negative Control: Current SupCon Final Config

**Date:** 2026-06-13  
**Status:** NEGATIVE — did not reproduce old SupCon-only anchor

## Selected Config

| Parameter | Value |
|---|---|
| lambda_supcon | 0.05 |
| batch_size | 32 |
| sampler | domain_class_balanced |
| supcon_positive_mode | global_class_instance_views |
| include_same_domain_same_class | True |
| temperature | 0.2 |
| warmup_epochs | 0 |

This config was selected via sequential single-env (E2) + single-seed (42) sweeps:
lambda → batch/sampler → temperature → positive_mask → warmup.

## Results vs Old Anchors (seed42)

| Method | Mean Acc | Mean F1 | E1 | E2 | E3 |
|---|---|---|---|---|---|
| 旧 CE Baseline seed42 | 0.8787 | 0.8515 | 0.9007 | 0.8547 | 0.8807 |
| 旧 SupCon only seed42 | 0.9027 | 0.8820 | 0.9147 | 0.8953 | 0.8980 |
| **Current SupCon final seed42** | **0.8614** | **0.8392** | 0.8967 | 0.8580 | 0.8297 |
| Current SupCon final 9-run mean | 0.8643 | 0.8444 | 0.8949 | 0.8544 | 0.8436 |

## Key Deltas (seed42, vs 旧 SupCon only)

- Mean Acc: **-4.13 pp**
- Mean F1: **-4.28 pp**
- E3 is worst: **-6.83 pp** (0.8297 vs 0.8980)
- Multi-seed variance did NOT cause the gap — seed42 alone is equally bad

## Root Cause Hypothesis

The sweep selected on E2/seed42 only. Config diverges from paper_aligned on
multiple axes simultaneously: batch_size (8→32), sampler (none→domain_class_balanced),
lambda (0.1→0.05), positive_mode (default→global_class_instance_views).

The paper_aligned point was never tested in the new sweep space.

## Next Step

Phase 1: Run paper_aligned config as-is on 3env/seed42 to determine if the
old anchor is reproducible under current code. See scripts/run_anchor_recovery.sh
