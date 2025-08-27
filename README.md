# Boston Housing Predictor

A solution on handling temporal drift under a strict chronological split.

## Executive Summary
- Outcome: Positive generalization under the mandatory split; best test RMSE 3.75, R² 0.522 (SVR with train-only calibration).
- Approach: Stability-first features, train-quantile winsorization, monotonic transforms, ratio/difference features, train-only scaling, time-aware CV tuning, simple linear calibration.
- Alignment: Strictly train on the first 70% and test on the last 30%; compare against baselines; analyze split performance differences; propose and implement regression-only robustness strategies.

## Overview
- Constraint: Train on first 70% (older houses), test on last 30% (newer houses) — non‑negotiable.
- Goal: Achieve positive R² on the constrained test split.

## Current Best Results (Phase 2)
- Model: SVR (with Phase 2 features + train‑only calibration)
- Test metrics: RMSE 3.75, R² 0.522
- Baselines for reference:
  - Prior best (pre‑Phase 2): GB RMSE 6.27, R² -0.332
  - Original (all features, standard): RMSE 12.52, R² -4.311

## Quick Start
- Prerequisites: Python 3.8+, `uv`
```bash
uv sync
uv shell
python create_production_notebook.py
jupyter notebook boston-housing-predictor.ipynb
```

## What the Notebook Does
- Stability analysis → select 8 stable features
- Phase 2 features: train‑quantile winsorization, monotonic transforms (log1p), ratio/difference features
- Train‑only scaling and target calibration
- Time‑aware CV tuning for SVR/GB on the train segment
- Strict 70/30 evaluation on the test segment

## Lessons Learned
- Feature explosion (13→43) worsened drift sensitivity — dropped.
- Domain adaptation (e.g., CORAL, quantile/z‑score matching) reduced MMD but harmed prediction — kept only as diagnostics, not used in prediction.
- Stability‑first features + robust transforms + simple calibration can flip R² positive under the strict split.

## Dataset
- Boston Housing (local copy at `data/housing.csv`), 490 samples after removing censored targets (MEDV ≥ 50.0).
