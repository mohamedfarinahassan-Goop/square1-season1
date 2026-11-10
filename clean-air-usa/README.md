# Clean air USA (clean-air-usa)

Predict daily mean PM2.5 (µg/m³) at 250 EPA sites for April to December 2025 from daily NASA POWER weather and the site's training history. Metric: MAE.

**Final model: `v7.py` → `submissions/v7.csv`.** It is LightGBM with an L1 loss on log1p(PM2.5), using the v2 feature set, 127 leaves, feature fraction 0.5 and 3 seeds. Holdout MAE is 2.564.

## Run

Put `train.csv`, `test.csv` and `sample_submission.csv` in this folder, then:

```bash
bash run.sh
```

`exp7.py` builds the v2 features (from `v2.py`), fits on data before 2024-07-01, scores July 2024 to March 2025 and stores the best iterations. `v7.py` refits on all of train for 1.1 × those iterations. Seeds 0–2 are fixed. If `submissions/v125mean.csv` exists (the mean of the v1, v2 and v5 outputs), `v7.py` also writes a 50/50 blend. It is optional.

## Files

| File | What it does | Holdout MAE |
|---|---|---|
| `v1.py` | weather, calendar, site/division codes, fold-only target encodings, weather lags | 2.702 |
| `v2.py` | + next-day weather, 7-day rolling stats, dry streak, stagnation proxy, division/national weather, 3 seeds | 2.636 |
| `v3.py` | season-aligned holdout (worse, underfit) | 2.815* |
| `v4.py` | out-of-fold target encodings by year | 2.840* |
| `v5.py` | down-weights the 2023 wildfire-smoke window | 2.676* |
| `v6.py` | lr 0.015, 5 seeds (no gain) | 2.641 |
| `exp7.py`, `v7.py` | L1 on log1p(target), final | 2.564 |

\* scored on a different, season-aligned holdout, so not directly comparable.

Target encodings are always fitted on the training fold only. Weather lags and leads use weather, never the target.
