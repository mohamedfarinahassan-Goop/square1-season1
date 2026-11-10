# Estimate the sun (solar-britain)

Predict GB half-hourly solar generation (MW) from NASA POWER weather at 12 points plus installed capacity. Metric: MAE.

**Final model: `v7.py` → `submissions/v7.csv`.** It is the equal mean of 9 LightGBM variants (3 seeds each) trained on capacity factor with an L1 loss and capacity weights. Holdout MAE is 115.08 MW.

## Run

Put `train.csv`, `test.csv` and `sample_submission.csv` in this folder, then:

```bash
bash run.sh
```

The steps are `feats.py → v5feats.py → exp5c.py → exp5d.py → v6.py → v7.py`. The `exp5*` scripts fit on data before 2024-07-01, score July 2024 to June 2025, and store each variant's best iteration. `v6.py` and `v7.py` refit on all of train for 1.1 × those iterations. Seeds are fixed (0, 1, 2 per variant). It takes several tens of minutes on 8 CPU cores.

## Files

| File | What it does | Holdout MAE |
|---|---|---|
| `v1.py` | weather + calendar, capacity-factor target | 140.65 |
| `v2.py` | sun position, ±1/±2 half-hour lags/leads, regional means, 3 seeds | 122.98 |
| `v3.py` | + sun elevation at ±30 min, leaf tuning (not uploaded) | 122.38 |
| `v4.py` | physical CF proxies, capacity-weighted regional irradiance | 118.70 |
| `feats.py` | v4 feature table, cached | |
| `v5feats.py` | per-city interpolated clearness × Haurwitz clear-sky at each city's sun position | |
| `exp5.py` | shared holdout helper | |
| `exp5b.py` | bias, recency-weighting and drop-capacity checks | |
| `exp5c.py`, `exp5d.py` | 11-variant screen | see `logs/` |
| `v6.py` | 5-variant mean | 115.55 |
| `v7.py` | 9-variant mean (final) | 115.08 |

The city capacity weights in `feats.py` are a fixed modelling assumption, not fitted to any target. No external data is used.
