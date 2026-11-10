# Emissions gap (emissions-gap)

Predict annual CO2e (tonnes) for unseen fossil power plants, 2021–2025, from fuel, country, capacity and location. Metric: RMSLE.

**Final model: `v3.py` → `submissions/v3.csv`.** It is the log1p-space mean of 4 LightGBM configs (v2 features, v2 + plant-level aggregates, 63 leaves/ff0.5, Huber), with 3 seeds each in the final fit. GroupKFold-by-plant CV RMSLE is 1.0060.

## Run

Put `train.csv`, `test.csv` and `sample_submission.csv` in this folder, then:

```bash
bash run.sh
```

`exp3.py` reuses the feature and encoding code from `v2.py`, runs 5-fold GroupKFold by plant for each config and stores the fold iterations. `v3.py` refits on all of train for 1.1 × the mean fold iteration count. Seeds are fixed. A fresh run reproduced the per-config CV scores in `logs/exp3.log` and the uploaded `v3.csv` exactly (12,560 rows, max difference 0.0).

## Files

| File | What it does | CV RMSLE |
|---|---|---|
| `v1.py` | fuel, country, capacity, location, fuel flags; log1p target | 1.043 |
| `v2.py` | + out-of-fold smoothed target encodings of log t/MW, nearest-plant intensity (BallTree), country context | 1.016 |
| `exp3.py` | plant-level feature-only aggregates, 4-config screen | 1.008–1.027 |
| `v3.py` | final 4-config log-space mean | 1.0060 |

Target encodings for training rows are computed out-of-fold (inner GroupKFold by plant), so a plant never sees its own target.
