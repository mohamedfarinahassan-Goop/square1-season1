# NBA home margin (nba-margin)

Predict home points minus away points for the 2012-13 to 2014-15 NBA seasons from pre-game FiveThirtyEight Elo data. Metric: MAE.

**Final model: `v4.py` → `submissions/v4_round_m02.csv`.** It is a 5-seed L1 LightGBM on pre-game features only, trained on all NBA seasons up to 2011-12. Predictions are `round(p − 0.2)`. Holdout MAE on 2010–2012 is 9.2226.

## Run

Put `train.csv`, `test.csv` and `sample_submission.csv` in this folder, then:

```bash
bash run.sh
```

It takes under 10 seconds. `v4.py` reuses the feature code in `v2.py` and the 5 early-stopping iteration counts from `exp4.py` (train ≤ 2009, validate 2010–2012). Running `python3 exp4.py` reproduces those counts and the rounding/shift study. Seeds 0–4 are fixed. A fresh run of `v4.py` reproduced the uploaded file exactly (3,944/3,944 identical rows).

## Files

| File | What it does | Holdout MAE |
|---|---|---|
| `v1.py` | Elo diff, forecast logit, rest, playoffs, season day | 9.235 |
| `v2.py` | + pre-game Elo dynamics (previous-game change, 5/10-game form), linear blend test | 9.234 |
| `v3.py` | team IDs, rest×Elo, quantile regression blend (no gain) | 9.241 |
| `exp4.py` | 5 seeds, integer rounding, shift study | raw 9.2327 |
| `v4.py` | final: 5 seeds, round(p − 0.2) | 9.2226 |

`python3 v2.py --future` shows a leak: the team's *next* pre-game Elo encodes this game's result (holdout MAE 0.347). It is post-game information, so it is **not** used in any submitted model.
