# Square1 AI Season 1: solution code (MoFarina)

Code for my entries in the four Square1 AI Season 1 competitions. Each folder is self-contained and has its own README and a one-command `run.sh`.

| Folder | Competition | Metric | Final model |
|---|---|---|---|
| [`solar-britain/`](solar-britain/) | Estimate the sun (GB solar generation) | MAE, MW | `v7.py`: mean of 9 LightGBM variants |
| [`nba-margin/`](nba-margin/) | NBA home margin | MAE, points | `v4.py`: 5-seed LightGBM, rounded with a −0.2 shift |
| [`clean-air-usa/`](clean-air-usa/) | Daily PM2.5 at 250 EPA sites | MAE, µg/m³ | `v7.py`: L1 LightGBM on log1p(PM2.5) |
| [`emissions-gap/`](emissions-gap/) | Power-plant annual CO2e | RMSLE | `v3.py`: log-space mean of 4 LightGBM configs |

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13.5, LightGBM 4.7.0, pandas 3.0.6, numpy 2.5.4 and scikit-learn 1.9.1 on an 8-core Linux machine with no GPU.

## Data

The competition data is **not** included. Download `train.csv`, `test.csv` and `sample_submission.csv` from each competition page on square1ai.com (sign-in required) and put them directly in the matching folder, next to the scripts. Then run `bash run.sh` inside that folder. Predictions are written to `<folder>/submissions/`.

No external data is used. Scripts only read the three shipped files.

## Layout

Each folder keeps the full version history (`v1.py` … final) so the progression in the reports can be checked. The `exp*.py` scripts are the validation screens whose early-stopping iteration counts the final scripts reuse. The final fits train on all of train.csv for 1.1 × the best holdout iteration count. `logs/` holds the console output of the validation runs that the reports quote.

## License

MIT, see [LICENSE](LICENSE). The competition data has its own licences, listed on each competition page.
