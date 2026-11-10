#!/usr/bin/env bash
# One-command run: regenerates submissions/v7.csv (final) and submissions/v6.csv from train.csv/test.csv.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p submissions
python3 feats.py      # v4 feature table -> tr_f.pkl / te_f.pkl
python3 v5feats.py    # adds interpolated-hour physics features -> tr_f5.pkl / te_f5.pkl
python3 exp5c.py      # holdout screen of 7 LightGBM variants x 3 seeds -> exp5c.pkl (CV + best iterations)
python3 exp5d.py      # 4 more variants -> exp5d.pkl
python3 v6.py         # full-train refit of 5 variants -> submissions/v6.csv
python3 v7.py         # 4 more refits, 9-variant mean -> submissions/v7.csv
