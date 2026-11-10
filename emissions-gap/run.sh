#!/usr/bin/env bash
# One-command run: regenerates submissions/v3.csv (final).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p submissions
python3 exp3.py   # 5-fold GroupKFold-by-plant CV of 4 configs -> exp3.pkl
python3 v3.py     # full-train refit, log-space mean -> submissions/v3.csv
