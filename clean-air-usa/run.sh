#!/usr/bin/env bash
# One-command run: regenerates submissions/v7.csv (final) and v7mean3.csv.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p submissions
python3 exp7.py   # holdout screen (train < 2024-07-01, valid 2024-07..2025-03) -> exp7.pkl
python3 v7.py     # full-train refit -> submissions/v7.csv
