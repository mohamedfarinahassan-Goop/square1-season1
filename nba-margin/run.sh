#!/usr/bin/env bash
# One-command run: regenerates submissions/v4_round_m02.csv (final) and v4_round.csv.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p submissions
python3 v4.py
