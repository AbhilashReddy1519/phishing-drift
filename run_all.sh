#!/usr/bin/env bash
# Full pipeline after data collection; falls back to an explicitly synthetic smoke test if real data is too short.
set -e
if [[ -z "${PYTHON:-}" ]]; then
	if [[ -f .venv/Scripts/python.exe ]]; then
		PYTHON="./.venv/Scripts/python.exe"
	else
		PYTHON="python"
	fi
fi

if [[ ! -f data/dataset.csv ]] || ! "$PYTHON" -c '
import sys
import numpy as np
import pandas as pd

df = pd.read_csv("data/dataset.csv", usecols=["ts"])
ts = pd.to_datetime(df["ts"], format="mixed", utc=True)
w = ((ts - ts.min()).dt.total_seconds() // (7 * 86400)).astype(int).to_numpy()
sizes = np.bincount(w)
keep = sizes >= 200
ready = keep.any() and np.flatnonzero(keep)[-1] - np.flatnonzero(keep)[0] + 1 >= 5
sys.exit(0 if ready else 1)
'; then
	echo "Real data is missing or has fewer than five usable weekly windows."
	echo "Running a synthetic smoke test only; these results are not research results."
	"$PYTHON" synth_data.py --weeks 12 --out data/run_all_smoke_dataset.csv
	"$PYTHON" experiment.py --data data/run_all_smoke_dataset.csv --out results_run_all_smoke --seeds 2 --betas 0.05 --trees 60
	"$PYTHON" analyze_results.py --res results_run_all_smoke --data "" --synthetic
	exit 0
fi

"$PYTHON" experiment.py --data data/dataset.csv --out results --tag main --seeds 10                 # main experiment + protocols
"$PYTHON" analyze_results.py --res results --tag main --beta 0.05 --data data/dataset.csv          # -> results/summary_main.md + figures
# robustness / sensitivity (fewer seeds is fine)
"$PYTHON" experiment.py --data data/dataset.csv --out results --tag delay1 --delay 1 --seeds 5 --betas 0.05
"$PYTHON" experiment.py --data data/dataset.csv --out results --tag delay2 --delay 2 --seeds 5 --betas 0.05
"$PYTHON" experiment.py --data data/dataset.csv --out results --tag noise3 --noise 0.03 --seeds 5 --betas 0.05
"$PYTHON" experiment.py --data data/dataset.csv --out results --tag full --features full --seeds 5 --betas 0.05   # only if benign URLs have real paths
for t in delay1 delay2 noise3 full; do "$PYTHON" analyze_results.py --res results --tag "$t" --beta 0.05 --data data/dataset.csv; done
