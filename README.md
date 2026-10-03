# DA-AL: Drift-Aware Active Retraining for Phishing-URL Detection (CSE339 project code)

Everything the paper claims is measured by this code. **No result in the paper is real until you run it on your own collected data.**

## What the code does

| File                   | Purpose                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `collect_phishtank.py` | Daily snapshot of the PhishTank feed -> `data/phishtank_raw.csv` (never visits URLs) |
| `build_dataset.py`     | Merges phishing + Tranco benign hostnames -> `data/dataset.csv`                      |
| `features.py`          | Lexical features only (host set `host`, host+path set `full`)                        |
| `experiment.py`        | Evaluation ladder P1/P2/P3 and streaming methods M1-M6 (paper Table II)              |
| `analyze_results.py`   | Tables, tests, figures, hypothesis verdicts -> `results/summary_main.md`             |
| `synth_data.py`        | **Synthetic** stream, only to check the code runs. Never report its numbers          |

## 1. Setup

```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. A laptop is enough; the main run is a few thousand model fits on <100k rows.

## 2. Collect data (START NOW: this is the slow part)

1. Read the current terms: https://phishtank.org/developer_info.php. The collector downloads the documented CSV feed (`online-valid.csv`). An application key is optional; PhishTank limits downloads without one.
2. If you have an application key, set `PHISHTANK_APPKEY`; optionally set `PHISHTANK_USER` for the User-Agent, then run `python collect_phishtank.py` **once per day** (cron / Windows Task Scheduler). The collector appends new rows to `data/phishtank_raw.csv`.
3. Download a Tranco list (https://tranco-list.eu, `top-1m.csv`, columns rank,domain) to `data/tranco_top1m.csv` and **write down the list ID** for the paper.
4. When you have enough days: `python build_dataset.py --window-days 7` (or `--window-days 1` for short collection).

**Time reality check.** The public feed only lists phish that are online _right now_, so you cannot back-fill weeks of history. Eight weekly windows need ~8-10 weeks. If your deadline is shorter, use `--window-days 1` and >= 3-4 weeks (>= 20 windows) and say so in the paper, or find a _timestamped_ public archive of phishing URLs and verify its terms and timestamps yourself. Never use a dataset without real timestamps for the temporal claims.

## 3. Check the code works (synthetic data)

```
python synth_data.py --weeks 12 --out data/synthetic_dataset.csv
python experiment.py --data data/synthetic_dataset.csv --out results_smoke --seeds 2 --betas 0.05 --trees 60
python analyze_results.py --res results_smoke --data "" --synthetic
```

The output is stamped SYNTHETIC. It only proves the pipeline runs.

## 4. Real run

`./run_all.sh` (or the commands inside). Defaults are the fixed settings in paper Table III; do not tune them on test windows.

## 5. Where each yellow cell in the paper comes from

| Paper placeholder                 | Source                                                                                  |
| --------------------------------- | --------------------------------------------------------------------------------------- |
| Names, university, emails         | you                                                                                     |
| Table I (dataset summary)         | top of `results/summary_main.md`                                                        |
| Table IV (protocols), Δopt, H1    | `summary_main.md` -> "Table IV"                                                         |
| Fig. 3, H2 (decay, slope, CI)     | `results/fig3_f1_over_time_main.png`, "H2" block                                        |
| Table V, Fig. 4, H3, H4, p-values | "Table V", "Paired Wilcoxon", "Hypothesis verdicts", `fig4_recovery_vs_budget_main.png` |
| Sensitivity paragraph (VI-D)      | `summary_delay1/2.md`, `summary_noise3.md`, `summary_full.md`                           |
| Abstract and conclusion numbers   | copy from the same files _last_                                                         |
| Hardware                          | your laptop (CPU, RAM, OS)                                                              |
| OpenPhish "confirm terms"         | delete the sentence if you do not use it                                                |

## 6. Rules that keep the study honest

- If `summary_main.md` says the drift gap F1(M2)-F1(M1) is not clearly positive, there is no drift to recover. Report that finding; do not report R.
- Report hypotheses that fail. A clearly explained negative result scores better than a hidden one.
- Do not change settings after seeing test results. If you must, report it as a new, labelled experiment.
- Δopt compares models trained on different amounts of data (P1/P2 use ~80%); state this in the discussion.
- Check the alarm timeline: an alarm in the first test window is often an ADWIN warm-up artifact. Report false alarms.
- Tranco gives hostnames only; the headline results use host-level features for this reason (see paper Section V-A).
