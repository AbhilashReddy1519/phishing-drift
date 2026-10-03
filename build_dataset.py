"""Merge PhishTank snapshots + Tranco hostnames into data/dataset.csv (url,label,ts,domain,target,delay_hours).
Tranco: download the list yourself from https://tranco-list.eu (top-1m.csv: rank,domain) and note the list ID in your paper."""
import argparse, numpy as np, pandas as pd
from features import registered_domain

ap = argparse.ArgumentParser()
ap.add_argument("--phish", default="data/phishtank_raw.csv")
ap.add_argument("--tranco", default="data/tranco_top1m.csv")
ap.add_argument("--ratio", type=float, default=5.0, help="benign per phishing URL (main experiment 5)")
ap.add_argument("--top-n", type=int, default=100000, help="sample benign hosts from the top-N ranks")
ap.add_argument("--window-days", type=int, default=7)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/dataset.csv")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

ph = pd.read_csv(a.phish)
submitted = pd.to_datetime(ph["submission_time"], utc=True, errors="coerce").dt.tz_localize(None)
ph["ts"] = pd.to_datetime(ph["first_seen_utc"], utc=True, errors="coerce").dt.tz_localize(None)
ver = pd.to_datetime(ph["verification_time"], utc=True, errors="coerce").dt.tz_localize(None)
ph["delay_hours"] = (ver - submitted).dt.total_seconds() / 3600
ph = ph.dropna(subset=["ts", "url"]).drop_duplicates("url")
ph["label"] = 1
ph["domain"] = ph["url"].map(registered_domain)

tr = pd.read_csv(a.tranco, header=None, names=["rank", "host"]).head(a.top_n)
n_b = int(len(ph) * a.ratio)
if n_b > len(tr):
    raise SystemExit(f"need {n_b} benign hosts but only {len(tr)} available; raise --top-n or lower --ratio")
# benign hosts get timestamps with the same weekly volume profile as phishing (so class ratio is stable per window)
b = tr.sample(n=n_b, random_state=a.seed).reset_index(drop=True)
src = ph["ts"].sample(n=n_b, replace=True, random_state=a.seed).reset_index(drop=True)
jitter = pd.to_timedelta(rng.uniform(-0.5, 0.5, n_b), unit="D")
be = pd.DataFrame({"url": "http://" + b["host"] + "/", "label": 0, "ts": src + jitter, "target": ""})
be["domain"] = be["url"].map(registered_domain); be["delay_hours"] = np.nan

cols = ["url", "label", "ts", "domain", "target", "delay_hours"]
ds = pd.concat([ph[cols], be[cols]]).sort_values("ts", kind="stable").reset_index(drop=True)
ds.to_csv(a.out, index=False)
print(f"wrote {a.out}: {len(ds)} rows, {int(ds.label.sum())} phishing, {ds.ts.min()} -> {ds.ts.max()}")
