"""Core experiments: evaluation ladder (P1-P3) and the streaming methods M1-M6 (see paper Table II)."""
import argparse, math, time, json, os
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from river.drift import ADWIN
from sklearn.metrics import f1_score, average_precision_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from features import feature_matrix

CFG = dict(window_days=7, init_windows=2, trees=150, depth=6, lr=0.1, delta=0.002, alpha=0.005, gamma=0.10,
           pool_factor=5, half_life=4.0, delay=0, min_window=200)

def load(path, mode, cfg):
    df = pd.read_csv(path, dtype={"target": "string"})
    df["ts"] = pd.to_datetime(df["ts"], format="mixed", utc=True)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    w = ((df.ts - df.ts.min()).dt.total_seconds() // (86400 * cfg["window_days"])).astype(int).values
    sizes = np.bincount(w); keep = sizes >= cfg["min_window"]          # drop sparse leading/trailing (partial) windows
    first, last = np.argmax(keep), len(keep) - 1 - np.argmax(keep[::-1])
    m = (w >= first) & (w <= last); df, w = df[m].reset_index(drop=True), w[m] - first
    X = feature_matrix(df.url, mode).values
    return df, X, df.label.values.astype(int), w

def fit(X, y, win, idx, t_cur, seed, cfg):
    yt = y[idx]; pos = max(1, yt.sum()); neg = max(1, len(yt) - pos)
    wts = 0.5 ** ((t_cur - win[idx]) / cfg["half_life"])
    m = XGBClassifier(n_estimators=cfg["trees"], max_depth=cfg["depth"], learning_rate=cfg["lr"], tree_method="hist",
                      scale_pos_weight=neg / pos, n_jobs=-1, random_state=seed, eval_metric="logloss", verbosity=0)
    m.fit(X[idx], yt, sample_weight=wts); return m

def metrics(y, p, thr=0.5):
    if y.sum() == 0 or y.sum() == len(y): return (np.nan, np.nan, np.nan)
    fpr, tpr, _ = roc_curve(y, p)
    return (f1_score(y, p >= thr, zero_division=0), average_precision_score(y, p), float(np.interp(0.01, fpr, tpr)))

def kcenter(Z, s):
    sel = [0]; d = np.linalg.norm(Z - Z[0], axis=1)
    while len(sel) < min(s, len(Z)):
        j = int(np.argmax(d)); sel.append(j); d = np.minimum(d, np.linalg.norm(Z - Z[j], axis=1))
    return np.array(sel)

def select_active(cand, u_cand, s, X, mu, sd, pool_factor, rng):
    if s <= 0 or len(cand) == 0: return np.array([], dtype=int)
    s = min(s, len(cand)); M = min(len(cand), pool_factor * s)
    order = np.argsort(-(u_cand + rng.uniform(0, 1e-9, len(u_cand))))[:M]
    pool = cand[order]; sel = kcenter((X[pool] - mu) / sd, s)
    return pool[sel]

def run_stream(X, y, win, method, beta, seed, cfg, noise=0.0):
    rng = np.random.default_rng(seed); T = int(win.max()) + 1; init = cfg["init_windows"]; d = cfg["delay"]
    by_w = [np.where(win == t)[0] for t in range(T)]
    init_idx = np.concatenate(by_w[:init]); mu = X[init_idx].mean(0); sd = X[init_idx].std(0) + 1e-6
    y_obs = y.copy()
    if noise > 0:  # label noise only affects what the learner sees, never evaluation
        flip = rng.random(len(y)) < noise; y_obs[flip] = 1 - y_obs[flip]
    det = method in ("M5", "M6"); adwin = None; clock = 0.0; n_retrain = 0
    if det:   # warm up ADWIN on a held-out initial window so the first test window does not alarm spuriously
        m0 = fit(X, y_obs, win, np.concatenate(by_w[:init - 1]), init - 2, seed, cfg)
        p0 = m0.predict_proba(X[by_w[init - 1]])[:, 1]; adwin = ADWIN(delta=cfg["delta"])
        for u in 1 - np.abs(2 * p0 - 1): adwin.update(float(u))
    model = fit(X, y_obs, win, init_idx, init - 1, seed, cfg)
    labeled = init_idx; pending = []; Z = 0.0; used = 0; recs = []
    for t in range(init, T):
        idx = by_w[t]; n = len(idx)
        p = model.predict_proba(X[idx])[:, 1]; f1, ap, tpr1 = metrics(y[idx], p)
        u = 1 - np.abs(2 * p - 1); alarm = False; q = np.array([], dtype=int)
        t0 = time.perf_counter()
        if method == "M2":
            q = idx
        elif method in ("M3", "M4"):
            Z += beta * n; s = int(Z); Z -= s
            q = rng.choice(idx, min(s, n), replace=False) if method == "M3" else select_active(idx, u, s, X, mu, sd, cfg["pool_factor"], rng)
        elif det:
            for ui in u:
                adwin.update(float(ui)); alarm = alarm or adwin.drift_detected
            Z += beta * n; a_n = min(int(Z), math.ceil(cfg["alpha"] * n)); a_pos = rng.choice(n, a_n, replace=False) if a_n > 0 else np.array([], dtype=int)
            Z -= a_n; q = idx[a_pos]
            if alarm:
                s = min(int(Z), math.ceil(cfg["gamma"] * n)); rest = np.setdiff1d(np.arange(n), a_pos)
                if s > 0 and len(rest):
                    extra = (rng.choice(rest, min(s, len(rest)), replace=False) if method == "M5" else
                             np.searchsorted(idx, select_active(idx[rest], u[rest], s, X, mu, sd, cfg["pool_factor"], rng)))
                    Z -= len(extra); q = np.concatenate([q, idx[extra]])
        used += len(q); trig = (not det) or alarm      # detector methods retrain only when an alarm-triggered batch arrives
        pending.append((t + d, q, trig))
        arrive = [x for x in pending if x[0] <= t]; pending = [x for x in pending if x[0] > t]
        if arrive:
            labeled = np.concatenate([labeled] + [x[1] for x in arrive])
            if any(x[2] for x in arrive) and sum(len(x[1]) for x in arrive) > 0:
                model = fit(X, y_obs, win, labeled, t, seed, cfg); n_retrain += 1
        clock += time.perf_counter() - t0
        recs.append(dict(method=method, beta=beta, seed=seed, t=t, n=n, f1=f1, ap=ap, tpr1=tpr1, labels_cum=used, alarm=int(alarm), retrains=n_retrain, time_cum=clock))
    return recs

def protocols(X, y, win, groups, cfg, seeds):
    rows = []
    def one(name, trn, tst, seed, size):
        m = fit(X, y, win, trn, win[trn].max(), seed, dict(cfg, half_life=1e9)); p = m.predict_proba(X[tst])[:, 1]
        rows.append(dict(protocol=name, seed=seed, train_size=size, **dict(zip(["f1", "ap", "tpr1"], metrics(y[tst], p)))))
    for s in range(seeds):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y): one("P1_random", tr, te, s, len(tr))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, groups): one("P2_group", tr, te, s, len(tr))
        T = int(win.max()) + 1
        for t in range(cfg["init_windows"], T):
            tst = np.where(win == t)[0]
            one("P3_expanding", np.where(win < t)[0], tst, s, int((win < t).sum()))
            one("P3_static", np.where(win < cfg["init_windows"])[0], tst, s, int((win < cfg["init_windows"]).sum()))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv"); ap.add_argument("--out", default="results")
    ap.add_argument("--features", default="host", choices=["host", "full"]); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--betas", default="0.01,0.02,0.05,0.10"); ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--window-days", type=int, default=7); ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--noise", type=float, default=0.0, help="label-noise rate for the robustness run")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    cfg = dict(CFG, window_days=a.window_days, trees=a.trees, delay=a.delay)
    os.makedirs(a.out, exist_ok=True)
    df, X, y, win = load(a.data, a.features, cfg)
    T = int(win.max()) + 1; print(f"{len(df)} URLs | {T} windows | phishing share {y.mean():.3f} | features={a.features}")
    assert T > cfg["init_windows"] + 2, "need at least init_windows+3 windows; collect longer or use --window-days 1"
    groups = pd.factorize(df.domain)[0]; betas = [float(b) for b in a.betas.split(",")]
    json.dump(dict(cfg=cfg, args=vars(a), n=len(df), windows=T), open(f"{a.out}/config_{a.tag}.json", "w"), indent=1)
    if a.tag == "main":
        protocols(X, y, win, groups, cfg, min(a.seeds, 5)).to_csv(f"{a.out}/protocols.csv", index=False); print("protocols done")
    recs = []
    for s in range(a.seeds):
        for m in ("M1", "M2"): recs += run_stream(X, y, win, m, 0.0, s, cfg, a.noise)
        for b in betas:
            for m in ("M3", "M4", "M5", "M6"): recs += run_stream(X, y, win, m, b, s, cfg, a.noise)
        print(f"seed {s + 1}/{a.seeds} done", flush=True)
        pd.DataFrame(recs).to_csv(f"{a.out}/windows_{a.tag}.csv", index=False)
