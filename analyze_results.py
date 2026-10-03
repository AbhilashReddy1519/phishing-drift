"""Turns results/windows_main.csv + protocols.csv into the numbers/figures for the paper placeholders.
Writes results/summary.md (copy numbers from it into the yellow cells) and results/fig3_*.png, fig4_*.png."""
import argparse, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results"); ap.add_argument("--data", default="data/dataset.csv")
ap.add_argument("--beta", type=float, default=0.05); ap.add_argument("--tag", default="main")
ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--min-gap", type=float, default=0.02, help="min F1(M2)-F1(M1) for R to be reported")
a = ap.parse_args(); rng = np.random.default_rng(0)
W = pd.read_csv(f"{a.res}/windows_{a.tag}.csv"); P = pd.read_csv(f"{a.res}/protocols.csv") if a.tag == "main" else None
NAMES = {"M1": "M1 Static", "M2": "M2 Full retrain", "M3": "M3 Random labels", "M4": "M4 Active only", "M5": "M5 Detector only", "M6": "M6 DA-AL"}
out = []; w = out.append
if a.synthetic: w("# !!! SYNTHETIC SMOKE TEST — THESE NUMBERS ARE NOT RESULTS. DO NOT PUT THEM IN THE PAPER !!!\n")

def series(method, beta, col="f1"):
    d = W[(W.method == method) & (np.isclose(W.beta, beta if method not in ("M1", "M2") else 0.0))]
    return d.groupby("t")[col].mean()          # mean over seeds, indexed by window

def block_idx(n, L=2):
    starts = rng.integers(0, n, size=int(np.ceil(n / L))); return np.concatenate([(s + np.arange(L)) % n for s in starts])[:n]

def boot(fn, n, B=2000):
    vals = [fn(block_idx(n)) for _ in range(B)]; return np.nanpercentile(vals, [2.5, 97.5])

def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj

f1_s, f1_f = series("M1", 0), series("M2", 0); T = len(f1_s); ts = f1_s.index.values
gap = f1_f.mean() - f1_s.mean()

# ---- Table I
if a.data and not a.synthetic:
    D = pd.read_csv(a.data, parse_dates=["ts"]); ph = D[D.label == 1]
    w("## Table I — dataset summary")
    w(f"- period: {D.ts.min():%Y-%m-%d} → {D.ts.max():%Y-%m-%d} ({T + 2} windows after trimming)\n- phishing URLs: {len(ph)}\n- benign hostnames: {(D.label == 0).sum()}\n"
      f"- phish:benign = 1:{(D.label == 0).sum() / max(1, len(ph)):.1f}\n- unique registered domains (phish): {ph.domain.nunique()}\n- distinct targets: {ph.target.replace('', np.nan).nunique()}\n"
      f"- median label delay (h): {ph.delay_hours.median():.2f}\n")

# ---- Table IV: protocols (H1)
if P is not None:
    w("## Table IV — same model, three protocols (mean over seeds; ± = std over seeds)")
    g = P.groupby("protocol")[["f1", "ap", "tpr1", "train_size"]].mean(); sd = P.groupby("protocol")[["f1", "ap", "tpr1"]].std()
    for k in g.index: w(f"- {k}: F1={g.loc[k,'f1']:.3f}±{sd.loc[k,'f1']:.3f}  PR-AUC={g.loc[k,'ap']:.3f}  TPR@1%FPR={g.loc[k,'tpr1']:.3f}  (mean train size {g.loc[k,'train_size']:.0f})")
    d_opt = g.loc["P1_random", "f1"] - g.loc["P3_expanding", "f1"]
    w(f"- **Δopt = F1(P1) − F1(P3_expanding) = {d_opt:.3f}**; campaign-leakage part (P1−P2) = {g.loc['P1_random','f1'] - g.loc['P2_group','f1']:.3f}; "
      f"time part (P2−P3) = {g.loc['P2_group','f1'] - g.loc['P3_expanding','f1']:.3f}")
    w("- CAUTION: P1/P2 train on ~80% of data, P3 on earlier windows only — state this training-size difference when you discuss Δopt.")
    w(f"- H1 verdict: {'SUPPORTED (Δopt>0)' if d_opt > 0 else 'NOT SUPPORTED'}; test it formally by comparing per-seed F1 (e.g., paired Wilcoxon over seeds).\n")

# ---- H2: decay of static model
rho, pval = spearmanr(ts, f1_s.values)
slope = np.polyfit(ts, f1_s.values, 1)[0]; lo, hi = boot(lambda ix: np.polyfit(ts[ix], f1_s.values[ix], 1)[0] if len(set(ts[ix])) > 1 else np.nan, T)
w("## H2 — does the static model decay?")
w(f"- Spearman ρ(window, F1_static) = {rho:.3f} (p={pval:.3f}); slope = {slope:.4f} F1/window, 95% block-bootstrap CI [{lo:.4f}, {hi:.4f}]")
w(f"- H2 verdict: {'SUPPORTED (negative trend, CI excludes 0)' if hi < 0 else 'NOT SUPPORTED at 95% (CI includes 0 or positive)'}")
w(f"- drift gap = F1(M2) − F1(M1) = {gap:.3f}. {'If this is not clearly > 0 there is no drift to recover: do NOT report R.' if gap < a.min_gap else ''}\n")

# ---- Table V at beta
def Rfun(m, beta, ix=None):
    s = series(m, beta).values; ix = np.arange(T) if ix is None else ix
    return (np.nanmean(s[ix]) - np.nanmean(f1_s.values[ix])) / (np.nanmean(f1_f.values[ix]) - np.nanmean(f1_s.values[ix]))
tot_eval = W[(W.method == "M2") & (W.seed == W.seed.min())].n.sum()
w(f"## Table V — methods at β={a.beta:.0%}")
w("| Method | F1 | PR-AUC | R (%) [95% CI] | Labels (% of eval stream) | Retrains | Time (s) |\n|---|---|---|---|---|---|---|")
rows = {}
for m in NAMES:
    b = 0.0 if m in ("M1", "M2") else a.beta; d = W[(W.method == m) & np.isclose(W.beta, b)]
    last = d[d.t == d.t.max()]; lab = last.labels_cum.mean() / tot_eval * 100; rt = last.retrains.mean(); tm = last.time_cum.mean()
    r = "0" if m == "M1" else "100" if m == "M2" else "—"
    if m not in ("M1", "M2") and gap >= a.min_gap:
        lo, hi = boot(lambda ix: Rfun(m, a.beta, ix), T); r = f"{100 * Rfun(m, a.beta):.0f} [{100 * lo:.0f}, {100 * hi:.0f}]"
    rows[m] = r; w(f"| {NAMES[m]} | {series(m, b).mean():.3f} | {series(m, b, 'ap').mean():.3f} | {r} | {lab:.1f} | {rt:.1f} | {tm:.1f} |")
w("")
# ---- Tests
w("## Paired Wilcoxon (across windows, seed-averaged F1), Holm-corrected")
base = series("M6", a.beta).values; names, ps, effs = [], [], []
for m in ("M1", "M3", "M4", "M5"):
    other = series(m, 0 if m == "M1" else a.beta).values; diff = base - other
    try: p = wilcoxon(base, other).pvalue if np.any(diff != 0) else 1.0
    except ValueError: p = 1.0
    names.append(m); ps.append(p); effs.append(diff.mean())
adj = holm(np.array(ps))
for m, p, pa, e in zip(names, ps, adj, effs): w(f"- M6 vs {NAMES[m]}: mean ΔF1 = {e:+.3f}, p = {p:.4f}, Holm-adj p = {pa:.4f}")
w(f"- NOTE: with only {T} windows the Wilcoxon test has a minimum attainable p-value; say so in the paper (limited power).\n")
if gap >= a.min_gap:
    R6 = Rfun("M6", a.beta); lab6 = W[(W.method == "M6") & np.isclose(W.beta, a.beta) & (W.t == W.t.max())].labels_cum.mean() / tot_eval * 100
    beats_rand = series("M6", a.beta).mean() > series("M3", a.beta).mean()
    w("## Hypothesis verdicts (pre-specified)")
    w(f"- H3: R(DA-AL) = {100*R6:.0f}% at {lab6:.1f}% labels → {'SUPPORTED' if (R6 >= 0.8 and beats_rand) else 'NOT SUPPORTED'} (needs R≥80% AND better than M3)")
    f = lambda m: series(m, a.beta).mean()
    w(f"- H4: F1 M6={f('M6'):.3f} vs M4={f('M4'):.3f}, M5={f('M5'):.3f} → {'SUPPORTED' if f('M6') > max(f('M4'), f('M5')) else 'NOT SUPPORTED'}  (also check the significance rows above)\n")

# ---- Fig 3 / Fig 4
fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=300)
for m, c, ls in [("M1", "#C2410C", "-"), ("M2", "#10222B", "--"), ("M6", "#0F766E", "-")]:
    s = series(m, 0 if m in ("M1", "M2") else a.beta); ax.plot(s.index, s.values, ls, color=c, lw=1.2, label=NAMES[m] + (f" (β={a.beta:.0%})" if m == "M6" else ""))
al = W[(W.method == "M6") & np.isclose(W.beta, a.beta)].groupby("t").alarm.mean()
for t_, v in al.items():
    if v > 0.5: ax.axvline(t_, color="#F59E0B", alpha=0.35, lw=5)
ax.set_xlabel("window index t", fontsize=7); ax.set_ylabel("F1", fontsize=7); ax.tick_params(labelsize=6); ax.legend(fontsize=5.5, frameon=False); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(f"{a.res}/fig3_f1_over_time_{a.tag}.png"); plt.close(fig)
betas = sorted(W[~W.method.isin(["M1", "M2"])].beta.unique())
fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=300)
if gap >= a.min_gap:
    for m, c in [("M3", "#9CA3AF"), ("M4", "#F59E0B"), ("M5", "#C2410C"), ("M6", "#0F766E")]:
        ys = [100 * Rfun(m, b) for b in betas]; ax.plot([100 * b for b in betas], ys, "o-", ms=3, lw=1.1, color=c, label=NAMES[m])
    ax.axhline(80, color="k", ls=":", lw=0.8); ax.set_xlabel("label budget β (%)", fontsize=7); ax.set_ylabel("recovery R (%)", fontsize=7)
    ax.tick_params(labelsize=6); ax.legend(fontsize=5.5, frameon=False); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(f"{a.res}/fig4_recovery_vs_budget_{a.tag}.png"); plt.close(fig)
w("## Figures\n- fig3_f1_over_time_<tag>.png → paper Fig. 3 (amber bands = windows where DA-AL alarmed)\n- fig4_recovery_vs_budget_<tag>.png → paper Fig. 4")
summary = "\n".join(out)
with open(f"{a.res}/summary_{a.tag}.md", "w", encoding="utf-8") as f:
    f.write(summary)
print(summary)
