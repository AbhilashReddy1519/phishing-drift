"""SYNTHETIC drifting URL stream, ONLY to smoke-test the pipeline.
Anything computed from this file is NOT a research result and must never go into the paper."""
import argparse, random, string, numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--weeks", type=int, default=10); ap.add_argument("--phish-per-week", type=int, default=400)
ap.add_argument("--ratio", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="data/synthetic_dataset.csv")
a = ap.parse_args()
random.seed(a.seed); rng = np.random.default_rng(a.seed)
WORDS = "apple river stone green cloud paper smart north global metro urban prime blue tech solar media trade market home travel garden music film sport news shop food study health city".split()
BRANDS = "paypal amazon netflix apple microsoft google meta dhl fedex hdfc sbi icici".split()
rnd = lambda n, ch=string.ascii_lowercase + string.digits: "".join(random.choice(ch) for _ in range(n))

def phish(family):
    if family == "A":   # long keyword subdomains on legacy TLDs
        return f"http://secure-login.{random.choice(BRANDS)}-verify{random.randint(1,99)}.{random.choice(['com','net','info'])}/signin/verify.php?id={rnd(8)}"
    if family == "B":   # short hyphenated brand look-alikes on cheap TLDs
        return f"https://{random.choice(BRANDS)}-{random.choice(WORDS)}.{random.choice(['xyz','top','site','club'])}/{rnd(6)}"
    if family == "C":   # random-looking short hosts on new TLDs (looks like many benign sites)
        return f"https://{rnd(random.randint(11,14),string.ascii_lowercase)}.{random.choice(['online','icu','shop','live'])}/"
    return f"http://{random.choice(WORDS)}{random.choice(WORDS)}{random.randint(10000,999999)}.{random.choice(['com','org','net'])}/wp-content/{rnd(5)}"  # D: word+digits hosts

def mix(week):  # family mixture drifts A -> B -> C -> D
    w = np.array([max(0.05, 1 - week / 3), max(0.05, 1 - abs(week - 3.5) / 3), max(0.05, 1 - abs(week - 6.5) / 3), max(0.05, (week - 5) / 4)])
    return w / w.sum()

rows, t0 = [], pd.Timestamp("2026-01-05")
for wk in range(a.weeks):
    n = a.phish_per_week
    fams = rng.choice(list("ABCD"), size=n, p=mix(wk))
    for fam in fams:
        rows.append((phish(fam), 1, t0 + pd.Timedelta(days=wk * 7 + rng.uniform(0, 7))))
    for _ in range(int(n * a.ratio)):
        r = random.random()
        if r < 0.55: host = f"www.{random.choice(WORDS)}{random.choice(WORDS)}.{random.choice(['com','org','net','in'])}"
        elif r < 0.75: host = f"{random.choice(WORDS)}.{random.choice(['com','io','co','org'])}"
        elif r < 0.85: host = f"{random.choice(WORDS)}-{random.choice(WORDS)}.{random.choice(['com','xyz','site','shop'])}"        # benign look-alikes of B
        elif r < 0.93: host = f"{rnd(random.randint(5,9), string.ascii_lowercase)}.{random.choice(['com','shop','live','online'])}"   # benign look-alikes of C
        else: host = f"{random.choice(WORDS)}{random.choice(WORDS)}{random.randint(1,99)}.{random.choice(['com','net','in'])}"       # benign look-alikes of D
        rows.append(("http://" + host + "/", 0, t0 + pd.Timedelta(days=wk * 7 + rng.uniform(0, 7))))
from features import registered_domain
df = pd.DataFrame(rows, columns=["url", "label", "ts"]).sort_values("ts").reset_index(drop=True)
df["domain"] = df["url"].map(registered_domain); df["target"] = ""; df["delay_hours"] = np.nan
df.to_csv(a.out, index=False); print("SYNTHETIC data written:", a.out, len(df), "rows")
