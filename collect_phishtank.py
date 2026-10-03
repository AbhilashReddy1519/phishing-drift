"""Daily snapshot of the PhishTank 'online-valid' feed. Run once per day (cron / Task Scheduler).
Appends NEW phish_id rows to data/phishtank_raw.csv. Never visits any phishing URL.
Needs: a PhishTank account + application key (free); set env PHISHTANK_APPKEY and PHISHTANK_USER.
Check the current terms at https://phishtank.org/developer_info.php before use."""
import os, sys, io, csv, datetime, urllib.request, urllib.error

OUT = "data/phishtank_raw.csv"
FIELDS = ["phish_id", "url", "submission_time", "verification_time", "target", "first_seen_utc"]

def main():
    key = os.environ.get("PHISHTANK_APPKEY", "")
    user = os.environ.get("PHISHTANK_USER", "phishing-drift-project")
    url = f"http://data.phishtank.com/data/{key}/online-valid.csv" if key else "http://data.phishtank.com/data/online-valid.csv"
    req = urllib.request.Request(url, headers={"User-Agent": f"phishtank/{user}"})
    try:
        raw = urllib.request.urlopen(req, timeout=120).read()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: PhishTank rejected the request. Check access, your app key, or rate limits.")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    required = {"phish_id", "url", "submission_time", "verification_time", "target"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        sys.exit("Unexpected PhishTank CSV format; required columns are missing.")
    rows = list(reader)
    seen = set()
    if os.path.exists(OUT):
        with open(OUT, newline="", encoding="utf-8") as f:
            seen = {r["phish_id"] for r in csv.DictReader(f)}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    new = [r for r in rows if str(r["phish_id"]) not in seen]
    os.makedirs("data", exist_ok=True)
    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header: w.writeheader()
        for r in new:
            w.writerow({"phish_id": r["phish_id"], "url": r["url"], "submission_time": r["submission_time"],
                        "verification_time": r.get("verification_time", ""), "target": r.get("target", ""), "first_seen_utc": now})
    print(f"feed rows: {len(rows)}  new rows appended: {len(new)}  total stored: {len(seen) + len(new)}")

if __name__ == "__main__":
    main()
