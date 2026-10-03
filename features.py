"""Lexical URL features. No network access, no page fetching: URLs are only parsed as strings."""
import math, ipaddress
from collections import Counter
from urllib.parse import urlsplit
import numpy as np, pandas as pd
import tldextract

# bundled public-suffix snapshot: works offline, no downloads
_EXT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

KEYWORDS = ["login", "signin", "secure", "verify", "account", "update", "bank", "confirm",
            "password", "wallet", "support", "billing", "service", "webmail", "auth"]
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly", "cutt.ly", "rb.gy", "shorturl.at"}
TLDS = ["com", "net", "org", "info", "xyz", "top", "online", "site", "club", "shop", "app", "io", "co",
        "in", "ru", "cn", "br", "de", "uk", "tk", "ml", "ga", "cf", "gq", "icu", "vip", "live", "link", "cc"]

def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())

def _tokens(s):
    cur, out = "", []
    for ch in s:
        if ch.isalnum():
            cur += ch
        else:
            if cur: out.append(cur)
            cur = ""
    if cur: out.append(cur)
    return out

def host_features(host: str) -> dict:
    ext = _EXT(host)
    try:
        ipaddress.ip_address(host.strip("[]")); is_ip = 1
    except ValueError:
        is_ip = 0
    toks = _tokens(host)
    f = {
        "h_len": len(host), "h_dots": host.count("."), "h_hyph": host.count("-"),
        "h_digits": sum(c.isdigit() for c in host), "h_digit_ratio": sum(c.isdigit() for c in host) / max(1, len(host)),
        "h_sub_depth": 0 if not ext.subdomain else ext.subdomain.count(".") + 1,
        "h_is_ip": is_ip, "h_entropy": entropy(host), "h_longest_tok": max((len(t) for t in toks), default=0),
        "h_ntok": len(toks), "h_punycode": int("xn--" in host), "h_kw": sum(k in host for k in KEYWORDS),
        "h_short": int(host in SHORTENERS), "h_domain_len": len(ext.domain), "h_domain_entropy": entropy(ext.domain),
        "h_suffix_len": len(ext.suffix),
    }
    suf = ext.suffix.split(".")[-1] if ext.suffix else ""
    for t in TLDS:
        f["tld_" + t] = int(suf == t)
    f["tld_other"] = int(suf not in TLDS)
    return f

def path_features(sp) -> dict:
    p, q = sp.path or "", sp.query or ""
    return {"p_len": len(p), "q_len": len(q), "p_slashes": p.count("/"), "q_params": q.count("&") + (1 if q else 0),
            "p_entropy": entropy(p), "p_kw": sum(k in p.lower() for k in KEYWORDS), "u_at": int("@" in sp.netloc),
            "u_pct": (p + q).count("%"), "p_digits": sum(c.isdigit() for c in p)}

def registered_domain(url: str) -> str:
    if "://" not in url: url = "http://" + url
    e = _EXT(urlsplit(url).hostname or "")
    return e.registered_domain or (urlsplit(url).hostname or "")

def extract(url: str, mode: str = "host") -> dict:
    u = url if "://" in url else "http://" + url
    try:
        sp = urlsplit(u)
        host = (sp.hostname or "").lower()
    except ValueError:
        sp, host = urlsplit("http://invalid/"), "invalid"
    f = host_features(host)
    if mode == "full":
        f.update(path_features(sp)); f["u_len"] = len(url)
    return f

def feature_matrix(urls, mode="host") -> pd.DataFrame:
    return pd.DataFrame([extract(u, mode) for u in urls]).astype(np.float32)
