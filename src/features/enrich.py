from __future__ import annotations
import re
import pandas as pd

# Tool fingerprints from UA (extend anytime)
TOOL_PATTERNS = [
    ("dirbuster", re.compile(r"dirbuster", re.I)),
    ("gobuster", re.compile(r"gobuster", re.I)),
    ("nikto", re.compile(r"nikto", re.I)),
    ("sqlmap", re.compile(r"sqlmap", re.I)),
    ("curl", re.compile(r"\bcurl/", re.I)),
    ("wget", re.compile(r"\bwget/", re.I)),
    ("python-requests", re.compile(r"python-requests", re.I)),
    ("go-http-client", re.compile(r"go-http-client", re.I)),
]

# Indicators from raw_line / action2 (path etc.)
INDICATORS = {
    "lfi_etc_passwd": re.compile(r"/etc/passwd", re.I),
    "path_traversal": re.compile(
        r"(?:\.\./|%2e%2e%2f|%2e%2e/|\.\.%2f|\.\.\\|%5c)", re.I
    ),
    "sql_injection": re.compile(
        r"(\bunion\b|\bselect\b|\bdrop\b|\bor\b\s+1=1|%27|%22)", re.I
    ),
    "cmd_injection": re.compile(r"(;|\|\||\|`|`|\$\(|%3b|%7c)", re.I),
    "wp_probe": re.compile(r"(wp-admin|wp-login\.php|xmlrpc\.php)", re.I),
}


def _tool_from_ua(ua: str | None) -> str:
    if not ua:
        return "browser/unknown"
    for name, rx in TOOL_PATTERNS:
        if rx.search(ua):
            return name
    return "browser/unknown"


def enrich_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns expected: ts, src_ip, src_port, user_agent, lang, action1, action2, extra, raw_line
    Returns df with extra columns:
      tool, ind_*, indicator_count
    """
    out = df.copy()

    # tool fingerprint
    out["tool"] = out["user_agent"].astype("string").fillna("").map(_tool_from_ua)

    # searchable text field for indicator matching
    text = (
        out["raw_line"].astype("string").fillna("")
        + " "
        + out.get("action1", pd.Series([None] * len(out))).astype("string").fillna("")
        + " "
        + out.get("action2", pd.Series([None] * len(out))).astype("string").fillna("")
    )

    # indicator flags
    ind_cols = []
    for key, rx in INDICATORS.items():
        col = f"ind_{key}"
        out[col] = text.str.contains(rx)
        ind_cols.append(col)

    out["indicator_count"] = out[ind_cols].sum(axis=1).astype("int64")
    return out
