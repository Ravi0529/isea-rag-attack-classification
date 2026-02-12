from __future__ import annotations

import re

import pandas as pd

# Tool fingerprints from User-Agent (extend anytime)
TOOL_PATTERNS = [
    ("dirbuster", re.compile(r"dirbuster", re.I)),
    ("gobuster", re.compile(r"gobuster", re.I)),
    ("ffuf", re.compile(r"\bffuf\b", re.I)),
    ("feroxbuster", re.compile(r"feroxbuster", re.I)),
    ("nikto", re.compile(r"nikto", re.I)),
    ("sqlmap", re.compile(r"sqlmap", re.I)),
    ("nuclei", re.compile(r"\bnuclei\b", re.I)),
    ("nmap", re.compile(r"\bnmap\b", re.I)),
    ("masscan", re.compile(r"\bmasscan\b", re.I)),
    ("curl", re.compile(r"\bcurl/", re.I)),
    ("wget", re.compile(r"\bwget/", re.I)),
    ("python-requests", re.compile(r"python-requests", re.I)),
    ("go-http-client", re.compile(r"go-http-client", re.I)),
    ("httpx", re.compile(r"\bhttpx\b", re.I)),
    ("zgrab", re.compile(r"\bzgrab\b", re.I)),
    ("burp", re.compile(r"burp", re.I)),
    ("owasp-zap", re.compile(r"(owasp[- ]?zap|zaproxy)", re.I)),
]

# Indicators from captured fields and raw line text.
INDICATORS = {
    "lfi_etc_passwd": re.compile(r"/etc/passwd", re.I),
    "path_traversal": re.compile(
        r"(?:\.\./|%2e%2e%2f|%2e%2e/|\.\.%2f|\.\.\\|%5c)", re.I
    ),
    "sql_injection": re.compile(
        r"(\bunion\b|\bselect\b|\bdrop\b|\binsert\b|\bupdate\b|\bor\b\s+1=1|%27|%22)",
        re.I,
    ),
    "cmd_injection": re.compile(
        r"(;|\|\||&&|`|\$\(|%3b|%7c|%26%26|\bcmd\.exe\b|\b/bin/sh\b)", re.I
    ),
    "wp_probe": re.compile(r"(wp-admin|wp-login\.php|xmlrpc\.php)", re.I),
    "xss_probe": re.compile(
        r"(<script|%3cscript|onerror=|onload=|javascript:|alert\()", re.I
    ),
    "ssti_probe": re.compile(r"(\{\{.*\}\}|%7b%7b.*%7d%7d|\$\{.*\})", re.I),
    "rce_probe": re.compile(
        r"(?:\b(?:wget|curl|nc|bash|sh|powershell)\b.{0,30}(?:http://|https://))",
        re.I,
    ),
}


def _tool_from_ua(ua: str | None) -> str:
    if not ua:
        return "browser/unknown"
    for name, rx in TOOL_PATTERNS:
        if rx.search(ua):
            return name
    return "browser/unknown"


def _ensure_event_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize old/new event column variants to the latest ingest schema.
    """
    out = df.copy()
    rename_map = {
        "ts": "timestamp",
        "src_ip": "source_ip",
        "src_port": "source_port",
        "lang": "language",
        "action1": "captured_cmd",
        "action2": "captured_args",
        "extra": "x_forwarded_for/real-ip",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    required = [
        "captured_cmd",
        "captured_args",
        "timestamp",
        "source_ip",
        "source_port",
        "user_agent",
        "language",
        "x_forwarded_for/real-ip",
    ]
    for c in required:
        if c not in out.columns:
            out[c] = None
    return out


def enrich_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df columns expected:
      captured_cmd, captured_args, timestamp, source_ip, source_port,
      user_agent, language, x_forwarded_for/real-ip
    Returns:
      original columns + tool + ind_* + indicator_count
    """
    out = _ensure_event_schema(df)

    out["tool"] = out["user_agent"].astype("string").fillna("").map(_tool_from_ua)

    text = (
        out["captured_cmd"].astype("string").fillna("")
        + " "
        + out["captured_args"].astype("string").fillna("")
        + " "
        + out["user_agent"].astype("string").fillna("")
    )

    ind_cols = []
    for key, rx in INDICATORS.items():
        col = f"ind_{key}"
        out[col] = text.str.contains(rx, na=False)
        ind_cols.append(col)

    out["indicator_count"] = out[ind_cols].sum(axis=1).astype("int64")
    return out
