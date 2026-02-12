from __future__ import annotations
import pandas as pd


def apply_rules(sessions: pd.DataFrame) -> pd.DataFrame:
    s = sessions.copy()
    s["rule_score"] = 0.0
    s["rule_reasons"] = [[] for _ in range(len(s))]

    def _col(name: str, default=0):
        if name in s.columns:
            return s[name]
        return pd.Series([default] * len(s), index=s.index)

    def add_reason(mask, score: float, reason: str):
        idx = s.index[mask]
        s.loc[mask, "rule_score"] += score
        for i in idx:
            s.at[i, "rule_reasons"].append(reason)

    # tool-based
    add_reason(
        _col("tool", "").isin(["gobuster", "dirbuster", "ffuf", "feroxbuster"]),
        0.85,
        "tool_fingerprint:dir_enum",
    )
    add_reason(
        _col("tool", "").isin(
            [
                "go-http-client",
                "python-requests",
                "curl",
                "wget",
                "httpx",
                "zgrab",
            ]
        ),
        0.20,
        "tool_fingerprint:automation",
    )
    add_reason(
        _col("tool", "").isin(["sqlmap", "nikto", "nuclei", "owasp-zap", "burp"]),
        0.60,
        "tool_fingerprint:scanner",
    )

    # indicator hits (from Phase 2)
    add_reason(_col("ind_lfi_etc_passwd") > 0, 0.90, "indicator:/etc/passwd")
    add_reason(_col("ind_path_traversal") > 0, 0.70, "indicator:path_traversal")
    add_reason(_col("ind_sql_injection") > 0, 0.70, "indicator:sql_injection")
    add_reason(_col("ind_cmd_injection") > 0, 0.70, "indicator:cmd_injection")
    add_reason(_col("ind_wp_probe") > 0, 0.40, "indicator:wp_probe")
    add_reason(_col("ind_xss_probe") > 0, 0.45, "indicator:xss_probe")
    add_reason(_col("ind_ssti_probe") > 0, 0.55, "indicator:ssti_probe")
    add_reason(_col("ind_rce_probe") > 0, 0.80, "indicator:rce_probe")

    # behavior thresholds
    add_reason(_col("rps").fillna(0) >= 3.0, 0.35, "behavior:burst_rps>=3")
    add_reason(_col("event_count").fillna(0) >= 300, 0.35, "behavior:high_volume>=300")
    add_reason(_col("unique_ports").fillna(0) >= 20, 0.25, "behavior:many_ports")

    # cap to [0, 1]
    s["rule_score"] = s["rule_score"].clip(0, 1)
    return s
