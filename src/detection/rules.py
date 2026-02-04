from __future__ import annotations
import pandas as pd


def apply_rules(sessions: pd.DataFrame) -> pd.DataFrame:
    s = sessions.copy()
    s["rule_score"] = 0.0
    s["rule_reasons"] = [[] for _ in range(len(s))]

    def add_reason(mask, score: float, reason: str):
        idx = s.index[mask]
        s.loc[mask, "rule_score"] += score
        for i in idx:
            s.at[i, "rule_reasons"].append(reason)

    # tool-based
    add_reason(
        s["tool"].isin(["gobuster", "dirbuster"]), 0.85, "tool_fingerprint:dir_enum"
    )
    add_reason(
        s["tool"].isin(["go-http-client", "python-requests", "curl", "wget"]),
        0.20,
        "tool_fingerprint:automation",
    )

    # indicator hits (from Phase 2)
    add_reason(s["ind_lfi_etc_passwd"] > 0, 0.90, "indicator:/etc/passwd")
    add_reason(s["ind_path_traversal"] > 0, 0.70, "indicator:path_traversal")
    add_reason(s["ind_sql_injection"] > 0, 0.70, "indicator:sql_injection")
    add_reason(s["ind_cmd_injection"] > 0, 0.70, "indicator:cmd_injection")
    add_reason(s["ind_wp_probe"] > 0, 0.40, "indicator:wp_probe")

    # behavior thresholds
    add_reason(s["rps"] >= 3.0, 0.35, "behavior:burst_rps>=3")
    add_reason(s["event_count"] >= 300, 0.35, "behavior:high_volume>=300")
    add_reason(s["unique_ports"] >= 20, 0.25, "behavior:many_ports")

    # cap to [0, 1]
    s["rule_score"] = s["rule_score"].clip(0, 1)
    return s
