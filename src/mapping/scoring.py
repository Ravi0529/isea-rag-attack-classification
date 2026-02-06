# src/mapping/scoring.py
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

BOOSTS = {
    "tool:gobuster": 0.15,
    "tool:dirbuster": 0.15,
    "indicator:cmd_injection": 0.15,
    "indicator:sql_injection": 0.15,
    "indicator:path_traversal": 0.15,
    "indicator:lfi_etc_passwd": 0.20,
    "behavior:burst_rps>=3": 0.10,
    "behavior:many_ports": 0.10,
}


def _as_list(x: Any) -> list:
    """Convert pandas/numpy/None/singleton into a clean Python list."""
    if x is None:
        return []
    # NaN check (but avoid crashing on lists/arrays)
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    # Already list/tuple/set
    if isinstance(x, (list, tuple, set)):
        return list(x)

    # numpy array / pandas Series
    try:
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass

    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass

    # fallback: single item
    return [x]


def compute_boost(row: dict) -> float:
    b = 0.0

    tool = row.get("tool")
    if tool in ["gobuster", "dirbuster"]:
        b += BOOSTS.get(f"tool:{tool}", 0.0)

    reasons = _as_list(row.get("rule_reasons"))
    # (optional) your data sometimes uses "reasons" too
    if not reasons:
        reasons = _as_list(row.get("reasons"))

    for r in reasons:
        b += BOOSTS.get(str(r), 0.0)

    return min(b, 0.35)  # cap boosts


def final_score(sim: float, boost: float) -> float:
    """
    Combine vector similarity (0..1-ish) with rule-based boost (0..0.35).
    Keeps result in [0, 1].
    """
    # Clamp sim into [0,1] just in case your distance/score is outside
    sim = float(sim)
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0

    boost = float(boost)
    if boost < 0.0:
        boost = 0.0
    if boost > 0.35:
        boost = 0.35

    # Weighted mix (tune later)
    conf = (0.85 * sim) + (0.15 * (boost / 0.35))

    # Final clamp
    return max(0.0, min(1.0, conf))
