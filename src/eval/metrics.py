from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


VALID_LABELS = ["benign", "suspicious", "attack_like"]


def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _as_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, float) and pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(v).strip() for v in arr if str(v).strip()]
        except Exception:
            pass
    if "|" in s:
        return [p.strip() for p in s.split("|") if p.strip()]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _flatten_tokens(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, str):
        return _as_list(x)
    if isinstance(x, (list, tuple, set)):
        out: list[str] = []
        for v in x:
            out.extend(_flatten_tokens(v))
        return out
    if hasattr(x, "tolist"):
        try:
            return _flatten_tokens(x.tolist())
        except Exception:
            return _as_list(str(x))
    return _as_list(str(x))


def _detection_core(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=VALID_LABELS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
    per_class = {}
    for i, lbl in enumerate(VALID_LABELS):
        per_class[lbl] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
    return {
        "label_order": VALID_LABELS,
        "confusion_matrix": cm.tolist(),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=VALID_LABELS, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=VALID_LABELS, average="weighted")
        ),
        "per_class": per_class,
    }


def detection_metrics_labeled(
    sessions_scored: pd.DataFrame,
    detection_labels: pd.DataFrame,
) -> dict[str, Any]:
    req = {"session_id", "true_label"}
    missing = req - set(detection_labels.columns)
    if missing:
        raise ValueError(f"detection labels missing columns: {sorted(missing)}")

    lbl = detection_labels.copy()
    lbl["true_label"] = lbl["true_label"].astype("string").str.strip().str.lower()
    bad = sorted(set(lbl["true_label"].dropna()) - set(VALID_LABELS))
    if bad:
        raise ValueError(f"invalid true_label values: {bad}")

    merged = lbl.merge(
        sessions_scored[["session_id", "label"]], on="session_id", how="left"
    ).dropna(subset=["label"])
    merged["label"] = merged["label"].astype("string").str.strip().str.lower()

    y_true = merged["true_label"].tolist()
    y_pred = merged["label"].tolist()
    out = _detection_core(y_true, y_pred)
    out.update(
        {
            "mode": "labeled",
            "n_evaluated": int(len(merged)),
            "coverage_vs_labels": float(len(merged) / max(len(lbl), 1)),
        }
    )
    return out


def detection_metrics_proxy(sessions_scored: pd.DataFrame) -> dict[str, Any]:
    s = sessions_scored.copy()
    s["proxy_true"] = "unknown"

    strong_attack = (
        (s["ind_lfi_etc_passwd"] > 0)
        | (s["ind_path_traversal"] > 0)
        | (s["ind_sql_injection"] > 0)
        | (s["ind_cmd_injection"] >= 3)
        | ((s["tool"].isin(["gobuster", "dirbuster"])) & (s["event_count"] >= 30))
    )
    clean_benign = (
        (s["indicator_hits"] == 0)
        & (s["tool"] == "browser/unknown")
        & (s["rps"] < 0.5)
        & (s["event_count"] <= 5)
        & (s["unique_ports"] <= 3)
    )
    moderate = (
        (s["indicator_hits"] > 0) | (s["rps"] >= 3.0) | (s["unique_ports"] >= 20)
    ) & (~strong_attack)

    s.loc[clean_benign, "proxy_true"] = "benign"
    s.loc[moderate, "proxy_true"] = "suspicious"
    s.loc[strong_attack, "proxy_true"] = "attack_like"

    eval_df = s[s["proxy_true"] != "unknown"].copy()
    y_true = eval_df["proxy_true"].tolist()
    y_pred = eval_df["label"].astype("string").str.lower().tolist()
    out = _detection_core(y_true, y_pred)
    out.update(
        {
            "mode": "proxy",
            "n_evaluated": int(len(eval_df)),
            "coverage_vs_sessions": float(len(eval_df) / max(len(s), 1)),
            "notes": "Proxy labels are heuristic and not ground truth.",
        }
    )
    return out


def mapping_metrics_labeled(
    session_mapping: pd.DataFrame,
    mapping_labels: pd.DataFrame,
    retrieval_k: int = 10,
) -> dict[str, Any]:
    req = {"session_id", "true_technique_ids"}
    missing = req - set(mapping_labels.columns)
    if missing:
        raise ValueError(f"mapping labels missing columns: {sorted(missing)}")

    truth = mapping_labels[["session_id", "true_technique_ids"]].copy()
    truth["true_technique_ids"] = truth["true_technique_ids"].apply(_as_list)

    pred = session_mapping[["session_id", "mapped_techniques"]].copy()
    pred["mapped_techniques"] = pred["mapped_techniques"].apply(_as_list)

    merged = truth.merge(pred, on="session_id", how="left")
    merged["mapped_techniques"] = merged["mapped_techniques"].apply(_as_list)
    merged = merged[
        (merged["true_technique_ids"].map(len) > 0)
        & (merged["mapped_techniques"].map(len) > 0)
    ].copy()

    if merged.empty:
        raise ValueError(
            "No evaluable mapping rows. Check session_id joins and labels."
        )

    def _hit(row: pd.Series, k: int) -> int:
        predk = row["mapped_techniques"][:k]
        truth_set = set(row["true_technique_ids"])
        return int(any(t in truth_set for t in predk))

    top1 = merged.apply(lambda r: _hit(r, 1), axis=1).mean()
    top3 = merged.apply(lambda r: _hit(r, 3), axis=1).mean()
    rak = merged.apply(lambda r: _hit(r, retrieval_k), axis=1).mean()

    return {
        "mode": "labeled",
        "n_evaluated": int(len(merged)),
        "coverage_vs_labels": float(len(merged) / max(len(truth), 1)),
        "top1_accuracy": float(top1),
        "top3_accuracy": float(top3),
        "retrieval_recall_at_k": {"k": int(retrieval_k), "value": float(rak)},
    }


def mapping_metrics_proxy(session_mapping: pd.DataFrame) -> dict[str, Any]:
    m = session_mapping.copy()
    m["mapped_tactics_flat"] = m["mapped_tactics"].apply(
        lambda xs: [str(t).lower() for t in _flatten_tokens(xs)]
    )
    m["why_list"] = m["why"].apply(_flatten_tokens)

    expected = []
    for reasons in m["why_list"]:
        e = set()
        text = " ".join(reasons).lower()
        if any(
            k in text for k in ["dir_enum", "many_ports", "high_volume", "wp_probe"]
        ):
            e.add("discovery")
        if "cmd_injection" in text:
            e.add("execution")
        if any(k in text for k in ["path_traversal", "/etc/passwd", "sql_injection"]):
            e.add("initial-access")
        if not e:
            e.add("discovery")
        expected.append(sorted(e))
    m["expected_tactics"] = expected

    def _tactic_hit(row: pd.Series, k: int) -> int:
        pred = row["mapped_tactics_flat"][:k]
        truth = set(row["expected_tactics"])
        return int(any(p in truth for p in pred))

    n = len(m)
    top1 = m.apply(lambda r: _tactic_hit(r, 1), axis=1).mean() if n else 0.0
    top3 = m.apply(lambda r: _tactic_hit(r, 3), axis=1).mean() if n else 0.0
    mean_conf = float(m["confidence"].mean()) if "confidence" in m.columns else 0.0

    return {
        "mode": "proxy",
        "n_evaluated": int(n),
        "tactic_alignment_top1": float(top1),
        "tactic_alignment_top3": float(top3),
        "avg_confidence": mean_conf,
        "notes": "Proxy mapping metrics are heuristic alignment, not true technique accuracy.",
    }
