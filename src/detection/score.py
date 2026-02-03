from __future__ import annotations
import pandas as pd


def combine_scores(df: pd.DataFrame, w_rule=0.65, w_ml=0.35) -> pd.DataFrame:
    out = df.copy()
    out["ml_score"] = out["ml_score"].fillna(0.0).clip(0, 1)
    out["rule_score"] = out["rule_score"].fillna(0.0).clip(0, 1)

    out["suspicious_score"] = (
        w_rule * out["rule_score"] + w_ml * out["ml_score"]
    ).clip(0, 1)

    # label thresholds (tune later)
    out["label"] = "benign"
    out.loc[out["suspicious_score"] >= 0.70, "label"] = "attack_like"
    out.loc[
        (out["suspicious_score"] >= 0.40) & (out["suspicious_score"] < 0.70), "label"
    ] = "suspicious"

    # build reasons
    out["reasons"] = out["rule_reasons"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    out.loc[out["ml_score"] >= 0.80, "reasons"] = out.loc[
        out["ml_score"] >= 0.80, "reasons"
    ].apply(lambda r: r + ["ml_anomaly:high"])
    return out
