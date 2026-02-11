from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.eval.metrics import (
    detection_metrics_labeled,
    detection_metrics_proxy,
    mapping_metrics_labeled,
    mapping_metrics_proxy,
)
from src.eval.plots import (
    save_confusion_matrix,
    save_histogram,
    save_score_bars,
    save_tsne_scatter,
)


def _add_bucket(
    df: pd.DataFrame, value_col: str, out_col: str, n_bins: int
) -> pd.DataFrame:
    out = df.copy()
    bins = max(int(n_bins), 1)
    try:
        out[out_col] = pd.qcut(out[value_col], q=bins, duplicates="drop")
        out[out_col] = out[out_col].astype("string")
    except Exception:
        out[out_col] = "all"
    return out


def _stratified_sample(
    df: pd.DataFrame, group_cols: list[str], n_total: int, random_seed: int
) -> pd.DataFrame:
    if n_total <= 0 or n_total >= len(df):
        return df.copy()
    sizes = (
        df.groupby(group_cols, dropna=False, observed=False)
        .size()
        .reset_index(name="size")
        .sort_values("size", ascending=False)
        .reset_index(drop=True)
    )
    if sizes.empty:
        return df.head(n_total).copy()

    sizes["target"] = (sizes["size"] / sizes["size"].sum() * n_total).astype(int)
    remaining = n_total - int(sizes["target"].sum())
    if remaining > 0:
        sizes["frac"] = sizes["size"] / sizes["size"].sum() * n_total - sizes["target"]
        for idx in sizes.sort_values("frac", ascending=False).index:
            if remaining == 0:
                break
            sizes.at[idx, "target"] += 1
            remaining -= 1
    sizes["target"] = sizes[["target", "size"]].min(axis=1)

    joined = df.merge(sizes[group_cols + ["target"]], on=group_cols, how="left")
    parts = []
    for _, g in joined.groupby(group_cols, dropna=False, observed=False):
        n = int(g["target"].iloc[0])
        if n <= 0:
            continue
        parts.append(g.sample(n=min(n, len(g)), random_state=random_seed))
    if not parts:
        return df.head(n_total).copy()
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n_total:
        out = out.sample(n=n_total, random_state=random_seed)
    return out.drop(columns=["target"], errors="ignore").reset_index(drop=True)


def create_eval_templates(
    sessions_scored_path: str,
    session_mapping_path: str,
    out_dir: str = "data/labels",
    sample_rows: int = 1000,
    stratified: bool = True,
    score_bins: int = 4,
    random_seed: int = 42,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    s = pd.read_parquet(sessions_scored_path)
    m = pd.read_parquet(session_mapping_path)

    det = s[
        ["session_id", "label", "suspicious_score", "tool", "src_ip", "rule_reasons"]
    ].copy()
    det = det.rename(columns={"label": "pred_label"})
    det["true_label"] = ""
    if sample_rows and sample_rows > 0:
        if stratified:
            det = _add_bucket(det, "suspicious_score", "score_bucket", score_bins)
            det = _stratified_sample(
                det,
                group_cols=["pred_label", "score_bucket"],
                n_total=sample_rows,
                random_seed=random_seed,
            )
            det = det.drop(columns=["score_bucket"], errors="ignore")
        else:
            det = det.sample(
                n=min(sample_rows, len(det)), random_state=random_seed
            ).reset_index(drop=True)

    mp = m[
        [
            "session_id",
            "mapped_techniques",
            "mapped_names",
            "confidence",
            "why",
            "summary",
        ]
    ].copy()
    mp = mp.rename(
        columns={
            "mapped_techniques": "pred_mapped_techniques",
            "mapped_names": "pred_mapped_names",
            "confidence": "pred_confidence",
            "why": "pred_why",
        }
    )
    mp["true_technique_ids"] = ""
    if sample_rows and sample_rows > 0:
        if stratified:
            mp["pred_label"] = m["label"].astype("string")
            mp = _add_bucket(mp, "pred_confidence", "conf_bucket", score_bins)
            mp = _stratified_sample(
                mp,
                group_cols=["pred_label", "conf_bucket"],
                n_total=sample_rows,
                random_seed=random_seed,
            )
            mp = mp.drop(columns=["conf_bucket"], errors="ignore")
        else:
            mp = mp.sample(n=min(sample_rows, len(mp)), random_state=random_seed)

    det_path = os.path.join(out_dir, "detection_labels.csv")
    map_path = os.path.join(out_dir, "mapping_labels.csv")
    det.to_csv(det_path, index=False)
    mp.to_csv(map_path, index=False)

    return {"detection_labels": det_path, "mapping_labels": map_path}


def run_eval(
    sessions_scored_path: str,
    session_mapping_path: str,
    attack_cache_path: str,
    out_json_path: str = "reports/metrics.json",
    figures_dir: str = "reports/figures",
    mode: str = "proxy",
    detection_labels_path: Optional[str] = None,
    mapping_labels_path: Optional[str] = None,
    retrieval_k: int = 10,
) -> dict:
    if mode not in {"proxy", "labeled"}:
        raise ValueError("mode must be one of: proxy, labeled")

    s = pd.read_parquet(sessions_scored_path)
    m = pd.read_parquet(session_mapping_path)

    # Touch cache path for provenance/check only (phase contract).
    if not os.path.exists(attack_cache_path):
        raise FileNotFoundError(f"attack cache not found: {attack_cache_path}")

    if mode == "labeled":
        if not detection_labels_path or not mapping_labels_path:
            raise ValueError(
                "labeled mode requires detection_labels_path and mapping_labels_path"
            )
        det_lbl = pd.read_csv(detection_labels_path)
        map_lbl = pd.read_csv(mapping_labels_path)
        det = detection_metrics_labeled(s, det_lbl)
        mp = mapping_metrics_labeled(m, map_lbl, retrieval_k=retrieval_k)
    else:
        det = detection_metrics_proxy(s)
        mp = mapping_metrics_proxy(m)

    # Confidence metrics over full unlabeled mapped set.
    sim_col = "sim_top1" if "sim_top1" in m.columns else None
    if sim_col is None and "similarity" in m.columns:
        sim_col = "similarity"
    if sim_col is None and "sim" in m.columns:
        sim_col = "sim"
    if sim_col is None:
        sim_col = "confidence"

    conf_series = pd.to_numeric(m[sim_col], errors="coerce").fillna(0.0)
    conf_metrics = {
        "similarity_source_column": sim_col,
        "mean_cosine_similarity": float(conf_series.mean()),
        "median_cosine_similarity": float(conf_series.median()),
        "std_cosine_similarity": float(conf_series.std(ddof=0)),
    }

    # Utility metrics over full unlabeled mapped set.
    utility_threshold = 0.70
    high_conf = pd.to_numeric(m["confidence"], errors="coerce").fillna(0.0) >= utility_threshold
    util_metrics = {
        "threshold": utility_threshold,
        "n_mapped_sessions": int(len(m)),
        "n_mapped_sessions_above_threshold": int(high_conf.sum()),
        "pct_mapped_sessions_above_threshold": float(high_conf.mean() if len(m) else 0.0),
        "mapping_coverage_vs_scored_sessions": float(len(m) / max(len(s), 1)),
    }

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    cm_path = os.path.join(figures_dir, "detection_confusion_matrix.png")
    save_confusion_matrix(det["confusion_matrix"], det["label_order"], cm_path)

    if mp.get("mode") == "labeled":
        bar_scores = {
            "top1": mp["top1_accuracy"],
            "top3": mp["top3_accuracy"],
            f"recall@{mp['retrieval_recall_at_k']['k']}": mp["retrieval_recall_at_k"][
                "value"
            ],
        }
        bar_title = "Labeled Mapping Metrics"
        bar_path = os.path.join(figures_dir, "mapping_labeled_metrics.png")
    else:
        bar_scores = {
            "tactic_top1": mp["tactic_alignment_top1"],
            "tactic_top3": mp["tactic_alignment_top3"],
            "avg_conf": mp["avg_confidence"],
        }
        bar_title = "Proxy Mapping Metrics"
        bar_path = os.path.join(figures_dir, "mapping_proxy_metrics.png")

    save_score_bars(bar_scores, bar_path, bar_title)

    # Confidence histogram plot.
    conf_hist_path = os.path.join(figures_dir, "similarity_histogram.png")
    save_histogram(
        conf_series.values,
        out_path=conf_hist_path,
        title="Similarity Distribution",
        xlabel=sim_col,
        bins=30,
    )

    # Visual t-SNE plot over full unlabeled sessions.
    tsne_features = [
        c
        for c in [
            "suspicious_score",
            "rule_score",
            "ml_score",
            "event_count",
            "duration_s",
            "rps",
            "unique_ports",
            "indicator_hits",
            "ind_lfi_etc_passwd",
            "ind_path_traversal",
            "ind_sql_injection",
            "ind_cmd_injection",
            "ind_wp_probe",
        ]
        if c in s.columns
    ]
    tsne_path = os.path.join(figures_dir, "tsne_scatter.png")
    if tsne_features and "label" in s.columns:
        save_tsne_scatter(
            s,
            feature_cols=tsne_features,
            label_col="label",
            out_path=tsne_path,
            max_points=4000,
            random_state=42,
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "inputs": {
            "sessions_scored_path": sessions_scored_path,
            "session_mapping_path": session_mapping_path,
            "attack_cache_path": attack_cache_path,
            "detection_labels_path": detection_labels_path,
            "mapping_labels_path": mapping_labels_path,
            "retrieval_k": retrieval_k,
        },
        "detection": det,
        "confidence": conf_metrics,
        "utility": util_metrics,
        "mapping": mp,
        "artifacts": {
            "detection_confusion_matrix": cm_path,
            "mapping_scores_plot": bar_path,
            "similarity_histogram": conf_hist_path,
            "tsne_scatter": tsne_path,
        },
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
