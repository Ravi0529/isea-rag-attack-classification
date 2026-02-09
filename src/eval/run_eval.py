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
from src.eval.plots import save_confusion_matrix, save_score_bars


def create_eval_templates(
    sessions_scored_path: str,
    session_mapping_path: str,
    out_dir: str = "data/labels",
    sample_rows: int = 1000,
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
        det = det.head(sample_rows)

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
        mp = mp.head(sample_rows)

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
        "mapping": mp,
        "artifacts": {
            "detection_confusion_matrix": cm_path,
            "mapping_scores_plot": bar_path,
        },
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
