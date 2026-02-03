from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def _session_to_text(s: pd.DataFrame) -> list[str]:
    # short “behavior summary” text per session
    texts = []
    for _, r in s.iterrows():
        texts.append(
            f"tool={r.get('tool','')}; events={r.get('event_count',0)}; "
            f"duration_s={r.get('duration_s',0)}; rps={r.get('rps',0):.3f}; "
            f"unique_ports={r.get('unique_ports',0)}; indicator_hits={r.get('indicator_hits',0)}; "
            f"etc_passwd={r.get('ind_lfi_etc_passwd',0)}; traversal={r.get('ind_path_traversal',0)}; "
            f"sqli={r.get('ind_sql_injection',0)}; cmdi={r.get('ind_cmd_injection',0)}; wp={r.get('ind_wp_probe',0)}"
        )
    return texts


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed_sessions(
    sessions: pd.DataFrame, model_name: str, device: str = "auto"
) -> np.ndarray:
    resolved_device = _resolve_device(device)
    print(f"🔧 Using device: {resolved_device}")

    model = SentenceTransformer(model_name, device=resolved_device)
    texts = _session_to_text(sessions)

    emb = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def isolation_forest_scores(X: np.ndarray, seed: int = 42) -> np.ndarray:
    clf = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X)
    # decision_function: higher = more normal => invert to anomaly
    normality = clf.decision_function(X)
    anomaly = normality.max() - normality  # bigger = more anomalous
    # normalize 0..1
    anomaly = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-9)
    return anomaly


def lof_scores(X: np.ndarray) -> np.ndarray:
    lof = LocalOutlierFactor(n_neighbors=35, metric="cosine")
    # fit_predict returns -1 outliers, but we want continuous score: use negative_outlier_factor_
    lof.fit_predict(X)
    score = -lof.negative_outlier_factor_
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return score
