from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def _session_to_text(s: pd.DataFrame) -> list[str]:
    # Short behavior summary text per session for embedding.
    ind_cols = [c for c in s.columns if c.startswith("ind_")]
    texts = []
    for _, r in s.iterrows():
        ind_part = "; ".join(f"{c}={r.get(c, 0)}" for c in ind_cols)
        texts.append(
            f"tool={r.get('tool','')}; events={r.get('event_count',0)}; "
            f"duration_s={r.get('duration_s',0)}; rps={r.get('rps',0):.3f}; "
            f"unique_ports={r.get('unique_ports',0)}; indicator_hits={r.get('indicator_hits',0)}; "
            f"{ind_part}"
        )
    return texts


def _resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device

    if torch.cuda.is_available():
        try:
            # sanity check that CUDA is actually usable
            _ = torch.zeros(1).cuda()
            return "cuda"
        except Exception as e:
            print(f"CUDA available but unusable; falling back to CPU ({e})")

    return "cpu"


def embed_sessions(
    sessions: pd.DataFrame,
    model_name: str,
    device: str = "auto",
) -> np.ndarray:
    resolved_device = _resolve_device(device)
    print(f"Embedding device selected: {resolved_device}")

    texts = _session_to_text(sessions)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    try:
        model = SentenceTransformer(model_name, device=resolved_device)
    except Exception as e:
        print(f"Failed to load model on {resolved_device}: {e}")
        print("Falling back to CPU")
        model = SentenceTransformer(model_name, device="cpu")

    try:
        emb = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA OOM; retrying on CPU with smaller batch")
            torch.cuda.empty_cache()
            model.to("cpu")
            emb = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        else:
            raise

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
