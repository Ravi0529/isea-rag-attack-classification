from __future__ import annotations
import pandas as pd
from qdrant_client.models import PointStruct

from src.rag.qdrant_client import get_qdrant, ensure_collection
from src.rag.embeddings import load_embedder, embed_texts


def _session_summary(r: pd.Series) -> str:
    return (
        f"Session from {r.get('src_ip')} tool={r.get('tool')} "
        f"label={r.get('label')} score={float(r.get('suspicious_score',0)):.2f}. "
        f"events={int(r.get('event_count',0))} duration_s={float(r.get('duration_s',0)):.1f} rps={float(r.get('rps',0)):.3f}. "
        f"indicators: hits={int(r.get('indicator_hits',0))}, "
        f"etc_passwd={int(r.get('ind_lfi_etc_passwd',0))}, "
        f"traversal={int(r.get('ind_path_traversal',0))}, "
        f"sqli={int(r.get('ind_sql_injection',0))}, "
        f"cmdi={int(r.get('ind_cmd_injection',0))}, "
        f"wp_probe={int(r.get('ind_wp_probe',0))}. "
        f"reasons={r.get('reasons')}"
    )


def index_log_sessions(
    scored_sessions_path: str = "data/processed/sessions_scored.parquet",
    collection: str = "log_sessions",
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "auto",
    batch_size: int = 256,
) -> None:
    s = pd.read_parquet(scored_sessions_path)
    # make sure timestamps are JSON/Qdrant friendly
    s["start_ts"] = pd.to_datetime(s["start_ts"], utc=True, errors="coerce")
    s["end_ts"] = pd.to_datetime(s["end_ts"], utc=True, errors="coerce")

    texts = s.apply(_session_summary, axis=1).tolist()

    model = load_embedder(embed_model, device=device)
    dim = int(model.get_sentence_embedding_dimension())

    client = get_qdrant()
    ensure_collection(client, collection, dim)

    # numeric ids for qdrant (stable by row index)
    ids = list(range(len(s)))

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        vecs = embed_texts(model, chunk, batch_size=min(256, batch_size))

        points = []
        for j, _ in enumerate(chunk):
            row = s.iloc[start + j]
            payload = {
                "session_id": row.get("session_id"),
                "src_ip": row.get("src_ip"),
                "tool": row.get("tool"),
                "label": row.get("label"),
                "suspicious_score": float(row.get("suspicious_score", 0.0)),
                "event_count": int(row.get("event_count", 0)),
                "rps": float(row.get("rps", 0.0)),
                "indicator_hits": int(row.get("indicator_hits", 0)),
                "start_ts": (
                    None if pd.isna(row["start_ts"]) else row["start_ts"].isoformat()
                ),
                "end_ts": None if pd.isna(row["end_ts"]) else row["end_ts"].isoformat(),
            }
            points.append(
                PointStruct(id=ids[start + j], vector=vecs[j].tolist(), payload=payload)
            )

        client.upsert(collection_name=collection, points=points)

    print(f"✅ indexed {len(s)} sessions -> {collection}")
