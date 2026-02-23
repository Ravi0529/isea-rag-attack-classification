from __future__ import annotations
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.rag.qdrant_client import get_qdrant
from src.rag.embeddings import load_embedder, embed_texts


def search_attack_for_text(
    query_text: str,
    top_k: int = 10,
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "auto",
):
    model = load_embedder(embed_model, device=device)
    q = embed_texts(model, [query_text])[0]

    client = get_qdrant()
    res = client.query_points(
        collection_name="mitre_attack",
        query=q.tolist(),
        limit=top_k,
        with_payload=True,
    )
    hits = res.points
    return hits


def search_sessions(
    query_text: str,
    top_k: int = 10,
    label: str | None = None,
    tool: str | None = None,
    embed_model: str = "sentence-transformers/all-mpnet-base-v2",
    device: str = "auto",
):
    model = load_embedder(embed_model, device=device)
    q = embed_texts(model, [query_text])[0]

    must = []
    if label:
        must.append(FieldCondition(key="label", match=MatchValue(value=label)))
    if tool:
        must.append(FieldCondition(key="tool", match=MatchValue(value=tool)))

    flt = Filter(must=must) if must else None

    client = get_qdrant()
    res = client.query_points(
        collection_name="log_sessions",
        query=q.tolist(),
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    )
    hits = res.points
    return hits
