from __future__ import annotations
import json
import numpy as np
from qdrant_client.models import PointStruct

from src.rag.qdrant_client import get_qdrant, ensure_collection
from src.rag.embeddings import load_embedder, embed_texts


def _tech_to_text(t: dict, mitigations: list[dict]) -> str:
    name = t.get("name", "")
    tid = t.get("technique_id", "")
    desc = (t.get("description", "") or "").replace("\n", " ")
    tactics = ", ".join(t.get("tactics", []) or [])
    mit_names = "; ".join([m.get("name", "") for m in mitigations[:8]])
    return f"{tid} {name}. Tactics: {tactics}. Description: {desc} Mitigations: {mit_names}"


def index_mitre_attack(
    cache_path: str = "data/attack/attack_stix_cache.json",
    collection: str = "mitre_attack",
    embed_model: str = "sentence-transformers/all-roberta-large-v1",
    device: str = "auto",
    batch_size: int = 128,
) -> None:
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    techniques = [t for t in cache["techniques"] if t.get("technique_id")]
    mit_map = cache.get("mitigations_by_technique", {})

    texts, payloads, ids = [], [], []
    for i, t in enumerate(techniques):
        tid = t["technique_id"]
        mits = mit_map.get(tid, [])
        texts.append(_tech_to_text(t, mits))
        payloads.append(
            {
                "technique_id": tid,
                "name": t.get("name"),
                "tactics": t.get("tactics", []),
                "is_subtechnique": bool(t.get("is_subtechnique", False)),
                "platforms": t.get("platforms", []),
                "mitigations": [m.get("name") for m in mits[:20]],
            }
        )
        ids.append(i)  # stable numeric ids

    model = load_embedder(embed_model, device=device)
    dim = int(model.get_sentence_embedding_dimension())

    client = get_qdrant()
    ensure_collection(client, collection, dim)

    # batch upsert
    for start in range(0, len(texts), batch_size):
        chunk_texts = texts[start : start + batch_size]
        vecs = embed_texts(model, chunk_texts, batch_size=min(256, batch_size))
        points = [
            PointStruct(
                id=ids[start + j], vector=vecs[j].tolist(), payload=payloads[start + j]
            )
            for j in range(len(chunk_texts))
        ]
        client.upsert(collection_name=collection, points=points)

    print(f"✅ indexed {len(texts)} techniques -> {collection}")
