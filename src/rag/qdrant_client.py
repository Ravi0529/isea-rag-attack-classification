from __future__ import annotations
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def get_qdrant() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"✅ created collection: {name} (dim={dim})")
