from __future__ import annotations
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder(
    model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "auto"
) -> SentenceTransformer:
    dev = resolve_device(device)
    print(f"🔧 Embeddings device: {dev}")
    return SentenceTransformer(model_name, device=dev)


def embed_texts(
    model: SentenceTransformer, texts: list[str], batch_size: int = 256
) -> np.ndarray:
    vecs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    return np.asarray(vecs, dtype=np.float32)
