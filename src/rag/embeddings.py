from __future__ import annotations
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device

    if torch.cuda.is_available():
        try:
            # Smoke test: can we actually allocate on GPU?
            _ = torch.zeros(1).cuda()
            return "cuda"
        except Exception as e:
            print(f"⚠️ CUDA detected but unusable, falling back to CPU: {e}")

    return "cpu"


def load_embedder(
    model_name: str = "sentence-transformers/all-roberta-large-v1",
    device: str = "auto",
) -> SentenceTransformer:
    dev = resolve_device(device)
    print(f"🔧 Embeddings device selected: {dev}")

    try:
        model = SentenceTransformer(model_name, device=dev)
        return model
    except Exception as e:
        print(f"❌ Failed to load model on {dev}: {e}")
        print("🔄 Retrying on CPU...")
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    try:
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("⚠️ CUDA OOM detected → switching to CPU")
            torch.cuda.empty_cache()
            model.to("cpu")
            vecs = model.encode(
                texts,
                batch_size=max(16, batch_size // 4),
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        else:
            raise

    return np.asarray(vecs, dtype=np.float32)
