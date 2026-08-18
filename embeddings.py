"""
Embedding generation via Foundry Local SDK + cosine similarity utilities.
"""

import numpy as np
from foundry_local_sdk.openai.embedding_client import EmbeddingClient

from config import EMBEDDING_MODEL
from foundry_manager import get_or_load_model

_emb_client: EmbeddingClient | None = None


def _get_client() -> EmbeddingClient:
    global _emb_client
    if _emb_client is None:
        print(f"[Embeddings] Initialising model '{EMBEDDING_MODEL}' …")
        model = get_or_load_model(EMBEDDING_MODEL)
        _emb_client = model.get_embedding_client()
        print("[Embeddings] Ready.")
    return _emb_client


def get_embedding(text: str) -> list[float]:
    """Return the embedding vector for a single piece of text."""
    client = _get_client()
    text = text.replace("\n", " ").strip()
    response = client.generate_embedding(text)
    return response.data[0].embedding


def warm_up_embedding_model() -> None:
    """Load the embedding model before the first user query."""
    get_embedding("warmup")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
