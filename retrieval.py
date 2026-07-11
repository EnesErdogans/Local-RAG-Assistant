"""
Retrieve the most relevant document chunks for a user query.
"""

from embeddings import get_embedding, cosine_similarity
from database import get_all_chunks
from config import TOP_K_CHUNKS


def get_top_chunks(query: str, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    """
    Embed the query, compute cosine similarity against every stored chunk,
    and return the top-k most relevant chunks as a list of dicts:
      { source, content, score }
    """
    query_embedding = get_embedding(query)
    chunks = get_all_chunks()

    if not chunks:
        return []

    scored = [
        {
            "source": chunk["source"],
            "content": chunk["content"],
            "score": cosine_similarity(query_embedding, chunk["embedding"]),
        }
        for chunk in chunks
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_context(chunks: list[dict]) -> str:
    """Build the context string that is injected into the LLM prompt."""
    parts = [
        f"[Source: {c['source']}]\n{c['content']}"
        for c in chunks
    ]
    return "\n\n---\n\n".join(parts)
