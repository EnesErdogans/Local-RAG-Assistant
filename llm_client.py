"""
LLM chat integration via Foundry Local SDK.
"""

from foundry_local_sdk.openai.chat_client import ChatClient

from config import CHAT_MODEL, MAX_TOKENS, TEMPERATURE, SYSTEM_PROMPT
from foundry_manager import get_or_load_model
from retrieval import get_top_chunks, format_context

_chat_client: ChatClient | None = None


def _get_client() -> ChatClient:
    global _chat_client
    if _chat_client is None:
        print(f"[LLM] Initialising model '{CHAT_MODEL}' …")
        model = get_or_load_model(CHAT_MODEL)
        _chat_client = model.get_chat_client()
        _chat_client.settings.temperature = TEMPERATURE
        _chat_client.settings.max_tokens = MAX_TOKENS
        print("[LLM] Ready.")
    return _chat_client


def answer_query(question: str) -> dict:
    """
    Full RAG pipeline for one question.
    Returns: { answer: str, sources: list[str], chunks: list[dict] }
    """
    chunks = get_top_chunks(question)

    if not chunks:
        return {
            "answer": (
                "The knowledge base is empty. "
                "Run  python ingest.py  to index your documents first."
            ),
            "sources": [],
            "chunks": [],
        }

    context = format_context(chunks)
    user_message = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    client = _get_client()
    response = client.complete_chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ])

    answer = response.choices[0].message.content.strip()
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {"answer": answer, "sources": sources, "chunks": chunks}
