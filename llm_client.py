"""
LLM chat integration via Foundry Local SDK.
"""

from foundry_local_sdk.openai.chat_client import ChatClient
from foundry_local_sdk.exception import FoundryLocalException

from config import CHAT_MODEL, MAX_TOKENS, TEMPERATURE, SYSTEM_PROMPT
from foundry_manager import get_or_load_model
from retrieval import get_top_chunks, format_context

_chat_client: ChatClient | None = None
MODEL_CANCELLED_ANSWER = (
    "The local model cancelled the request before finishing. "
    "Please wait a moment and try again. If this happens on the first question, "
    "the model may still be loading."
)


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


def warm_up_chat_model() -> None:
    """Load the chat model before the first user query."""
    _get_client()


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
    try:
        response = client.complete_chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ])
    except FoundryLocalException as exc:
        if "cancelled" in str(exc).lower():
            return {
                "answer": MODEL_CANCELLED_ANSWER,
                "sources": [],
                "chunks": chunks,
            }
        raise

    answer = response.choices[0].message.content.strip()
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {"answer": answer, "sources": sources, "chunks": chunks}
