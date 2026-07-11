"""
Local RAG Assistant — Streamlit web UI.

Run with:
    streamlit run app.py
"""

import streamlit as st
from database import init_db, count_chunks, list_sources
from llm_client import answer_query

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="centered",
)

# ── Initialise DB (no-op if already exists) ───────────────────────────────────

init_db()

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Hide Streamlit deploy button — this is a local-only project */
    [data-testid="stAppDeployButton"] { display: none !important; }
    [data-testid="stToolbar"]         { display: none !important; }

    .block-container { max-width: 860px; }

    .source-tag {
        display: inline-block;
        background: #0078d4;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin-right: 4px;
    }

    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 8px 0;
        font-size: 0.78rem;
        color: #888;
        background: var(--background-color);
        border-top: 1px solid #e0e0e0;
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("📚 Local RAG Assistant")
st.caption("Offline Q&A powered by **Microsoft Foundry Local** · RAG · SQLite · On-device model")
st.divider()

# ── Sidebar — knowledge-base stats ───────────────────────────────────────────

with st.sidebar:
    st.header("Knowledge Base")

    total = count_chunks()
    sources = list_sources()

    if total == 0:
        st.warning("No documents indexed yet.\nRun `python ingest.py` to load your docs.")
    else:
        st.success(f"{total} chunks indexed")
        st.markdown(f"**{len(sources)} document(s):**")
        for s in sources:
            st.markdown(f"- {s}")

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Your question is embedded locally\n"
        "2. The top matching chunks are retrieved from SQLite\n"
        "3. Chunks + question sent to Foundry Local\n"
        "4. Answer generated fully offline"
    )

# ── Conversation history (session state) ─────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown(
                " ".join(
                    f'<span class="source-tag">📄 {s}</span>'
                    for s in msg["sources"]
                ),
                unsafe_allow_html=True,
            )

# ── Chat input ────────────────────────────────────────────────────────────────

if total == 0:
    st.info(
        "The knowledge base is empty. "
        "Add `.txt` or `.md` files to the `docs/` folder and run `python ingest.py`."
    )
    st.stop()

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating answer..."):
            result = answer_query(question)

        answer  = result["answer"]
        sources = result["sources"]
        chunks  = result["chunks"]

        st.markdown(answer)

        if sources:
            st.markdown(
                " ".join(
                    f'<span class="source-tag">📄 {s}</span>'
                    for s in sources
                ),
                unsafe_allow_html=True,
            )

        if chunks:
            with st.expander("Retrieved context chunks"):
                for c in chunks:
                    score_pct = f"{c['score'] * 100:.1f}%"
                    st.markdown(
                        f"**{c['source']}** — similarity: `{score_pct}`\n\n"
                        f"```\n{c['content'][:400]}{'...' if len(c['content']) > 400 else ''}\n```"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">Made by Enes Erdogan</div>',
    unsafe_allow_html=True,
)
