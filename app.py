"""
Local RAG Assistant — Streamlit web UI.
Run with:  streamlit run app.py
"""

import streamlit as st
from database import init_db, count_chunks, list_sources
from embeddings import warm_up_embedding_model
from llm_client import answer_query, warm_up_chat_model

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="📚",
    layout="centered",
)

init_db()


@st.cache_resource(show_spinner=False)
def warm_up_models() -> bool:
    warm_up_embedding_model()
    warm_up_chat_model()
    return True

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Hide Streamlit chrome — local-only project, no deploy needed */
[data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stToolbar"]         { display: none !important; }

.block-container { max-width: 900px; padding-top: 1.5rem; }

/* ── Source tag ── */
.source-tag {
    display: inline-block;
    background: #0078d4;
    color: #fff;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: .78rem;
    margin-right: 4px;
}

/* ── Footer ── */
.footer {
    position: fixed; bottom: 0; left: 0; width: 100%;
    text-align: center; padding: 7px 0;
    font-size: .76rem; color: #888;
    background: var(--background-color);
    border-top: 1px solid #e0e0e0;
    z-index: 999;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

total   = count_chunks()
sources = list_sources()

with st.sidebar:
    st.markdown("## 📚 Knowledge Base")
    if total == 0:
        st.warning("No documents indexed yet.\nRun `python ingest.py` to load your docs.")
    else:
        st.success(f"{total} chunks indexed")
        st.markdown(f"**{len(sources)} document(s):**")
        for s in sources:
            st.markdown(f"- {s}")
    st.divider()
    st.markdown(
        "**Quick start**\n"
        "```\n"
        "python ingest.py\n"
        "streamlit run app.py\n"
        "```"
    )

# ── Chat ──────────────────────────────────────────────────────────────────────

st.markdown("### Local RAG Assistant")
st.caption("Offline Q&A powered by **Microsoft Foundry Local** · RAG · SQLite · On-device model")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            st.markdown(
                " ".join(f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"]),
                unsafe_allow_html=True,
            )

if total == 0:
    st.info("The knowledge base is empty. Add `.txt` or `.md` files to `docs/` and run `python ingest.py`.")
    st.stop()

with st.spinner("Preparing local models..."):
    try:
        warm_up_models()
    except Exception as exc:
        st.error(f"Local models could not be prepared: {exc}")
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
                " ".join(f'<span class="source-tag">📄 {s}</span>' for s in sources),
                unsafe_allow_html=True,
            )

        if chunks:
            with st.expander("Retrieved context chunks"):
                for c in chunks:
                    st.markdown(
                        f"**{c['source']}** — similarity: `{c['score']*100:.1f}%`\n\n"
                        f"```\n{c['content'][:400]}{'...' if len(c['content'])>400 else ''}\n```"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown('<div class="footer">Made by Enes Erdogan</div>', unsafe_allow_html=True)
