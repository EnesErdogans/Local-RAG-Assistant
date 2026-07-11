"""
Local RAG Assistant — Streamlit web UI.
Run with:  streamlit run app.py
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

init_db()

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

/* ── Architecture SVG container ── */
.arch-wrap {
    background: #f8faff;
    border: 1px solid #dde4f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0 1.6rem;
    text-align: center;
}

/* ── Step cards (Retrieve / Augment / Generate) ── */
.steps-row {
    display: flex;
    gap: 14px;
    margin: 1rem 0 1.6rem;
    flex-wrap: wrap;
}
.step-card {
    flex: 1;
    min-width: 150px;
    border-radius: 10px;
    padding: 1.1rem 1rem;
    text-align: center;
    color: #fff;
}
.step-card h3 { margin: 0 0 .4rem; font-size: 1rem; }
.step-card p  { margin: 0; font-size: .82rem; line-height: 1.45; opacity: .92; }
.step-card .icon { font-size: 1.8rem; margin-bottom: .5rem; }
.step-r { background: linear-gradient(135deg,#0078d4,#005fa3); }
.step-a { background: linear-gradient(135deg,#107c10,#0a5c0a); }
.step-g { background: linear-gradient(135deg,#8764b8,#5c3d8f); }

/* ── Arrow between steps ── */
.arrow { display:flex; align-items:center; color:#0078d4;
         font-size:1.4rem; font-weight:bold; padding-top:.2rem; }

/* ── Tech grid ── */
.tech-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 14px;
    margin: 1rem 0 1.6rem;
}
.tech-card {
    background: #fff;
    border: 1.5px solid #dde4f0;
    border-radius: 10px;
    padding: 1rem;
    transition: box-shadow .2s;
}
.tech-card:hover { box-shadow: 0 4px 16px rgba(0,120,212,.15); }
.tech-card .tc-icon { font-size: 1.6rem; margin-bottom: .45rem; }
.tech-card h4 { margin: 0 0 .3rem; color: #0078d4; font-size: .95rem; }
.tech-card p  { margin: 0; font-size: .8rem; color: #555; line-height: 1.45; }

/* ── Section label ── */
.sec-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #0078d4;
    margin-bottom: .3rem;
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

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_chat, tab_how = st.tabs(["💬  Chat", "📖  How It Works"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ════════════════════════════════════════════════════════════════════

with tab_chat:
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

# ════════════════════════════════════════════════════════════════════
# TAB 2 — HOW IT WORKS
# ════════════════════════════════════════════════════════════════════

with tab_how:

    st.markdown("### How the Local RAG Assistant Works")
    st.caption("A fully offline pipeline — no internet, no cloud, no data leaving your device.")
    st.divider()

    # ── Architecture diagram ─────────────────────────────────────────
    st.markdown('<p class="sec-label">System Architecture</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="arch-wrap">
<svg viewBox="0 0 820 320" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:820px;font-family:Segoe UI,Arial,sans-serif;">

  <!-- ── USER ── -->
  <rect x="20" y="125" width="110" height="70" rx="10"
        fill="#e8f1fb" stroke="#0078d4" stroke-width="1.8"/>
  <text x="75" y="153" text-anchor="middle" font-size="11" font-weight="600" fill="#0078d4">👤 User</text>
  <text x="75" y="170" text-anchor="middle" font-size="10" fill="#444">Question</text>
  <text x="75" y="184" text-anchor="middle" font-size="10" fill="#444">/ Answer</text>

  <!-- arrow user → app -->
  <line x1="130" y1="160" x2="178" y2="160" stroke="#0078d4" stroke-width="1.8" marker-end="url(#arr)"/>

  <!-- ── LOCAL RAG APP ── -->
  <rect x="180" y="105" width="145" height="110" rx="10"
        fill="#fff" stroke="#0078d4" stroke-width="2"/>
  <text x="252" y="130" text-anchor="middle" font-size="11" font-weight="700" fill="#0078d4">⚙️ Local RAG App</text>
  <line x1="195" y1="138" x2="310" y2="138" stroke="#dde4f0" stroke-width="1"/>
  <text x="252" y="155" text-anchor="middle" font-size="10" fill="#555">server + pipeline</text>
  <text x="252" y="170" text-anchor="middle" font-size="10" fill="#555">query embedding</text>
  <text x="252" y="185" text-anchor="middle" font-size="10" fill="#555">context assembly</text>

  <!-- arrow app → sqlite -->
  <line x1="325" y1="140" x2="375" y2="100" stroke="#107c10" stroke-width="1.8" marker-end="url(#arr-g)"/>

  <!-- ── SQLITE ── -->
  <rect x="377" y="40" width="145" height="110" rx="10"
        fill="#f0faf0" stroke="#107c10" stroke-width="1.8"/>
  <text x="450" y="66" text-anchor="middle" font-size="11" font-weight="700" fill="#107c10">🗄️ SQLite</text>
  <line x1="392" y1="74" x2="507" y2="74" stroke="#c3e6c3" stroke-width="1"/>
  <text x="450" y="91" text-anchor="middle" font-size="10" fill="#555">Vector Database</text>
  <text x="450" y="106" text-anchor="middle" font-size="10" fill="#555">Document Embeddings</text>
  <text x="450" y="121" text-anchor="middle" font-size="10" fill="#555">Cosine Similarity</text>
  <text x="450" y="136" text-anchor="middle" font-size="10" fill="#555">Search</text>

  <!-- arrow sqlite → app (retrieved chunks) -->
  <line x1="377" y1="165" x2="327" y2="185" stroke="#107c10" stroke-width="1.8" stroke-dasharray="5,3" marker-end="url(#arr-g)"/>
  <text x="345" y="200" text-anchor="middle" font-size="9" fill="#107c10" font-style="italic">retrieved chunks</text>

  <!-- arrow app → foundry -->
  <line x1="325" y1="185" x2="375" y2="230" stroke="#8764b8" stroke-width="1.8" marker-end="url(#arr-p)"/>

  <!-- ── FOUNDRY LOCAL ── -->
  <rect x="377" y="175" width="145" height="110" rx="10"
        fill="#f5f0fc" stroke="#8764b8" stroke-width="1.8"/>
  <text x="450" y="201" text-anchor="middle" font-size="11" font-weight="700" fill="#8764b8">🧠 Foundry Local</text>
  <line x1="392" y1="209" x2="507" y2="209" stroke="#d9c8f0" stroke-width="1"/>
  <text x="450" y="226" text-anchor="middle" font-size="10" fill="#555">On-device LLM</text>
  <text x="450" y="241" text-anchor="middle" font-size="10" fill="#555">phi-3.5-mini</text>
  <text x="450" y="256" text-anchor="middle" font-size="10" fill="#555">CPU / NPU</text>
  <text x="450" y="271" text-anchor="middle" font-size="10" fill="#555">Zero network calls</text>

  <!-- arrow foundry → app (answer) -->
  <line x1="377" y1="230" x2="327" y2="215" stroke="#8764b8" stroke-width="1.8" stroke-dasharray="5,3" marker-end="url(#arr-p)"/>
  <text x="348" y="230" text-anchor="middle" font-size="9" fill="#8764b8" font-style="italic">answer</text>

  <!-- ── EMBEDDING MODEL box (bottom) ── -->
  <rect x="570" y="110" width="145" height="100" rx="10"
        fill="#fff8ec" stroke="#c8850a" stroke-width="1.8"/>
  <text x="643" y="136" text-anchor="middle" font-size="11" font-weight="700" fill="#c8850a">🔢 Embeddings</text>
  <line x1="585" y1="144" x2="700" y2="144" stroke="#f5d88a" stroke-width="1"/>
  <text x="643" y="161" text-anchor="middle" font-size="10" fill="#555">qwen3-embedding</text>
  <text x="643" y="176" text-anchor="middle" font-size="10" fill="#555">-0.6b</text>
  <text x="643" y="191" text-anchor="middle" font-size="10" fill="#555">1536-dim vectors</text>

  <!-- arrow app ↔ embedding -->
  <line x1="522" y1="155" x2="568" y2="155" stroke="#c8850a" stroke-width="1.5" marker-end="url(#arr-o)"/>
  <text x="545" y="148" text-anchor="middle" font-size="9" fill="#c8850a">embed</text>

  <!-- ── OFFLINE BADGE ── -->
  <rect x="640" y="268" width="150" height="32" rx="16"
        fill="#d4edda" stroke="#107c10" stroke-width="1.2"/>
  <text x="715" y="289" text-anchor="middle" font-size="11" font-weight="700" fill="#107c10">✅ 100% Offline</text>

  <!-- arrowhead markers -->
  <defs>
    <marker id="arr"   markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#0078d4"/>
    </marker>
    <marker id="arr-g" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#107c10"/>
    </marker>
    <marker id="arr-p" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#8764b8"/>
    </marker>
    <marker id="arr-o" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#c8850a"/>
    </marker>
  </defs>
</svg>
</div>
""", unsafe_allow_html=True)

    # ── RAG Pipeline steps ───────────────────────────────────────────
    st.markdown('<p class="sec-label">The RAG Pipeline — 3 Steps</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="steps-row">
  <div class="step-card step-r">
    <div class="icon">🔍</div>
    <h3>1 · Retrieve</h3>
    <p>The user's question is converted to an embedding vector. Cosine similarity is computed against every stored chunk in SQLite. The top-3 most relevant passages are selected.</p>
  </div>
  <div class="arrow">→</div>
  <div class="step-card step-a">
    <div class="icon">📝</div>
    <h3>2 · Augment</h3>
    <p>The retrieved passages are injected into the prompt as context alongside the original question. The model is instructed to answer only from this context.</p>
  </div>
  <div class="arrow">→</div>
  <div class="step-card step-g">
    <div class="icon">💡</div>
    <h3>3 · Generate</h3>
    <p>Foundry Local runs phi-3.5-mini on-device. The model reads the context + question and produces a grounded, factual answer with source citations.</p>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Technology cards ─────────────────────────────────────────────
    st.markdown('<p class="sec-label">Key Technologies</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="tech-grid">

  <div class="tech-card">
    <div class="tc-icon">🏭</div>
    <h4>Microsoft Foundry Local</h4>
    <p>End-to-end on-device LLM runtime. No cloud account or GPU required. Runs models using CPU/NPU acceleration — fully offline.</p>
  </div>

  <div class="tech-card">
    <div class="tc-icon">🔢</div>
    <h4>Text Embeddings</h4>
    <p>qwen3-embedding-0.6b converts text to 1536-dimensional vectors. Similar meanings → similar vectors, enabling semantic search beyond keyword matching.</p>
  </div>

  <div class="tech-card">
    <div class="tc-icon">🗄️</div>
    <h4>SQLite</h4>
    <p>Serverless, single-file database storing document chunks and their embedding vectors as JSON. No setup required — built into Python.</p>
  </div>

  <div class="tech-card">
    <div class="tc-icon">🧠</div>
    <h4>phi-3.5-mini (3.8B)</h4>
    <p>Microsoft's compact language model. Fast inference on consumer hardware. Constrained to context-only answers to prevent hallucination.</p>
  </div>

  <div class="tech-card">
    <div class="tc-icon">📐</div>
    <h4>Cosine Similarity</h4>
    <p>Measures the angle between two embedding vectors. Score of 1.0 = identical meaning. Used to rank retrieved chunks by relevance.</p>
  </div>

  <div class="tech-card">
    <div class="tc-icon">✂️</div>
    <h4>Chunking Strategy</h4>
    <p>Documents are split at sentence boundaries into ~500-character overlapping passages. Overlap prevents context loss at chunk edges.</p>
  </div>

</div>
""", unsafe_allow_html=True)

    # ── Data flow detail ─────────────────────────────────────────────
    st.markdown('<p class="sec-label">Ingestion vs Query — Two Phases</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="tech-card" style="height:100%">
  <div class="tc-icon">📥</div>
  <h4>Ingestion Phase <span style="font-weight:400;color:#888">(run once)</span></h4>
  <ol style="font-size:.82rem;color:#444;line-height:1.7;padding-left:1.1rem;margin:.5rem 0 0">
    <li>Read <code>.txt</code> / <code>.md</code> files from <code>docs/</code></li>
    <li>Split each file into overlapping chunks</li>
    <li>Embed every chunk with <em>qwen3-embedding-0.6b</em></li>
    <li>Store (text, embedding) pairs in <em>SQLite</em></li>
  </ol>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="tech-card" style="height:100%">
  <div class="tc-icon">💬</div>
  <h4>Query Phase <span style="font-weight:400;color:#888">(per question)</span></h4>
  <ol style="font-size:.82rem;color:#444;line-height:1.7;padding-left:1.1rem;margin:.5rem 0 0">
    <li>Embed the user's question</li>
    <li>Compute cosine similarity vs all stored chunks</li>
    <li>Pick top-3 most relevant passages</li>
    <li>Build prompt: system + context + question</li>
    <li><em>phi-3.5-mini</em> generates the final answer</li>
  </ol>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Why RAG? ─────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">Why RAG over plain LLM?</p>', unsafe_allow_html=True)

    st.markdown("""
<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:1.4rem;">

  <div class="tech-card" style="flex:1;min-width:160px;border-color:#d4edda;background:#f6fdf6">
    <div class="tc-icon">✅</div>
    <h4 style="color:#107c10">With RAG</h4>
    <ul style="font-size:.8rem;color:#444;line-height:1.6;padding-left:1rem;margin:.4rem 0 0">
      <li>Answers grounded in your documents</li>
      <li>Source citations provided</li>
      <li>Handles domain-specific knowledge</li>
      <li>Fewer hallucinations</li>
      <li>No retraining required</li>
    </ul>
  </div>

  <div class="tech-card" style="flex:1;min-width:160px;border-color:#f8d7da;background:#fff8f8">
    <div class="tc-icon">❌</div>
    <h4 style="color:#c0392b">Plain LLM only</h4>
    <ul style="font-size:.8rem;color:#444;line-height:1.6;padding-left:1rem;margin:.4rem 0 0">
      <li>Limited to training-time knowledge</li>
      <li>No source references</li>
      <li>Cannot access custom documents</li>
      <li>Prone to hallucination</li>
      <li>Requires fine-tuning to specialise</li>
    </ul>
  </div>

</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown('<div class="footer">Made by Enes Erdogan</div>', unsafe_allow_html=True)
