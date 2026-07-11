# Local RAG AI Assistant
**Offline Q&A powered by Microsoft Foundry Local · RAG · SQLite · Python**

> Built as part of the Microsoft Summer School Foundry Local one-month programme.

---

## What this is

A fully **offline** document Q&A assistant. You load your own `.txt` / `.md` files, the app indexes them locally (embedding each chunk and storing vectors in SQLite), and then you can ask questions in natural language. A local LLM (running via Microsoft Foundry Local) generates grounded answers — no internet connection needed, no cloud account required.

Architecture:

```
User Question
     │
     ▼
[ Streamlit / CLI ]
     │
     ▼
[ Embedding Model ]  ←── Foundry Local (qwen3-embedding-0.6b)
     │
     ▼
[ SQLite Vector DB ]  ──── cosine similarity search ────► Top-K chunks
     │                                                         │
     └─────────────────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                               [ Chat LLM ]  ←── Foundry Local (phi-3.5-mini)
                                                               │
                                                               ▼
                                                          Answer + Sources
```

---

## Quick start

### 1. Install Foundry Local

Download and install Foundry Local from the official Microsoft page, then verify:
```
foundry-local --version
```

### 2. Clone / open this project

```
cd local-rag-assistant
```

### 3. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Add your documents

Place `.txt` or `.md` files in the `docs/` folder. Six sample documents are already included covering RAG, Foundry Local, embeddings, SQLite, prompt engineering, Python basics, and AI fundamentals.

### 5. Ingest documents into the knowledge base

```bash
python ingest.py
```

Options:
```bash
python ingest.py --reset          # clear existing DB and re-ingest all
python ingest.py --dir /my/docs   # ingest from a custom directory
```

### 6. Run the assistant

**Streamlit web UI (recommended):**
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

**Command-line interface:**
```bash
python main.py
```

---

## Project structure

```
├── config.py          # Model names, paths, chunk settings, system prompt
├── database.py        # SQLite: init, insert, query, count, clear
├── embeddings.py      # Foundry Local embedding model + cosine similarity
├── retrieval.py       # get_top_chunks() — semantic vector search
├── llm_client.py      # answer_query() — full RAG pipeline (retrieve + generate)
├── ingest.py          # Document ingestion script
├── main.py            # CLI interactive loop
├── app.py             # Streamlit web UI
├── test_rag.py        # Functional test suite
├── requirements.txt
├── knowledge_base.db  # Created automatically on first ingest
└── docs/
    ├── rag_explained.txt
    ├── foundry_local_guide.txt
    ├── embeddings_and_vector_search.txt
    ├── sqlite_guide.txt
    ├── prompt_engineering.txt
    ├── python_basics.txt
    └── ai_fundamentals.txt
```

---

## Running tests

After ingesting documents:
```bash
python test_rag.py
```

The test suite checks:
- That in-scope questions receive relevant answers (with keyword verification)
- That out-of-scope questions are gracefully declined

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `qwen3-embedding-0.6b` | Foundry Local embedding model alias |
| `CHAT_MODEL` | `phi-3.5-mini` | Foundry Local chat model alias |
| `CHUNK_SIZE` | `500` | Max characters per document chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between consecutive chunks |
| `TOP_K_CHUNKS` | `3` | Number of chunks retrieved per query |
| `MAX_TOKENS` | `512` | Max tokens in the LLM response |
| `TEMPERATURE` | `0.1` | LLM temperature (low = more factual) |

---

## Design decisions

- **SQLite over a vector DB**: For a small knowledge base (< 10 000 chunks), brute-force cosine similarity in numpy is fast enough (< 100 ms). No external service to install or manage.
- **JSON-serialised embeddings**: Embeddings are stored as JSON strings in a TEXT column. Simple, portable, and readable.
- **Sentence-boundary chunking**: The ingestion pipeline tries to break chunks at sentence boundaries (`. `) rather than cutting mid-word, improving retrieval quality.
- **Lazy model loading**: Foundry Local models are loaded on first use, not at import time, so the CLI starts instantly even if the models are large.
- **Low temperature (0.1)**: For factual Q&A, we want deterministic, consistent answers, not creative variation.

---

## Limitations

- Vector search is brute-force O(N). For very large knowledge bases, replace with FAISS or Chroma.
- Only `.txt` and `.md` files are supported out of the box. Add PDF support with `pymupdf` if needed.
- The quality of answers depends heavily on the chosen chat model and chunk size. Experiment with `config.py` settings.

---

## References

- [Microsoft Tech Community: Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968)
- [Microsoft Learn: Get started with Foundry Local](https://learn.microsoft.com/foundry-local)
- [Microsoft Learn: Tutorial — Build a RAG application](https://learn.microsoft.com/azure/ai-studio/tutorials/rag-studio)
- [SQLite documentation](https://sqlite.org/docs.html)
