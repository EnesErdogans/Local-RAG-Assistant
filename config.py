import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
DB_PATH = os.path.join(BASE_DIR, "knowledge_base.db")

# Foundry Local model aliases
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "phi-3.5-mini"

# Chunking
CHUNK_SIZE = 500        # max characters per chunk
CHUNK_OVERLAP = 80      # overlap between consecutive chunks

# Retrieval
TOP_K_CHUNKS = 3        # number of top chunks to retrieve per query

# LLM generation
MAX_TOKENS = 256
TEMPERATURE = 0.1       # low temperature for factual Q&A

SYSTEM_PROMPT = (
    "You are a precise Q&A assistant. "
    "Answer questions using ONLY the context provided below. "
    "Always mention which source document the information comes from. "
    "If the context does not contain enough information to answer, "
    "respond with: \"I don't have that information in my knowledge base.\" "
    "Do not use outside knowledge. Be concise and factual."
)
