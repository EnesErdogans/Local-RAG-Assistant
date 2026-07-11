"""
Document ingestion pipeline.

Usage:
    python ingest.py              # ingest docs/ folder (skip already-indexed files)
    python ingest.py --reset      # clear DB first, then ingest everything
    python ingest.py --dir <path> # ingest from a custom directory
"""

import os
import sys
import argparse

from config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from database import init_db, insert_chunk, clear_db, count_chunks, list_sources
from embeddings import get_embedding


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping fixed-size character chunks.
    Tries to break at sentence boundaries ('. ') when possible.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to end the chunk at the last sentence boundary within the window
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1   # include the '.'

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = end - overlap

    return chunks


# ---------------------------------------------------------------------------
# Single-file ingestion
# ---------------------------------------------------------------------------

def ingest_file(filepath: str, source_name: str | None = None) -> int:
    if source_name is None:
        source_name = os.path.basename(filepath)

    print(f"\n  [{source_name}]")
    with open(filepath, encoding="utf-8") as fh:
        text = fh.read()

    chunks = chunk_text(text)
    if not chunks:
        print("  -> No text found, skipping.")
        return 0

    print(f"  -> {len(chunks)} chunk(s)")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        insert_chunk(source_name, i, chunk, embedding)
        print(f"  -> Embedded chunk {i + 1}/{len(chunks)}", end="\r", flush=True)

    print(f"  -> Done ({len(chunks)} chunks stored)          ")
    return len(chunks)


# ---------------------------------------------------------------------------
# Bulk ingestion
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def ingest_all(docs_dir: str = DOCS_DIR, reset: bool = False) -> None:
    if not os.path.isdir(docs_dir):
        print(f"Error: directory not found - {docs_dir}")
        sys.exit(1)

    print("\n=== Local RAG Assistant - Ingestion Pipeline ===\n")
    init_db()

    if reset:
        print("Clearing existing knowledge base ...")
        clear_db()

    already_indexed = set(list_sources()) if not reset else set()
    files = sorted(
        f for f in os.listdir(docs_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No .txt / .md files found in: {docs_dir}")
        return

    print(f"Found {len(files)} document(s) in '{docs_dir}':")
    total_new = 0
    skipped = 0

    for filename in files:
        if filename in already_indexed:
            print(f"  [SKIP] {filename} (already indexed)")
            skipped += 1
            continue
        filepath = os.path.join(docs_dir, filename)
        total_new += ingest_file(filepath)

    print(f"\n{'=' * 48}")
    print(f"  Ingested : {total_new} new chunks")
    print(f"  Skipped  : {skipped} file(s) already in DB")
    print(f"  DB total : {count_chunks()} chunks")
    print("=" * 48)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG knowledge base.")
    parser.add_argument("--reset", action="store_true", help="Clear the DB before ingesting")
    parser.add_argument("--dir",   default=DOCS_DIR,   help="Directory containing documents")
    args = parser.parse_args()

    ingest_all(docs_dir=args.dir, reset=args.reset)
