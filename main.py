"""
Local RAG Assistant — CLI entry point.

Run with:
    python main.py
"""

import sys
from database import init_db, count_chunks, list_sources
from llm_client import answer_query

BANNER = """
================================================================
  Local RAG Assistant  |  Microsoft Foundry Local
  Offline Q&A - powered by RAG + SQLite + on-device model
----------------------------------------------------------------
  Commands:  quit  |  sources  |  help
================================================================
"""


def print_help() -> None:
    print(
        "\nAvailable commands:\n"
        "  quit / exit  — exit the assistant\n"
        "  sources      — list all indexed documents\n"
        "  help         — show this help\n"
        "  <anything>   — ask a question\n"
    )


def run() -> None:
    init_db()
    print(BANNER)

    total = count_chunks()
    if total == 0:
        print(
            "  Warning: knowledge base is empty.\n"
            "  Run  python ingest.py  to index your documents first.\n"
        )
    else:
        sources = list_sources()
        print(f"  Knowledge base ready: {total} chunks from {len(sources)} document(s).\n")

    while True:
        try:
            raw = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if cmd == "sources":
            srcs = list_sources()
            if srcs:
                print("\n  Indexed documents:")
                for s in srcs:
                    print(f"    • {s}")
            else:
                print("  No documents indexed yet.")
            print()
            continue

        if cmd == "help":
            print_help()
            continue

        # --- RAG pipeline ---
        print()
        result = answer_query(raw)

        print(f"Assistant › {result['answer']}\n")
        if result["sources"]:
            print(f"  Sources: {', '.join(result['sources'])}")
        print()


if __name__ == "__main__":
    run()
