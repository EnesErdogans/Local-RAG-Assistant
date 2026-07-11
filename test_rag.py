"""
Functional test suite for the Local RAG Assistant.
Run after ingesting documents:
    python ingest.py
    python test_rag.py
"""

import sys
from database import init_db, count_chunks
from retrieval import get_top_chunks
from llm_client import answer_query

# ── Test cases ────────────────────────────────────────────────────────────────
# Each tuple: (question, expected_keywords_in_answer, should_have_answer)
TEST_CASES = [
    # Questions that SHOULD be answered from the docs
    (
        "What are the three steps of RAG?",
        ["retrieve", "augment", "generate"],
        True,
    ),
    (
        "What is Foundry Local?",
        ["foundry", "local", "offline"],
        True,
    ),
    (
        "How is cosine similarity calculated?",
        ["dot", "norm"],
        True,
    ),
    (
        "What is a text embedding?",
        ["vector", "semantic"],
        True,
    ),
    (
        "How do you store embeddings in SQLite?",
        ["json", "text"],
        True,
    ),
    (
        "What temperature should I use for factual Q&A?",
        ["0", "low"],
        True,
    ),
    (
        "What is hallucination in LLMs?",
        ["hallucin", "incorrect"],
        True,
    ),
    # Questions that should NOT be answered (out-of-scope)
    (
        "What is the boiling point of water?",
        [],
        False,
    ),
    (
        "Who won the 2024 football World Cup?",
        [],
        False,
    ),
]


def run_tests() -> None:
    init_db()

    total = count_chunks()
    if total == 0:
        print("ERROR: Knowledge base is empty. Run 'python ingest.py' first.")
        sys.exit(1)

    print(f"\n=== RAG Assistant — Functional Tests ({len(TEST_CASES)} cases) ===\n")
    passed = 0
    failed = 0

    for i, (question, keywords, should_answer) in enumerate(TEST_CASES, 1):
        print(f"[{i:02d}] {question}")

        result = answer_query(question)
        answer = result["answer"].lower()

        if should_answer:
            # Check that the model produced an answer (not a "don't know" response)
            refused = "don't have" in answer or "i don't know" in answer or "no information" in answer
            if refused:
                print(f"     FAIL — expected an answer but model said it doesn't know")
                print(f"     Answer: {result['answer'][:120]}")
                failed += 1
            elif keywords and not any(kw.lower() in answer for kw in keywords):
                print(f"     FAIL — answer did not contain any of: {keywords}")
                print(f"     Answer: {result['answer'][:120]}")
                failed += 1
            else:
                print(f"     PASS — sources: {', '.join(result['sources'])}")
                passed += 1
        else:
            # Out-of-scope: model should decline to answer
            refused = (
                "don't have" in answer
                or "i don't know" in answer
                or "no information" in answer
                or "not" in answer
            )
            if refused:
                print(f"     PASS — correctly declined to answer out-of-scope question")
                passed += 1
            else:
                print(f"     WARN — model answered an out-of-scope question (check system prompt)")
                print(f"     Answer: {result['answer'][:120]}")
                failed += 1

        print()

    print("=" * 52)
    print(f"  Results: {passed} passed / {failed} failed / {len(TEST_CASES)} total")
    print("=" * 52)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
