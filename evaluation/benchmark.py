"""
Retrieval evaluation benchmark.

Compares Graph, Vector, TF-IDF, BM25, Ensemble,
and Hybrid methods on test queries using
Recall@K, Precision@K, MRR, and answer quality.

Usage:
  python -m evaluation.benchmark
  python -m evaluation.benchmark --llm=openai
"""

import sys
import time

from retrieval.llm import get_llm
from retrieval.retriever import Retriever


TEST_QUERIES = [
    {
        "question": (
            "What are people's experiences "
            "with RAG?"
        ),
        "expected_topics": [
            "rag", "retrieval", "augmented"
        ],
        "expected_entities": [
            "rag", "vector", "embedding"
        ],
        "type": "semantic",
    },
    {
        "question": (
            "Who are the most influential "
            "voices discussing AI safety?"
        ),
        "expected_topics": [
            "safety", "alignment", "guardrail"
        ],
        "expected_entities": [
            "safety", "anthropic", "openai"
        ],
        "type": "graph",
    },
    {
        "question": (
            "What is the best open source "
            "LLM in 2026?"
        ),
        "expected_topics": [
            "open source", "local", "llm"
        ],
        "expected_entities": [
            "llama", "mistral", "qwen",
            "ollama",
        ],
        "type": "semantic",
    },
    {
        "question": (
            "How has sentiment about RAG "
            "changed over time?"
        ),
        "expected_topics": [
            "rag", "sentiment", "changed"
        ],
        "expected_entities": ["rag"],
        "type": "temporal",
    },
    {
        "question": (
            "Which companies are leading "
            "open-source LLM development?"
        ),
        "expected_topics": [
            "open source", "company"
        ],
        "expected_entities": [
            "meta", "google", "mistral",
        ],
        "type": "graph",
    },
    {
        "question": (
            "What do people think about "
            "Claude vs GPT-4o?"
        ),
        "expected_topics": [
            "claude", "gpt", "comparison"
        ],
        "expected_entities": [
            "claude", "gpt", "openai",
            "anthropic",
        ],
        "type": "semantic",
    },
    {
        "question": (
            "What local LLM setup are people "
            "using for coding in 2026?"
        ),
        "expected_topics": [
            "local", "coding", "code"
        ],
        "expected_entities": [
            "code", "copilot", "local"
        ],
        "type": "semantic",
    },
    {
        "question": (
            "What topics are discussed "
            "alongside vector databases?"
        ),
        "expected_topics": [
            "vector", "database", "embedding"
        ],
        "expected_entities": [
            "chroma", "pinecone", "qdrant",
            "vector",
        ],
        "type": "graph",
    },
    {
        "question": (
            "How have opinions on agentic AI "
            "evolved from 2025 to 2026?"
        ),
        "expected_topics": [
            "agent", "agentic", "tool"
        ],
        "expected_entities": [
            "agent", "agentic"
        ],
        "type": "temporal",
    },
    {
        "question": (
            "What are the best practices "
            "for fine-tuning LLMs?"
        ),
        "expected_topics": [
            "fine-tun", "lora", "training"
        ],
        "expected_entities": [
            "fine-tun", "lora", "qlora"
        ],
        "type": "semantic",
    },
    {
        "question": (
            "Which subreddits are most active "
            "in discussing AI models?"
        ),
        "expected_topics": [
            "subreddit", "community"
        ],
        "expected_entities": [
            "locallama", "openai", "claude"
        ],
        "type": "graph",
    },
    {
        "question": (
            "What emerging AI safety concerns "
            "appeared in 2026?"
        ),
        "expected_topics": [
            "safety", "concern", "risk"
        ],
        "expected_entities": [
            "safety", "alignment"
        ],
        "type": "temporal",
    },
    {
        "question": (
            "Can you compensate for weak LLMs "
            "with RAG?"
        ),
        "expected_topics": [
            "rag", "weak", "compensat"
        ],
        "expected_entities": [
            "rag", "retrieval"
        ],
        "type": "semantic",
    },
    {
        "question": (
            "What do people say about "
            "Gemini's strengths and "
            "weaknesses?"
        ),
        "expected_topics": [
            "gemini", "strength", "weakness"
        ],
        "expected_entities": [
            "gemini", "google"
        ],
        "type": "semantic",
    },
    {
        "question": (
            "How has the discussion around "
            "embeddings changed over time?"
        ),
        "expected_topics": [
            "embedding", "vector"
        ],
        "expected_entities": [
            "embedding", "sentence"
        ],
        "type": "temporal",
    },
]

METHODS_TO_TEST = [
    "vector", "tfidf", "bm25",
    "ensemble", "hybrid",
]

K = 5


def compute_recall_at_k(
    docs, expected_keywords, k
):
    top_k_text = " ".join(
        d.text.lower() for d in docs[:k]
    )
    hits = sum(
        1 for kw in expected_keywords
        if kw.lower() in top_k_text
    )
    return hits / len(expected_keywords) if expected_keywords else 0


def compute_precision_at_k(
    docs, expected_keywords, k
):
    relevant = 0
    for d in docs[:k]:
        text = d.text.lower()
        if any(
            kw.lower() in text
            for kw in expected_keywords
        ):
            relevant += 1
    return relevant / k if k > 0 else 0


def compute_mrr(docs, expected_keywords):
    for rank, d in enumerate(docs):
        text = d.text.lower()
        if any(
            kw.lower() in text
            for kw in expected_keywords
        ):
            return 1.0 / (rank + 1)
    return 0.0


def main():
    llm_name = "deepseek"
    for arg in sys.argv:
        if arg.startswith("--llm="):
            llm_name = arg.split("=")[1]

    print("=" * 70)
    print("RETRIEVAL EVALUATION BENCHMARK")
    print(
        f"Methods: {METHODS_TO_TEST} | "
        f"K={K} | "
        f"Queries: {len(TEST_QUERIES)}"
    )
    print("=" * 70)

    llm = get_llm(llm_name)

    retrievers = {}
    for method in METHODS_TO_TEST:
        print(f"Loading {method}...", end=" ")
        try:
            retrievers[method] = Retriever(
                llm=llm,
                embedding_name=(
                    "sentence-transformer"
                ),
                method=method,
            )
            print("OK")
        except Exception as e:
            print(f"SKIP ({e})")

    results = {
        m: {
            "recall": [], "precision": [],
            "mrr": [], "time": [],
        }
        for m in retrievers
    }

    for qi, tq in enumerate(TEST_QUERIES):
        q = tq["question"]
        expected = (
            tq["expected_topics"]
            + tq["expected_entities"]
        )

        print(
            f"\n[{qi+1}/{len(TEST_QUERIES)}] "
            f"{q}"
        )

        for method, retriever in (
            retrievers.items()
        ):
            start = time.time()
            try:
                docs = retriever._base.retrieve(
                    q, n_results=K
                )
            except Exception:
                docs = []
            elapsed = time.time() - start

            recall = compute_recall_at_k(
                docs, expected, K
            )
            precision = compute_precision_at_k(
                docs, expected, K
            )
            mrr = compute_mrr(docs, expected)

            results[method]["recall"].append(
                recall
            )
            results[method]["precision"].append(
                precision
            )
            results[method]["mrr"].append(mrr)
            results[method]["time"].append(
                elapsed
            )

            print(
                f"  {method:12s} | "
                f"R@{K}={recall:.2f} "
                f"P@{K}={precision:.2f} "
                f"MRR={mrr:.2f} "
                f"({elapsed:.1f}s) "
                f"docs={len(docs)}"
            )

    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")
    print(
        f"{'Method':12s} | "
        f"{'Recall@'+str(K):>9s} | "
        f"{'Prec@'+str(K):>9s} | "
        f"{'MRR':>9s} | "
        f"{'Avg Time':>9s}"
    )
    print("-" * 60)

    for method in retrievers:
        r = results[method]
        n = len(r["recall"])
        if n == 0:
            continue

        avg_recall = sum(r["recall"]) / n
        avg_prec = sum(r["precision"]) / n
        avg_mrr = sum(r["mrr"]) / n
        avg_time = sum(r["time"]) / n

        print(
            f"{method:12s} | "
            f"{avg_recall:9.3f} | "
            f"{avg_prec:9.3f} | "
            f"{avg_mrr:9.3f} | "
            f"{avg_time:8.1f}s"
        )

    print(f"{'=' * 70}")

    best_method = max(
        retrievers,
        key=lambda m: (
            sum(results[m]["recall"])
            / max(len(results[m]["recall"]), 1)
        ),
    )
    print(
        f"\nBest by Recall@{K}: {best_method}"
    )

    best_mrr = max(
        retrievers,
        key=lambda m: (
            sum(results[m]["mrr"])
            / max(len(results[m]["mrr"]), 1)
        ),
    )
    print(f"Best by MRR: {best_mrr}")


if __name__ == "__main__":
    main()
