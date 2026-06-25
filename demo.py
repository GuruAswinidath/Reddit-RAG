"""
Demo script — runs 4 query types and shows:
  1. Graph-only results
  2. Vector-only results
  3. Fused (hybrid) results
  4. Final LLM answer

Usage:
  python demo.py
  python demo.py --llm=openai
"""

import sys

from retrieval.llm import get_llm
from retrieval.base import RetrievedDoc
from retrieval.retriever import Retriever

QUERIES = [
    {
        "label": (
            "A. Purely Semantic "
            "(vector-dominant)"
        ),
        "question": (
            "What do people think about using "
            "Claude for code generation?"
        ),
        "why": (
            "Needs semantic understanding of "
            "opinions/experiences — not entity "
            "lookups. Vector search excels at "
            "matching meaning across different "
            "phrasings."
        ),
    },
    {
        "label": (
            "B. Relationship/Traversal "
            "(graph-dominant)"
        ),
        "question": (
            "Who are the most influential "
            "users discussing Agentic AI "
            "and what companies do they "
            "mention?"
        ),
        "why": (
            "Requires graph traversal: "
            "User-[:AUTHORED]->Post"
            "-[:DISCUSSES]->Topic, "
            "Post-[:MENTIONS]->Company. "
            "Influence scores stored on "
            "User nodes."
        ),
    },
    {
        "label": (
            "C. Hybrid "
            "(needs both)"
        ),
        "question": (
            "Which open-source LLMs are being "
            "compared to Claude and Gemini, "
            "and what do people say about them?"
        ),
        "why": (
            "Graph finds entity connections "
            "(Model/Company co-mentions with "
            "Open Source LLM topic). Vector "
            "finds the actual opinions and "
            "experiences people describe."
        ),
    },
    {
        "label": (
            "D. Time-series Comparison "
            "(temporal)"
        ),
        "question": (
            "How has the discussion around "
            "RAG and Agentic AI evolved from "
            "early 2025 to mid 2026?"
        ),
        "why": (
            "Graph provides sentiment "
            "snapshots, trend growth %, "
            "topic evolution, and community "
            "activity per quarter. Vector "
            "retrieves representative posts "
            "from each time window."
        ),
    },
]

DIVIDER = "=" * 60
SUB_DIVIDER = "-" * 45


def main():
    llm_name = "deepseek"
    embed_name = "sentence-transformer"

    for arg in sys.argv:
        if arg.startswith("--llm="):
            llm_name = arg.split("=")[1]
        if arg.startswith("--embed="):
            embed_name = arg.split("=")[1]

    print(DIVIDER)
    print("REDDIT TEMPORAL RAG — DEMO")
    print(
        f"LLM: {llm_name} | "
        f"Embed: {embed_name}"
    )
    print(
        "Showing graph-only, vector-only, "
        "fused, and final answer for each query"
    )
    print(DIVIDER)

    llm = get_llm(llm_name)

    print("\nLoading hybrid retriever...")
    hybrid = Retriever(
        llm=llm,
        embedding_name=embed_name,
        method="hybrid",
    )

    for i, q in enumerate(QUERIES):
        label = q["label"]
        question = q["question"]
        why = q["why"]

        print(f"\n{DIVIDER}")
        print(f"QUERY {label}")
        print(f"{DIVIDER}")
        print(f"Q: {question}")
        print(f"Why: {why}")
        print(DIVIDER)

        if "Temporal" in label:
            _run_temporal(hybrid, question)
        else:
            _run_standard(hybrid, question)

    print(f"\n{DIVIDER}")
    print("DEMO COMPLETE")
    print(DIVIDER)


def _run_standard(retriever, question):
    base = retriever._base

    # --- 1. Graph-only ---
    print(f"\n{SUB_DIVIDER}")
    print("1. GRAPH-ONLY RESULTS")
    print(SUB_DIVIDER)

    graph_docs = []
    try:
        graph_docs = base._graph.retrieve(
            question, n_results=5
        )
        _print_docs(graph_docs)
        if base._graph.analytics:
            print(
                f"\n  Graph Analytics:\n"
                f"{base._graph.analytics}"
            )
    except Exception as e:
        print(f"  Error: {e}")

    # --- 2. Vector-only ---
    print(f"\n{SUB_DIVIDER}")
    print("2. VECTOR-ONLY RESULTS")
    print(SUB_DIVIDER)

    vector_docs = base._vector.retrieve(
        question, n_results=5
    )
    _print_docs(vector_docs)

    # --- 3. Fused ---
    print(f"\n{SUB_DIVIDER}")
    print("3. FUSED (HYBRID) RESULTS")
    print(SUB_DIVIDER)

    result = retriever.ask(
        question, n_results=5
    )

    print(
        f"  Route: {result.get('route', '?')}"
    )
    print(
        f"  Vector: "
        f"{result.get('vector_count', '?')} docs"
    )
    print(
        f"  Graph: "
        f"{result.get('graph_count', '?')} docs"
    )
    print(
        f"  Fused: "
        f"{result.get('doc_count', '?')} docs"
    )

    if result.get("sources"):
        print(
            f"\n  Top fused sources:"
        )
        for s in result["sources"][:5]:
            title = s.get("title", "")
            sub = s.get("subreddit", "")
            date = s.get("date", "")
            if s["type"] == "post" and title:
                print(
                    f"    [{s['type']}] "
                    f"\"{title[:60]}\" "
                    f"(r/{sub}, {date}) "
                    f"score={s['score']}"
                )
            else:
                author = s.get("author", "")
                print(
                    f"    [{s['type']}] "
                    f"u/{author} ({date}) "
                    f"score={s['score']}"
                )

    # --- 4. Final answer ---
    print(f"\n{SUB_DIVIDER}")
    print("4. FINAL LLM ANSWER")
    print(SUB_DIVIDER)
    print(result["answer"])


def _run_temporal(retriever, question):
    base = retriever._base

    # --- 1. Graph temporal analytics ---
    print(f"\n{SUB_DIVIDER}")
    print("1. GRAPH TEMPORAL ANALYTICS")
    print(SUB_DIVIDER)

    try:
        analytics = (
            base._graph.get_temporal_analytics(
                question
            )
        )
        if analytics:
            print(analytics)
        else:
            print("  No temporal analytics")
    except Exception as e:
        print(f"  Error: {e}")

    # --- 2. Vector per window ---
    print(f"\n{SUB_DIVIDER}")
    print("2. VECTOR RESULTS PER TIME WINDOW")
    print(SUB_DIVIDER)

    for window in ["W1", "W2", "W3"]:
        docs = base._vector.retrieve(
            question, n_results=3,
            where={"time_window": window},
        )
        print(
            f"\n  {window}: {len(docs)} results"
        )
        for d in docs[:2]:
            title = d.metadata.get(
                "title", ""
            )
            if not title and d.source == "post":
                title = d.text.split(
                    "\n"
                )[0][:80]
            print(
                f"    [{d.source}] "
                f"score={d.score} "
                f"{title[:70]}..."
            )

    # --- 3. Fused temporal ---
    print(f"\n{SUB_DIVIDER}")
    print("3. FUSED TEMPORAL COMPARISON")
    print(SUB_DIVIDER)

    result = retriever.ask_temporal_comparison(
        question
    )

    # --- 4. Final answer ---
    print(f"\n{SUB_DIVIDER}")
    print("4. FINAL LLM ANSWER")
    print(SUB_DIVIDER)
    print(result["answer"])


def _print_docs(docs: list[RetrievedDoc]):
    if not docs:
        print("  (no results)")
        return

    for i, d in enumerate(docs[:5]):
        retriever_tag = d.metadata.get(
            "retriever", ""
        )
        tag = (
            f" [{retriever_tag}]"
            if retriever_tag else ""
        )

        title = d.metadata.get("title", "")
        if not title and d.source == "post":
            title = d.text.split("\n")[0][:80]

        sub = d.metadata.get("subreddit", "")
        date = (
            d.metadata.get("created_at", "")
            or d.metadata.get("period", "")
        )
        if date and "T" in str(date):
            date = str(date)[:10]

        print(
            f"\n  [{i+1}]{tag} "
            f"score={d.score} ({d.source})"
        )
        if title:
            print(f"    \"{title}\"")
        if sub or date:
            print(
                f"    r/{sub} | {date}"
            )
        print(
            f"    {d.text[:120]}..."
        )


if __name__ == "__main__":
    main()
