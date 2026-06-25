import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from retrieval.llm import get_llm, LLM_MODELS
from retrieval.retriever import Retriever, METHODS


EMBED_MODEL = "sentence-transformer"

METHOD_INFO = {
    "vector": {
        "label": "Vector (Cosine Similarity)",
        "desc": "Embeds the query into a dense vector and finds nearest neighbors "
                "via ChromaDB HNSW index. Best for semantic / conceptual questions.",
        "speed": "Fast",
        "icon": "🔮",
    },
    "tfidf": {
        "label": "TF-IDF (Default)",
        "desc": "Sparse term-frequency vectors with cosine similarity. "
                "Best for exact keyword / name matches (model names, acronyms).",
        "speed": "Fast",
        "icon": "📊",
    },
    "bm25": {
        "label": "BM25",
        "desc": "Okapi BM25 probabilistic keyword ranking. Like TF-IDF but with "
                "term-frequency saturation and document-length normalization.",
        "speed": "Fast",
        "icon": "📈",
    },
    "multi-query": {
        "label": "Multi-Query (LLM-Expanded)",
        "desc": "Uses the LLM to rephrase the question into 3 variants, runs vector "
                "search for each, deduplicates and merges. Best for broad recall.",
        "speed": "Slower",
        "icon": "🔄",
    },
    "ensemble": {
        "label": "Ensemble (Vector+BM25+TF-IDF)",
        "desc": "Runs Vector + BM25 + TF-IDF in parallel and fuses results with "
                "Reciprocal Rank Fusion (k=60). Best single-system answer quality.",
        "speed": "Slower",
        "icon": "⚡",
    },
    "hybrid": {
        "label": "Hybrid (Vector + Graph + RRF)",
        "desc": "Runs Vector search (ChromaDB) and Graph traversal (Neo4j) in parallel. "
                "LLM routes the query type, results fused with RRF, graph analytics "
                "(sentiment trends, top authors, related entities) added to context. "
                "Requires NEO4J credentials in .env.",
        "speed": "Slowest",
        "icon": "🧠",
    },
}


# ── Page config ──────────────────────────────

st.set_page_config(
    page_title="Reddit RAG",
    page_icon="🔍",
    layout="wide",
)


# ── Auto-load retriever ─────────────────────

@st.cache_resource
def load_retriever(
    llm_name: str,
    method: str,
) -> Retriever:
    llm = get_llm(llm_name)
    return Retriever(
        llm=llm,
        embedding_name=EMBED_MODEL,
        method=method,
    )


# ── Session state init ───────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Sidebar ──────────────────────────────────

with st.sidebar:
    st.header("Settings")

    llm_name = st.selectbox(
        "LLM Model",
        options=list(LLM_MODELS.keys()),
        index=0,
    )

    method = st.selectbox(
        "Retrieval Method",
        options=METHODS,
        index=0,
        format_func=lambda m: (
            f"{METHOD_INFO.get(m, {}).get('icon', '•')} "
            f"{METHOD_INFO.get(m, {}).get('label', m)}"
        ),
    )

    info = METHOD_INFO.get(method, {})
    if info:
        st.caption(
            f"**{info['label']}** — {info['speed']}"
        )
        st.caption(info["desc"])

    st.divider()

    mode = st.radio(
        "Search Mode",
        options=["General", "Subreddit", "Temporal"],
        index=0,
    )

    subreddit = ""
    if mode == "Subreddit":
        subreddit = st.text_input(
            "Subreddit name",
            placeholder="e.g. MachineLearning",
        )

    n_results = st.slider(
        "Results per query",
        min_value=3,
        max_value=20,
        value=10,
    )

    st.divider()

    if st.button(
        "Clear Chat", use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        f"Embedding: {EMBED_MODEL} "
        f"(matches vector DB)"
    )


# ── Load retriever (auto, cached) ────────────

retriever = load_retriever(llm_name, method)


# ── Helpers ──────────────────────────────────

def format_sources(sources: list[dict]) -> str:
    if not sources:
        return ""

    lines = ["\n---\n**Sources:**"]
    seen = set()
    for s in sources:
        if s["type"] == "post":
            url = s.get("url", "")
            if url and url not in seen:
                seen.add(url)
                title = s.get("title", "")
                sub = s.get("subreddit", "")
                date = s.get("date", "")
                score = s.get("score", 0)
                label = (
                    f"\"{title}\"" if title
                    else "Post"
                )
                lines.append(
                    f"- {label} "
                    f"(r/{sub}, {date}) "
                    f"— score: {score} "
                    f"— [link]({url})"
                )
        else:
            author = s.get("author", "unknown")
            date = s.get("date", "")
            score = s.get("score", 0)
            lines.append(
                f"- Comment by u/{author} "
                f"({date}) "
                f"— score: {score}"
            )
    return "\n".join(lines)


def format_hybrid_meta(result: dict) -> str:
    parts = []

    route = result.get("route")
    if route:
        parts.append(f"Route: **{route}**")

    vc = result.get("vector_count")
    gc = result.get("graph_count")
    if vc is not None and gc is not None:
        parts.append(
            f"Vector: {vc} docs · "
            f"Graph: {gc} docs · "
            f"Fused: {result.get('doc_count', '?')}"
        )

    if not parts:
        return ""

    return (
        "\n\n---\n"
        f"*🧠 {' · '.join(parts)}*"
    )


def run_query(question: str) -> dict:
    if mode == "Temporal":
        return retriever.ask_temporal_comparison(
            question, n_results=n_results
        )
    elif mode == "Subreddit" and subreddit:
        return retriever.ask_by_subreddit(
            question=question,
            subreddit=subreddit,
            n_results=n_results,
        )
    else:
        return retriever.ask(
            question, n_results=n_results
        )


# ── Main chat area ───────────────────────────

st.title("Reddit RAG")

method_label = METHOD_INFO.get(
    method, {}
).get("label", method)
st.caption(
    f"Method: **{method_label}** · "
    f"LLM: **{llm_name}** · "
    f"Mode: **{mode}**"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask a question about Reddit discussions..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            f"Searching ({method_label})..."
        ):
            result = run_query(question)

        answer = result["answer"]

        sources_md = ""
        if "sources" in result:
            sources_md = format_sources(
                result["sources"]
            )

        if (
            mode == "Temporal"
            and "windows" in result
        ):
            windows = ", ".join(result["windows"])
            answer = (
                f"**Temporal comparison across "
                f"{windows}:**\n\n{answer}"
            )

        hybrid_md = format_hybrid_meta(result)

        result_method = result.get(
            "method", method
        )
        method_tag = (
            f"\n\n---\n*Retrieved via "
            f"**{METHOD_INFO.get(result_method, {}).get('label', result_method)}***"
        )

        full_response = (
            answer
            + sources_md
            + hybrid_md
            + method_tag
        )
        st.markdown(full_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )
