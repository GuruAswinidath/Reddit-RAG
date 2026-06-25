from vector_store.embeddings import (
    get_embedding_model,
)
from vector_store.store import VectorStore
from retrieval.base import BaseRetriever, RetrievedDoc
from retrieval.llm import LLMModel


SYSTEM_PROMPT = (
    "You are a Reddit discussion analyst. "
    "Answer the user's question based ONLY on "
    "the retrieved Reddit posts and comments "
    "provided below. "
    "Cite your sources using the format: "
    "\"Title\" (r/subreddit, date) [Post N]. "
    "If the context doesn't contain enough "
    "information, say so. "
    "When Graph Analytics are provided, "
    "incorporate the sentiment trends, "
    "influential users, entity relationships, "
    "and community data into your answer. "
    "For temporal questions, compare across "
    "time windows (W1: Jan-Jun 2025, "
    "W2: Jul-Dec 2025, W3: Jan-Jun 2026)."
)

METHODS = [
    "vector", "tfidf", "bm25",
    "multi-query", "ensemble", "hybrid",
]


class Retriever:

    def __init__(
        self,
        llm: LLMModel,
        embedding_name: str = "sentence-transformer",
        method: str = "tfidf",
    ):
        self._llm = llm
        self._embedding_name = embedding_name
        self._method = method
        self._store = None
        self._base = self._build_retriever(method)

    def _get_store(self) -> VectorStore:
        if self._store is None:
            embedder = get_embedding_model(
                self._embedding_name
            )
            self._store = VectorStore(embedder)
        return self._store

    def _build_retriever(
        self, method: str
    ) -> BaseRetriever:
        if method == "vector":
            from retrieval.vector_retriever import (
                VectorRetriever,
            )
            return VectorRetriever(
                self._get_store()
            )

        if method == "tfidf":
            from retrieval.tfidf_retriever import (
                TFIDFRetriever,
            )
            return TFIDFRetriever()

        if method == "bm25":
            from retrieval.bm25_retriever import (
                BM25Retriever,
            )
            return BM25Retriever()

        if method == "multi-query":
            from retrieval.vector_retriever import (
                VectorRetriever,
            )
            from retrieval.multi_query import (
                MultiQueryRetriever,
            )
            return MultiQueryRetriever(
                retriever=VectorRetriever(
                    self._get_store()
                ),
                llm=self._llm,
            )

        if method == "ensemble":
            from retrieval.vector_retriever import (
                VectorRetriever,
            )
            from retrieval.bm25_retriever import (
                BM25Retriever,
            )
            from retrieval.tfidf_retriever import (
                TFIDFRetriever,
            )
            from retrieval.ensemble import (
                EnsembleRetriever,
            )
            return EnsembleRetriever(
                retrievers=[
                    VectorRetriever(
                        self._get_store()
                    ),
                    BM25Retriever(),
                    TFIDFRetriever(),
                ],
            )

        if method == "hybrid":
            from retrieval.vector_retriever import (
                VectorRetriever,
            )
            from retrieval.graph_retriever import (
                GraphRetriever,
            )
            from retrieval.hybrid import (
                HybridRetriever,
            )
            return HybridRetriever(
                vector_retriever=VectorRetriever(
                    self._get_store()
                ),
                graph_retriever=GraphRetriever(),
                llm=self._llm,
            )

        raise ValueError(
            f"Unknown method: {method}. "
            f"Options: {METHODS}"
        )

    def ask(
        self,
        question: str,
        n_results: int = 10,
        where: dict = None,
    ) -> dict:
        docs = self._base.retrieve(
            question,
            n_results=n_results,
            where=where,
        )

        context = _build_context_from_docs(docs)

        analytics = ""
        if hasattr(self._base, "analytics"):
            analytics = self._base.analytics

        route = ""
        if hasattr(self._base, "route"):
            route = self._base.route

        prompt_parts = [
            f"## Retrieved Context\n\n{context}",
        ]
        if analytics:
            prompt_parts.append(
                f"## Graph Analytics\n\n"
                f"{analytics}"
            )
        prompt_parts.append(
            f"## Question\n\n{question}"
        )

        prompt = "\n\n".join(prompt_parts)

        answer = self._llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
        )

        sources = _extract_sources_from_docs(docs)

        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "method": self._method,
            "doc_count": len(docs),
        }

        if route:
            result["route"] = route

        if hasattr(self._base, "vector_docs"):
            result["vector_count"] = len(
                self._base.vector_docs
            )
            result["graph_count"] = len(
                self._base.graph_docs
            )

        return result

    def ask_with_time_filter(
        self,
        question: str,
        time_window: str,
        n_results: int = 10,
    ) -> dict:
        return self.ask(
            question=question,
            n_results=n_results,
            where={"time_window": time_window},
        )

    def ask_temporal_comparison(
        self,
        question: str,
        windows: list[str] = None,
        n_results: int = 5,
    ) -> dict:
        if not windows:
            windows = ["W1", "W2", "W3"]

        window_contexts = {}

        for window in windows:
            docs = self._base.retrieve(
                question,
                n_results=n_results,
                where={"time_window": window},
            )
            context = _build_context_from_docs(
                docs
            )
            window_contexts[window] = context

        sections = []
        for window, ctx in (
            window_contexts.items()
        ):
            sections.append(
                f"### {window}\n{ctx}"
            )

        full_context = "\n\n".join(sections)

        temporal_analytics = ""
        if hasattr(
            self._base, "_graph"
        ) and hasattr(
            self._base._graph,
            "get_temporal_analytics",
        ):
            try:
                temporal_analytics = (
                    self._base._graph
                    .get_temporal_analytics(
                        question, windows
                    )
                )
            except Exception:
                pass

        prompt_parts = [
            f"## Retrieved Context "
            f"(by time window)\n\n"
            f"{full_context}",
        ]
        if temporal_analytics:
            prompt_parts.append(
                f"## Graph Temporal Analytics"
                f"\n\n{temporal_analytics}"
            )
        prompt_parts.append(
            f"## Question\n\n"
            f"{question}\n\n"
            f"Compare the discussion across "
            f"time windows. Highlight what "
            f"changed, what's new, and what "
            f"trends emerged."
        )

        prompt = "\n\n".join(prompt_parts)

        answer = self._llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
        )

        return {
            "question": question,
            "answer": answer,
            "windows": windows,
            "method": self._method,
        }

    def ask_by_subreddit(
        self,
        question: str,
        subreddit: str,
        n_results: int = 10,
    ) -> dict:
        return self.ask(
            question=question,
            n_results=n_results,
            where={"subreddit": subreddit},
        )


def _extract_title(doc: RetrievedDoc) -> str:
    title = doc.metadata.get("title", "")
    if not title and doc.source == "post":
        title = doc.text.split("\n")[0][:120]
    return title


def _format_date(doc: RetrievedDoc) -> str:
    raw = (
        doc.metadata.get("created_at")
        or doc.metadata.get("period", "")
    )
    if not raw:
        return doc.metadata.get(
            "time_window", ""
        )
    if "T" in str(raw):
        return str(raw)[:10]
    return str(raw)


def _build_context_from_docs(
    docs: list[RetrievedDoc],
) -> str:
    posts = [
        d for d in docs if d.source == "post"
    ]
    comments = [
        d for d in docs
        if d.source == "comment"
    ]

    parts = []

    if posts:
        parts.append("### Posts")
        for i, doc in enumerate(posts):
            meta = doc.metadata
            title = _extract_title(doc)
            sub = meta.get("subreddit", "")
            date = _format_date(doc)
            author = meta.get("author", "")
            url = meta.get("url", "")
            topics = meta.get("topics", "")
            via = meta.get("retriever", "")
            via_tag = (
                f" [{via}]" if via else ""
            )

            header = (
                f"[Post {i+1}]{via_tag} "
                f"\"{title}\" "
                f"(r/{sub}, {date})"
            )
            detail = (
                f"score={doc.score} | "
                f"u/{author} | "
                f"topics: {topics}"
            )

            parts.append(
                f"{header}\n{detail}\n"
                f"URL: {url}\n"
                f"{doc.text[:1000]}"
            )

    if comments:
        parts.append("\n### Comments")
        for i, doc in enumerate(comments):
            meta = doc.metadata
            author = meta.get("author", "")
            depth = meta.get("depth", 0)
            date = _format_date(doc)
            via = meta.get("retriever", "")
            via_tag = (
                f" [{via}]" if via else ""
            )

            parts.append(
                f"[Comment {i+1}]{via_tag} "
                f"u/{author} ({date}) "
                f"depth={depth} "
                f"score={doc.score}\n"
                f"{doc.text[:500]}"
            )

    return "\n\n".join(parts)


def _extract_sources_from_docs(
    docs: list[RetrievedDoc],
) -> list[dict]:
    sources = []

    for i, doc in enumerate(docs):
        entry = {
            "type": doc.source,
            "index": i + 1,
            "score": doc.score,
            "time_window": doc.metadata.get(
                "time_window", ""
            ),
            "date": _format_date(doc),
        }

        if doc.source == "post":
            entry["title"] = _extract_title(doc)
            entry["post_id"] = doc.metadata.get(
                "post_id", ""
            )
            entry["subreddit"] = doc.metadata.get(
                "subreddit", ""
            )
            entry["url"] = doc.metadata.get(
                "url", ""
            )
        else:
            entry["comment_id"] = (
                doc.metadata.get(
                    "comment_id", ""
                )
            )
            entry["post_id"] = doc.metadata.get(
                "post_id", ""
            )
            entry["author"] = doc.metadata.get(
                "author", ""
            )

        sources.append(entry)

    return sources
