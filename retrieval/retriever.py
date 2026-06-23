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
    "Cite your sources using [Post N] or "
    "[Comment N] references. "
    "If the context doesn't contain enough "
    "information, say so. "
    "For temporal questions, compare across "
    "time windows (W1: Jan-Jun 2025, "
    "W2: Jul-Dec 2025, W3: Jan-Jun 2026)."
)

METHODS = [
    "vector", "tfidf", "bm25",
    "multi-query", "ensemble",
]


class Retriever:

    def __init__(
        self,
        llm: LLMModel,
        embedding_name: str = "sentence-transformer",
        method: str = "vector",
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

        prompt = (
            f"## Retrieved Context\n\n"
            f"{context}\n\n"
            f"## Question\n\n"
            f"{question}"
        )

        answer = self._llm.generate(
            prompt=prompt,
            system=SYSTEM_PROMPT,
        )

        sources = _extract_sources_from_docs(docs)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "method": self._method,
            "doc_count": len(docs),
        }

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

        prompt = (
            f"## Retrieved Context "
            f"(by time window)\n\n"
            f"{full_context}\n\n"
            f"## Question\n\n"
            f"{question}\n\n"
            f"Compare the discussion across "
            f"time windows. Highlight what "
            f"changed, what's new, and what "
            f"trends emerged."
        )

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


def _build_context_from_docs(
    docs: list[RetrievedDoc],
) -> str:
    posts = [
        d for d in docs if d.source == "post"
    ]
    comments = [
        d for d in docs if d.source == "comment"
    ]

    parts = []

    if posts:
        parts.append("### Posts")
        for i, doc in enumerate(posts):
            meta = doc.metadata
            sub = meta.get("subreddit", "")
            tw = meta.get("time_window", "")
            author = meta.get("author", "")
            url = meta.get("url", "")
            topics = meta.get("topics", "")

            parts.append(
                f"[Post {i+1}] "
                f"score={doc.score} | "
                f"r/{sub} | {tw} | "
                f"u/{author} | "
                f"topics: {topics}\n"
                f"URL: {url}\n"
                f"{doc.text[:1000]}"
            )

    if comments:
        parts.append("\n### Comments")
        for i, doc in enumerate(comments):
            meta = doc.metadata
            author = meta.get("author", "")
            depth = meta.get("depth", 0)
            tw = meta.get("time_window", "")

            parts.append(
                f"[Comment {i+1}] "
                f"score={doc.score} | "
                f"u/{author} | "
                f"depth={depth} | {tw}\n"
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
        }

        if doc.source == "post":
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
