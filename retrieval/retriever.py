from vector_store.embeddings import (
    get_embedding_model,
)
from vector_store.store import VectorStore
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


class Retriever:

    def __init__(
        self,
        llm: LLMModel,
        embedding_name: str = "sentence-transformer",
    ):
        self._llm = llm
        self._embedder = get_embedding_model(
            embedding_name
        )
        self._store = VectorStore(self._embedder)

    def ask(
        self,
        question: str,
        n_results: int = 10,
        where: dict = None,
    ) -> dict:
        emb = self._store._embed_query(question)

        post_results = self._store.query_posts(
            question,
            n_results=n_results,
            where=where,
            query_embedding=emb,
        )

        comment_results = (
            self._store.query_comments(
                question,
                n_results=n_results,
                where=where,
                query_embedding=emb,
            )
        )

        context = _build_context(
            post_results, comment_results
        )

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

        sources = _extract_sources(
            post_results, comment_results
        )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "post_count": len(
                post_results["ids"][0]
            ),
            "comment_count": len(
                comment_results["ids"][0]
            ),
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

        emb = self._store._embed_query(question)

        window_contexts = {}

        for window in windows:
            post_results = (
                self._store.query_posts(
                    question,
                    n_results=n_results,
                    where={
                        "time_window": window
                    },
                    query_embedding=emb,
                )
            )
            comment_results = (
                self._store.query_comments(
                    question,
                    n_results=n_results,
                    where={
                        "time_window": window
                    },
                    query_embedding=emb,
                )
            )

            context = _build_context(
                post_results, comment_results
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


def _build_context(
    post_results: dict,
    comment_results: dict,
) -> str:
    parts = []

    if post_results["ids"][0]:
        parts.append("### Posts")
        for i, (doc, meta, dist) in enumerate(
            zip(
                post_results["documents"][0],
                post_results["metadatas"][0],
                post_results["distances"][0],
            )
        ):
            score = round(1 - dist, 3)
            sub = meta.get("subreddit", "")
            tw = meta.get("time_window", "")
            author = meta.get("author", "")
            url = meta.get("url", "")
            topics = meta.get("topics", "")

            parts.append(
                f"[Post {i+1}] "
                f"score={score} | "
                f"r/{sub} | {tw} | "
                f"u/{author} | "
                f"topics: {topics}\n"
                f"URL: {url}\n"
                f"{doc[:1000]}"
            )

    if comment_results["ids"][0]:
        parts.append("\n### Comments")
        for i, (doc, meta, dist) in enumerate(
            zip(
                comment_results["documents"][0],
                comment_results["metadatas"][0],
                comment_results["distances"][0],
            )
        ):
            score = round(1 - dist, 3)
            author = meta.get("author", "")
            depth = meta.get("depth", 0)
            tw = meta.get("time_window", "")

            parts.append(
                f"[Comment {i+1}] "
                f"score={score} | "
                f"u/{author} | "
                f"depth={depth} | {tw}\n"
                f"{doc[:500]}"
            )

    return "\n\n".join(parts)


def _extract_sources(
    post_results: dict,
    comment_results: dict,
) -> list[dict]:
    sources = []

    for i, (meta, dist) in enumerate(
        zip(
            post_results["metadatas"][0],
            post_results["distances"][0],
        )
    ):
        sources.append({
            "type": "post",
            "index": i + 1,
            "score": round(1 - dist, 3),
            "post_id": meta.get("post_id", ""),
            "subreddit": meta.get(
                "subreddit", ""
            ),
            "url": meta.get("url", ""),
            "time_window": meta.get(
                "time_window", ""
            ),
        })

    for i, (meta, dist) in enumerate(
        zip(
            comment_results["metadatas"][0],
            comment_results["distances"][0],
        )
    ):
        sources.append({
            "type": "comment",
            "index": i + 1,
            "score": round(1 - dist, 3),
            "comment_id": meta.get(
                "comment_id", ""
            ),
            "post_id": meta.get("post_id", ""),
            "author": meta.get("author", ""),
            "time_window": meta.get(
                "time_window", ""
            ),
        })

    return sources
