from retrieval.base import (
    BaseRetriever,
    RetrievedDoc,
)
from vector_store.store import VectorStore


class VectorRetriever(BaseRetriever):

    def __init__(self, store: VectorStore):
        self._store = store

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> list[RetrievedDoc]:
        emb = self._store._embed_query(query)

        post_results = self._store.query_posts(
            query,
            n_results=n_results,
            where=where,
            query_embedding=emb,
        )

        comment_results = (
            self._store.query_comments(
                query,
                n_results=n_results,
                where=where,
                query_embedding=emb,
            )
        )

        docs = []

        if post_results["ids"][0]:
            for doc, meta, dist in zip(
                post_results["documents"][0],
                post_results["metadatas"][0],
                post_results["distances"][0],
            ):
                docs.append(RetrievedDoc(
                    id=meta.get("post_id", ""),
                    text=doc,
                    score=round(1 - dist, 4),
                    metadata=meta,
                    source="post",
                ))

        if comment_results["ids"][0]:
            for doc, meta, dist in zip(
                comment_results["documents"][0],
                comment_results["metadatas"][0],
                comment_results["distances"][0],
            ):
                docs.append(RetrievedDoc(
                    id=meta.get(
                        "comment_id", ""
                    ),
                    text=doc,
                    score=round(1 - dist, 4),
                    metadata=meta,
                    source="comment",
                ))

        docs.sort(
            key=lambda d: d.score, reverse=True
        )
        return docs
