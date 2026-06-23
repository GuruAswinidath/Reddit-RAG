from rank_bm25 import BM25Okapi

from retrieval.base import (
    BaseRetriever,
    RetrievedDoc,
    load_corpus,
)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever(BaseRetriever):

    def __init__(self):
        posts, comments = load_corpus()
        self._corpus = posts + comments

        tokenized = [
            _tokenize(item["text"])
            for item in self._corpus
        ]

        self._bm25 = BM25Okapi(tokenized)

        print(
            f"[BM25] Built index: "
            f"{len(self._corpus)} documents"
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> list[RetrievedDoc]:
        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(
            tokenized_query
        )

        scored_items = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            item = self._corpus[idx]
            if where:
                meta = item.get("metadata", {})
                if not all(
                    meta.get(k) == v
                    for k, v in where.items()
                ):
                    continue
            scored_items.append((idx, score))

        scored_items.sort(
            key=lambda x: x[1], reverse=True
        )

        max_score = (
            scored_items[0][1]
            if scored_items
            else 1.0
        )

        docs = []
        for idx, score in (
            scored_items[:n_results]
        ):
            item = self._corpus[idx]
            docs.append(RetrievedDoc(
                id=item["id"],
                text=item["text"],
                score=round(
                    float(score / max_score), 4
                ),
                metadata=item["metadata"],
                source=item["source"],
            ))

        return docs
