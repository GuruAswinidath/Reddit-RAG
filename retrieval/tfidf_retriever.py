import numpy as np
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import (
    cosine_similarity,
)

from retrieval.base import (
    BaseRetriever,
    RetrievedDoc,
    load_corpus,
    apply_where_filter,
)


class TFIDFRetriever(BaseRetriever):

    def __init__(self):
        posts, comments = load_corpus()
        self._corpus = posts + comments

        texts = [
            item["text"]
            for item in self._corpus
        ]

        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._tfidf_matrix = (
            self._vectorizer.fit_transform(texts)
        )

        print(
            f"[TF-IDF] Built index: "
            f"{len(self._corpus)} documents, "
            f"{len(self._vectorizer.vocabulary_)} "
            f"features"
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> list[RetrievedDoc]:
        query_vec = self._vectorizer.transform(
            [query]
        )
        scores = cosine_similarity(
            query_vec, self._tfidf_matrix
        )[0]

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

        docs = []
        for idx, score in (
            scored_items[:n_results]
        ):
            item = self._corpus[idx]
            docs.append(RetrievedDoc(
                id=item["id"],
                text=item["text"],
                score=round(float(score), 4),
                metadata=item["metadata"],
                source=item["source"],
            ))

        return docs
