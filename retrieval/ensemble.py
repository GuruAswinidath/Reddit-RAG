from retrieval.base import (
    BaseRetriever,
    RetrievedDoc,
)


class EnsembleRetriever(BaseRetriever):

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        k: int = 60,
    ):
        self._retrievers = retrievers
        self._k = k

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> list[RetrievedDoc]:
        all_results = []

        for retriever in self._retrievers:
            results = retriever.retrieve(
                query,
                n_results=n_results,
                where=where,
            )
            all_results.append(results)

        fused = reciprocal_rank_fusion(
            all_results, self._k
        )

        print(
            f"[Ensemble] Fused results from "
            f"{len(self._retrievers)} retrievers, "
            f"{len(fused)} unique documents"
        )

        return fused[:n_results]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedDoc]],
    k: int = 60,
) -> list[RetrievedDoc]:
    rrf_scores = {}
    doc_map = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            rrf_scores[doc.id] = (
                rrf_scores.get(doc.id, 0.0)
                + 1.0 / (k + rank + 1)
            )
            if doc.id not in doc_map:
                doc_map[doc.id] = doc

    fused = []
    for doc_id, rrf_score in sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        doc = doc_map[doc_id]
        fused.append(RetrievedDoc(
            id=doc.id,
            text=doc.text,
            score=round(rrf_score, 6),
            metadata=doc.metadata,
            source=doc.source,
        ))

    return fused
