from retrieval.base import (
    BaseRetriever,
    RetrievedDoc,
)
from retrieval.ensemble import (
    reciprocal_rank_fusion,
)
from retrieval.router import QueryRouter
from retrieval.llm import LLMModel


class HybridRetriever(BaseRetriever):

    ROUTE_WEIGHTS = {
        "VECTOR": [1.4, 0.6],
        "GRAPH": [0.2, 1.8],
        "HYBRID": [1.0, 1.0],
        "TEMPORAL": [0.6, 1.4],
    }

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        graph_retriever,
        llm: LLMModel,
        k: int = 60,
    ):
        self._vector = vector_retriever
        self._graph = graph_retriever
        self._router = QueryRouter(llm)
        self._k = k
        self._last_route = ""
        self._last_analytics = ""
        self._last_vector_docs = []
        self._last_graph_docs = []

    @property
    def route(self) -> str:
        return self._last_route

    @property
    def analytics(self) -> str:
        return self._last_analytics

    @property
    def vector_docs(self) -> list[RetrievedDoc]:
        return self._last_vector_docs

    @property
    def graph_docs(self) -> list[RetrievedDoc]:
        return self._last_graph_docs

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> list[RetrievedDoc]:
        self._last_route = self._router.route(
            query
        )

        print(
            f"[Router] Query classified as: "
            f"{self._last_route}"
        )

        self._last_vector_docs = (
            self._vector.retrieve(
                query, n_results, where
            )
        )

        try:
            self._last_graph_docs = (
                self._graph.retrieve(
                    query, n_results, where
                )
            )
            self._last_analytics = (
                self._graph.analytics
            )
        except Exception as e:
            print(f"[Hybrid] Graph error: {e}")
            self._last_graph_docs = []
            self._last_analytics = ""

        if (
            not self._last_vector_docs
            and not self._last_graph_docs
        ):
            return []

        if not self._last_graph_docs:
            return self._last_vector_docs

        if not self._last_vector_docs:
            return self._last_graph_docs

        weights = self.ROUTE_WEIGHTS.get(
            self._last_route, [1.0, 1.5]
        )

        fused = reciprocal_rank_fusion(
            [
                self._last_vector_docs,
                self._last_graph_docs,
            ],
            self._k,
            weights=weights,
        )

        print(
            f"[Hybrid] Vector: "
            f"{len(self._last_vector_docs)} docs, "
            f"Graph: "
            f"{len(self._last_graph_docs)} docs, "
            f"Fused: {len(fused)} docs"
        )

        return fused[:n_results]
