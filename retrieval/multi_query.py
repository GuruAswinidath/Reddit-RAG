from retrieval.base import (
    BaseRetriever,
    RetrievedDoc,
)
from retrieval.llm import LLMModel


REPHRASE_PROMPT = (
    "Given the following question, generate "
    "{n} alternative phrasings that would help "
    "find relevant Reddit discussions. "
    "Return ONLY the rephrased questions, "
    "one per line. No numbering, no explanations."
    "\n\nOriginal question: {question}"
)


class MultiQueryRetriever(BaseRetriever):

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLMModel,
        n_queries: int = 3,
    ):
        self._retriever = retriever
        self._llm = llm
        self._n_queries = n_queries

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> list[RetrievedDoc]:
        queries = self._generate_variants(query)
        all_queries = [query] + queries

        print(
            f"[Multi-Query] Searching with "
            f"{len(all_queries)} query variants"
        )

        seen = {}

        for q in all_queries:
            results = self._retriever.retrieve(
                q,
                n_results=n_results,
                where=where,
            )
            for doc in results:
                if (
                    doc.id not in seen
                    or doc.score > seen[doc.id].score
                ):
                    seen[doc.id] = doc

        merged = sorted(
            seen.values(),
            key=lambda d: d.score,
            reverse=True,
        )

        return merged[:n_results]

    def _generate_variants(
        self, question: str
    ) -> list[str]:
        prompt = REPHRASE_PROMPT.format(
            n=self._n_queries,
            question=question,
        )

        response = self._llm.generate(
            prompt=prompt
        )

        variants = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip()
            and len(line.strip()) > 10
        ]

        return variants[: self._n_queries]
