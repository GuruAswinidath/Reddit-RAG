from retrieval.llm import LLMModel


ROUTE_PROMPT = (
    "Classify this question into exactly one "
    "category. Return ONLY the category name, "
    "nothing else.\n\n"
    "Categories:\n"
    "VECTOR - Semantic questions about opinions, "
    "experiences, or general discussions "
    "(e.g. 'What do people think about X?')\n"
    "GRAPH - Questions about relationships, "
    "influential users, entity connections, "
    "community structure "
    "(e.g. 'Who are the top contributors?')\n"
    "HYBRID - Questions needing both semantic "
    "understanding AND entity/relationship analysis "
    "(e.g. 'Which companies lead open-source LLMs "
    "and what do people say?')\n"
    "TEMPORAL - Questions about changes over time, "
    "trends, evolution, period comparisons "
    "(e.g. 'How has sentiment changed?')\n\n"
    "Question: {question}\n\n"
    "Category:"
)

VALID_ROUTES = {
    "VECTOR", "GRAPH", "HYBRID", "TEMPORAL"
}


class QueryRouter:

    def __init__(self, llm: LLMModel):
        self._llm = llm

    def route(self, question: str) -> str:
        prompt = ROUTE_PROMPT.format(
            question=question
        )

        response = self._llm.generate(
            prompt=prompt
        ).strip().upper()

        for route in VALID_ROUTES:
            if route in response:
                return route

        return "HYBRID"
