from difflib import SequenceMatcher

CANONICAL_MAP = {
    "rag": "RAG",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "retrieval augmented": "RAG",
    "llm": "LLM",
    "large language model": "LLM",
    "large language models": "LLM",
    "llms": "LLM",
    "ai safety": "AI Safety",
    "ai alignment": "AI Safety",
    "alignment": "AI Safety",
    "guardrails": "AI Safety",
    "red teaming": "AI Safety",
    "jailbreak": "AI Safety",
    "jailbreaking": "AI Safety",
    "open source llm": "Open Source LLM",
    "open-source llm": "Open Source LLM",
    "open source": "Open Source LLM",
    "local llm": "Open Source LLM",
    "self-hosted": "Open Source LLM",
    "agentic ai": "Agentic AI",
    "ai agent": "Agentic AI",
    "ai agents": "Agentic AI",
    "agentic": "Agentic AI",
    "multi-agent": "Agentic AI",
    "autonomous agent": "Agentic AI",
    "tool use": "Agentic AI",
    "function calling": "Agentic AI",
    "fine-tuning": "Fine-tuning",
    "fine tuning": "Fine-tuning",
    "finetuning": "Fine-tuning",
    "finetune": "Fine-tuning",
    "fine-tune": "Fine-tuning",
    "lora": "Fine-tuning",
    "qlora": "Fine-tuning",
    "peft": "Fine-tuning",
    "vector database": "Vector Databases",
    "vector databases": "Vector Databases",
    "vector store": "Vector Databases",
    "vector db": "Vector Databases",
    "embedding": "Embeddings",
    "embeddings": "Embeddings",
    "text-embedding": "Embeddings",
    "sentence-transformer": "Embeddings",
    "prompt engineering": "Prompt Engineering",
    "prompting": "Prompt Engineering",
    "system prompt": "Prompt Engineering",
    "few-shot": "Prompt Engineering",
    "chain of thought": "Prompt Engineering",
    "zero-shot": "Prompt Engineering",
    "llm evaluation": "LLM Evaluation",
    "evaluation": "LLM Evaluation",
    "benchmark": "LLM Evaluation",
    "leaderboard": "LLM Evaluation",
    "mmlu": "LLM Evaluation",
    "arena": "LLM Evaluation",
    "inference": "Inference",
    "quantization": "Inference",
    "gguf": "Inference",
    "gptq": "Inference",
    "vllm": "Inference",
    "multimodal": "Multimodal",
    "vision": "Multimodal",
    "image generation": "Multimodal",
    "text-to-image": "Multimodal",
    "code generation": "Code Generation",
    "copilot": "Code Generation",
    "code assistant": "Code Generation",
    "coding agent": "Code Generation",
}

CANONICAL_NAMES = sorted(
    set(CANONICAL_MAP.values())
)

_SIMILARITY_THRESHOLD = 0.8


def canonicalize_topic(topic: str) -> str:
    key = topic.lower().strip()

    if key in CANONICAL_MAP:
        return CANONICAL_MAP[key]

    best_score = 0
    best_match = None
    for alias, canonical in CANONICAL_MAP.items():
        ratio = SequenceMatcher(
            None, key, alias
        ).ratio()
        if ratio > best_score:
            best_score = ratio
            best_match = canonical

    if (
        best_match
        and best_score >= _SIMILARITY_THRESHOLD
    ):
        return best_match

    return topic


def canonicalize_topics(
    topics: list[str],
) -> list[str]:
    seen = set()
    result = []
    for t in topics:
        canonical = canonicalize_topic(t)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result
