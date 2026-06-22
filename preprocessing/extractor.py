import re
from datetime import datetime, timedelta

from preprocessing.patterns import (
    URL_PATTERN,
    REDDIT_USER_PATTERN,
    SUBREDDIT_PATTERN,
    RELATIVE_TIME_PATTERN,
    ABSOLUTE_DATE_PATTERN,
    AI_COMPANY_PATTERN,
    AI_MODEL_PATTERN,
    MODEL_NORMALIZATION,
    REDDIT_ASSET_PATTERNS,
)


# Maps all abbreviations to a canonical unit
TIME_UNIT_MAP = {
    "s": "second", "sec": "second",
    "secs": "second", "second": "second",
    "seconds": "second",
    "m": "minute", "min": "minute",
    "mins": "minute", "minute": "minute",
    "minutes": "minute",
    "h": "hour", "hr": "hour",
    "hrs": "hour", "hour": "hour",
    "hours": "hour",
    "d": "day", "day": "day", "days": "day",
    "w": "week", "wk": "week", "wks": "week",
    "week": "week", "weeks": "week",
    "mo": "month", "mos": "month",
    "month": "month", "months": "month",
    "y": "year", "yr": "year", "yrs": "year",
    "year": "year", "years": "year",
}

TIME_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
    "week": 604800,
    "month": 2592000,
    "year": 31536000,
}

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3,
    "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

W1_START = datetime(2025, 1, 1)
W1_END = datetime(2025, 6, 30, 23, 59, 59)
W2_START = datetime(2025, 7, 1)
W2_END = datetime(2025, 12, 31, 23, 59, 59)
W3_START = datetime(2026, 1, 1)
W3_END = datetime(2026, 6, 30, 23, 59, 59)


def extract_timestamp(
    text: str,
    scraped_at: str,
) -> str | None:
    if not text:
        return None

    match = RELATIVE_TIME_PATTERN.search(text)
    if match:
        amount = int(match.group(1))
        raw_unit = match.group(2).lower().rstrip(".")
        unit = TIME_UNIT_MAP.get(raw_unit)

        if unit:
            seconds = (
                amount * TIME_SECONDS[unit]
            )

            try:
                scrape_dt = (
                    datetime.fromisoformat(
                        scraped_at
                    )
                )
            except (ValueError, TypeError):
                scrape_dt = datetime.now()

            created = scrape_dt - timedelta(
                seconds=seconds
            )
            return created.isoformat()

    match = ABSOLUTE_DATE_PATTERN.search(text)
    if match:
        month = MONTH_MAP.get(
            match.group(1).lower()[:3], 1
        )
        day = int(match.group(2))
        year = int(match.group(3))
        try:
            dt = datetime(year, month, day)
            return dt.isoformat()
        except ValueError:
            pass

    return None


def assign_time_window(
    created_at: str | None,
) -> str:
    if not created_at:
        return "UNKNOWN"

    try:
        dt = datetime.fromisoformat(created_at)
    except (ValueError, TypeError):
        return "UNKNOWN"

    if W1_START <= dt <= W1_END:
        return "W1"
    if W2_START <= dt <= W2_END:
        return "W2"
    if W3_START <= dt <= W3_END:
        return "W3"

    return "OUT_OF_RANGE"


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    urls = URL_PATTERN.findall(text)
    return [
        url for url in set(urls)
        if not any(
            asset in url
            for asset in REDDIT_ASSET_PATTERNS
        )
    ]


def extract_mentioned_users(
    text: str,
) -> list[str]:
    if not text:
        return []
    return list(
        set(REDDIT_USER_PATTERN.findall(text))
    )


def extract_mentioned_subreddits(
    text: str,
) -> list[str]:
    if not text:
        return []
    return list(
        set(SUBREDDIT_PATTERN.findall(text))
    )


def extract_companies(text: str) -> list[str]:
    if not text:
        return []
    matches = AI_COMPANY_PATTERN.findall(text)
    seen = set()
    result = []
    for m in matches:
        normalized = m.strip().title()
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def normalize_model_name(name: str) -> str:
    key = name.strip().lower().rstrip(".,;:!?")
    return MODEL_NORMALIZATION.get(key, name.strip())


def extract_models(text: str) -> list[str]:
    if not text:
        return []
    matches = AI_MODEL_PATTERN.findall(text)
    seen = set()
    result = []
    for m in matches:
        normalized = normalize_model_name(m)
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


TOPIC_KEYWORDS = {
    "RAG": [
        "rag", "retrieval augmented",
        "retrieval-augmented",
    ],
    "Vector Databases": [
        "vector database", "vector store",
        "pinecone", "qdrant", "chroma",
        "weaviate", "milvus", "faiss",
    ],
    "Embeddings": [
        "embedding", "embeddings",
        "text-embedding", "sentence-transformer",
    ],
    "AI Safety": [
        "ai safety", "alignment",
        "guardrails", "jailbreak",
        "red teaming",
    ],
    "Fine-tuning": [
        "fine-tune", "fine-tuning",
        "finetune", "finetuning",
        "lora", "qlora", "peft",
    ],
    "Agents": [
        "ai agent", "agentic",
        "tool use", "function calling",
        "multi-agent", "autonomous agent",
    ],
    "Open Source LLM": [
        "open source", "open-source",
        "local llm", "self-hosted",
        "ollama", "llama.cpp",
    ],
    "Prompt Engineering": [
        "prompt engineering", "prompting",
        "system prompt", "few-shot",
        "chain of thought", "zero-shot",
    ],
    "LLM Evaluation": [
        "evaluation", "benchmark",
        "leaderboard", "eval", "mmlu",
        "arena", "elo",
    ],
    "Inference": [
        "inference", "quantization",
        "gguf", "gptq", "awq",
        "vllm", "tgi",
    ],
    "Multimodal": [
        "multimodal", "vision",
        "image generation", "text-to-image",
        "image-to-text",
    ],
    "Code Generation": [
        "code generation", "copilot",
        "code assistant", "coding agent",
    ],
}


def extract_topics(text: str) -> list[str]:
    if not text:
        return []
    lower = text.lower()
    found = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            found.append(topic)
    return found


def extract_all(
    text: str,
    scraped_at: str = "",
) -> dict:
    return {
        "urls": extract_urls(text),
        "mentioned_users": (
            extract_mentioned_users(text)
        ),
        "mentioned_subreddits": (
            extract_mentioned_subreddits(text)
        ),
        "mentioned_companies": (
            extract_companies(text)
        ),
        "mentioned_models": (
            extract_models(text)
        ),
    }
