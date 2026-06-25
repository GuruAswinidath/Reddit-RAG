import json
import os
import re

from dotenv import load_dotenv

from preprocessing.canonicalize import (
    canonicalize_topics,
)

load_dotenv()

EXTRACT_PROMPT = (
    "Extract structured data from this Reddit "
    "post. Return ONLY valid JSON.\n\n"
    '{{\n'
    '  "topics": ["list of discussion topics '
    'like RAG, AI Safety, Fine-tuning"],\n'
    '  "companies": ["company names '
    'like OpenAI, Anthropic, Google"],\n'
    '  "models": ["AI model names '
    'like GPT-4, Claude, Llama"],\n'
    '  "sentiment": "positive or negative '
    'or neutral",\n'
    '  "sentiment_score": 0.0,\n'
    '  "summary": "one line summary"\n'
    '}}\n\n'
    "Text:\n{text}"
)


def llm_extract(
    text: str,
    llm_name: str = "deepseek",
) -> dict:
    if not text or len(text.strip()) < 20:
        return {}

    from retrieval.llm import get_llm

    llm = get_llm(llm_name)

    truncated = text[:2000]
    prompt = EXTRACT_PROMPT.format(
        text=truncated
    )

    try:
        response = llm.generate(prompt=prompt)
        return _parse_response(response)
    except Exception as e:
        print(f"  [LLM] Extraction error: {e}")
        return {}


def _parse_response(response: str) -> dict:
    response = response.strip()

    json_match = re.search(
        r"\{[\s\S]*\}", response
    )
    if json_match:
        response = json_match.group()

    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return {}

    result = {}

    topics = data.get("topics", [])
    if isinstance(topics, list):
        result["topics"] = canonicalize_topics(
            [str(t) for t in topics if t]
        )

    companies = data.get("companies", [])
    if isinstance(companies, list):
        result["companies"] = [
            str(c).strip()
            for c in companies if c
        ]

    models = data.get("models", [])
    if isinstance(models, list):
        result["models"] = [
            str(m).strip()
            for m in models if m
        ]

    sentiment = data.get("sentiment", "")
    if sentiment in (
        "positive", "negative", "neutral"
    ):
        result["sentiment"] = sentiment

    score = data.get("sentiment_score", None)
    if isinstance(score, (int, float)):
        result["sentiment_score"] = round(
            float(score), 4
        )

    summary = data.get("summary", "")
    if isinstance(summary, str) and summary:
        result["summary"] = summary[:200]

    return result


def enrich_post(
    post: dict,
    llm_name: str = "deepseek",
) -> dict:
    title = post.get("title") or ""
    body = post.get("body") or ""
    text = f"{title}\n\n{body}".strip()

    extracted = llm_extract(text, llm_name)
    if not extracted:
        return post

    existing_topics = set(
        post.get("topics", [])
    )
    for t in extracted.get("topics", []):
        existing_topics.add(t)
    post["topics"] = sorted(existing_topics)

    ext = post.get("extracted", {})

    existing_companies = set(
        ext.get("mentioned_companies", [])
    )
    for c in extracted.get("companies", []):
        existing_companies.add(c)
    ext["mentioned_companies"] = sorted(
        existing_companies
    )

    existing_models = set(
        ext.get("mentioned_models", [])
    )
    for m in extracted.get("models", []):
        existing_models.add(m)
    ext["mentioned_models"] = sorted(
        existing_models
    )

    post["extracted"] = ext

    if "sentiment" in extracted:
        post["llm_sentiment"] = (
            extracted["sentiment"]
        )
    if "sentiment_score" in extracted:
        post["llm_sentiment_score"] = (
            extracted["sentiment_score"]
        )
    if "summary" in extracted:
        post["summary"] = extracted["summary"]

    return post
