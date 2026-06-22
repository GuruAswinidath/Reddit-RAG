import os
import re
from datetime import datetime
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()


TARGET_SUBREDDITS = [
    "OpenAI",
    "ClaudeAI",
    "GeminiAI",
    "LocalLLaMA",
    "MachineLearning",
]

TOPICS = [
    "RAG",
    "AI Safety",
    "GPT-4o",
    "Claude",
    "Gemini",
    "Open Source LLM",
    "Agentic AI",
    "Vector Database",
]

TIME_VARIANTS = ["2025", "2026"]

POST_URL_PATTERN = re.compile(
    r"reddit\.com/r/\w+/comments/\w+"
)


def _get_post_id(url: str) -> str | None:
    match = re.search(
        r"/comments/([^/\?]+)", url
    )
    return match.group(1) if match else None


def _is_valid_post_url(url: str) -> bool:
    return bool(POST_URL_PATTERN.search(url))


def _deduplicate_urls(
    urls: list[dict],
) -> list[dict]:
    seen_ids = set()
    unique = []

    for entry in urls:
        post_id = _get_post_id(entry["url"])
        if post_id and post_id not in seen_ids:
            seen_ids.add(post_id)
            entry["post_id"] = post_id
            unique.append(entry)

    return unique


def _build_subreddit_query(query: str) -> str:
    subreddit_filter = " OR ".join(
        f"r/{sub}" for sub in TARGET_SUBREDDITS
    )
    return f"{query} ({subreddit_filter})"


def _filter_subreddit_urls(
    urls: list[str],
) -> list[str]:
    pattern = re.compile(
        r"reddit\.com/r/("
        + "|".join(TARGET_SUBREDDITS)
        + r")/",
        re.IGNORECASE,
    )
    return [
        url for url in urls
        if pattern.search(url)
    ]


def _build_time_queries(
    topic: str,
) -> list[str]:
    queries = [topic]
    for year in TIME_VARIANTS:
        queries.append(f"{topic} {year}")
    return queries


# -----------------------------------------
# Single topic search
# -----------------------------------------

async def search_reddit_urls(
    query: str,
    num_results: int = 50,
    engine: str = "tavily",
    restrict_subreddits: bool = True,
) -> list[dict]:
    if engine == "google_cse":
        raw_urls = await _google_cse_search(
            query, num_results
        )
    else:
        raw_urls = _tavily_search(
            query, num_results,
            restrict_subreddits,
        )

    now = datetime.now().isoformat()

    results = []
    for url in raw_urls:
        if not _is_valid_post_url(url):
            continue
        results.append({
            "url": url,
            "search_topic": query,
            "search_engine": engine,
            "searched_at": now,
        })

    return _deduplicate_urls(results)


# -----------------------------------------
# Multi-topic search (all topics + time)
# -----------------------------------------

async def search_all_topics(
    num_results: int = 50,
    engine: str = "tavily",
    restrict_subreddits: bool = True,
    use_time_variants: bool = True,
) -> list[dict]:
    all_results = []

    for topic in TOPICS:
        if use_time_variants:
            queries = _build_time_queries(topic)
        else:
            queries = [topic]

        for query in queries:
            print(f"  Searching: {query}")

            topic_results = (
                await search_reddit_urls(
                    query=query,
                    num_results=num_results,
                    engine=engine,
                    restrict_subreddits=(
                        restrict_subreddits
                    ),
                )
            )

            for r in topic_results:
                r["search_topic"] = topic

            all_results.extend(topic_results)
            print(
                f"    Found {len(topic_results)} "
                f"URLs"
            )

    deduped = _deduplicate_urls(all_results)

    print(
        f"\n  Total unique posts: {len(deduped)}"
    )
    return deduped


# -----------------------------------------
# Tavily
# -----------------------------------------

def _tavily_search(
    query: str,
    num_results: int,
    restrict_subreddits: bool,
) -> list[str]:
    from tavily import TavilyClient

    client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    search_query = query
    if restrict_subreddits:
        search_query = _build_subreddit_query(
            query
        )

    response = client.search(
        query=search_query,
        max_results=num_results,
        include_domains=["reddit.com"],
    )

    urls = [
        result["url"]
        for result in response["results"]
    ]

    if restrict_subreddits:
        urls = _filter_subreddit_urls(urls)

    return urls


# -----------------------------------------
# Google CSE
# -----------------------------------------

async def _google_cse_search(
    query: str,
    num_results: int,
) -> list[str]:
    from crawl4ai import AsyncWebCrawler

    cse_id = os.getenv("GOOGLE_CSE_ID")
    encoded_query = quote_plus(query)
    url = (
        f"https://cse.google.com/cse"
        f"?cx={cse_id}&q={encoded_query}"
    )

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    reddit_urls = []
    seen = set()

    if result.links:
        all_links = (
            result.links.get("external", [])
            + result.links.get("internal", [])
        )
        for link in all_links:
            href = (
                link.get("href", "")
                if isinstance(link, dict)
                else str(link)
            )
            if (
                "reddit.com" in href
                and href not in seen
            ):
                seen.add(href)
                reddit_urls.append(href)

    if not reddit_urls and result.markdown:
        pattern = (
            r"https?://(?:www\.)?reddit\.com"
            r"/r/[^\s\)\]\"\'><,]+"
        )
        for match in re.findall(
            pattern, result.markdown
        ):
            clean = match.rstrip(".,;:!?)")
            if clean not in seen:
                seen.add(clean)
                reddit_urls.append(clean)

    return reddit_urls[:num_results]
