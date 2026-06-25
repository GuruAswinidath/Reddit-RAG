import os
import re
import time as _time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()


TARGET_SUBREDDITS = [
    "OpenAI",
    "ClaudeAI",
    "GeminiAI",
    "LocalLLaMA",
    "MachineLearning",
]

DEFAULT_TOPICS = [
    "RAG",
    "AI Safety",
    "GPT-4o",
    "Claude",
    "Gemini",
    "Open Source LLM",
    "Agentic AI",
    "Vector Database",
]

TIME_VARIANTS = ["2023", "2024", "2025", "2026"]

POST_URL_PATTERN = re.compile(
    r"reddit\.com/r/\w+/comments/\w+"
)

REDDIT_HEADERS = {
    "User-Agent": (
        "reddit-rag-scraper/1.0 "
        "(research project)"
    ),
}

REDDIT_RATE_DELAY = 2.0


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


def _year_bounds(year: int):
    start = datetime(
        year, 1, 1, tzinfo=timezone.utc
    ).timestamp()
    end = datetime(
        year + 1, 1, 1, tzinfo=timezone.utc
    ).timestamp()
    return start, end


# -----------------------------------------
# Single topic search (Tavily)
# -----------------------------------------

async def search_reddit_urls(
    query: str,
    num_results: int = 20,
) -> list[dict]:
    raw_urls = _tavily_search(
        query, min(num_results, 20),
    )

    now = datetime.now().isoformat()

    results = []
    for url in raw_urls:
        if not _is_valid_post_url(url):
            continue
        results.append({
            "url": url,
            "search_topic": query,
            "search_engine": "tavily",
            "searched_at": now,
        })

    return _deduplicate_urls(results)


# -----------------------------------------
# Multi-topic search (Tavily + Reddit API)
# -----------------------------------------

async def search_all_topics(
    topics: list[str] = None,
    num_results: int = 20,
    restrict_subreddits: bool = False,
    use_time_variants: bool = True,
    year_filter: int = None,
    max_pages: int = 3,
) -> list[dict]:
    _ = num_results
    if topics is None:
        topics = DEFAULT_TOPICS

    subreddits = (
        TARGET_SUBREDDITS
        if restrict_subreddits
        else ["all"]
    )

    all_results = []

    # --- Phase 1: Tavily (current/trending) ---
    print("\n  Phase 1: Tavily search "
          "(current/trending)...")

    for topic in topics:
        queries = [topic]
        if year_filter:
            queries.append(
                f"{topic} {year_filter}"
            )
        elif use_time_variants:
            for year in TIME_VARIANTS:
                queries.append(
                    f"{topic} {year}"
                )

        for query in queries:
            for sub in subreddits:
                full_q = (
                    f"{query} r/{sub}"
                    if sub != "all"
                    else query
                )
                results = (
                    await search_reddit_urls(
                        full_q, num_results=20,
                    )
                )
                for r in results:
                    r["search_topic"] = topic
                all_results.extend(results)

        tavily_count = len(
            _deduplicate_urls(
                list(all_results)
            )
        )
        print(
            f"    {topic}: running total "
            f"{tavily_count} unique"
        )

    # --- Phase 2: Reddit API (historical) ---
    year_label = (
        f" (year={year_filter})"
        if year_filter else ""
    )
    print(
        f"\n  Phase 2: Reddit JSON API "
        f"(historical + paginated)"
        f"{year_label}..."
    )

    for topic in topics:
        for sub in subreddits:
            reddit_posts = _reddit_api_search(
                query=topic,
                subreddit=sub,
                sort_options=[
                    "relevance", "top", "new",
                ],
                time_filters=["all", "year"],
                max_pages=max_pages,
                year_filter=year_filter,
            )

            now = datetime.now().isoformat()
            for post in reddit_posts:
                all_results.append({
                    "url": post["url"],
                    "search_topic": topic,
                    "search_engine": "reddit_api",
                    "searched_at": now,
                    "post_id": post.get(
                        "post_id"
                    ),
                    "api_title": post.get(
                        "title"
                    ),
                    "api_author": post.get(
                        "author"
                    ),
                    "api_created_at": post.get(
                        "created_at"
                    ),
                    "api_score": post.get(
                        "score"
                    ),
                    "api_num_comments": post.get(
                        "num_comments"
                    ),
                    "api_subreddit": post.get(
                        "subreddit"
                    ),
                })

        reddit_count = len(
            _deduplicate_urls(
                list(all_results)
            )
        )
        print(
            f"    {topic}: running total "
            f"{reddit_count} unique"
        )

    deduped = _deduplicate_urls(all_results)

    if year_filter:
        print(
            f"\n  Filtered to year "
            f"{year_filter}"
        )

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
) -> list[str]:
    from tavily import TavilyClient

    client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    try:
        response = client.search(
            query=query,
            max_results=min(num_results, 20),
            search_depth="advanced",
            include_domains=["reddit.com"],
        )
    except Exception as e:
        print(f"    Tavily error: {e}")
        return []

    return [
        result["url"]
        for result in response.get(
            "results", []
        )
    ]


# -----------------------------------------
# Reddit JSON API (no auth needed)
# -----------------------------------------

def _reddit_api_search(
    query: str,
    subreddit: str,
    sort_options: list[str] = None,
    time_filters: list[str] = None,
    max_pages: int = 3,
    year_filter: int = None,
) -> list[dict]:
    if sort_options is None:
        sort_options = ["relevance", "top"]
    if time_filters is None:
        time_filters = ["all", "year"]

    yr_start, yr_end = (
        _year_bounds(year_filter)
        if year_filter else (None, None)
    )

    found = {}

    for sort in sort_options:
        for t_filter in time_filters:
            after = None

            for _page in range(max_pages):
                posts, after = (
                    _reddit_api_page(
                        query, subreddit,
                        sort, t_filter, after,
                    )
                )

                for p in posts:
                    pid = p.get("post_id")
                    if not pid or pid in found:
                        continue

                    if yr_start is not None:
                        utc = p.get(
                            "created_utc", 0
                        )
                        if (
                            utc < yr_start
                            or utc >= yr_end
                        ):
                            continue

                    found[pid] = p

                if not after:
                    break

    return list(found.values())


def _reddit_api_page(
    query: str,
    subreddit: str,
    sort: str,
    time_filter: str,
    after: str | None,
) -> tuple[list[dict], str | None]:
    url = (
        f"https://www.reddit.com"
        f"/r/{subreddit}/search.json"
    )

    params = {
        "q": query,
        "sort": sort,
        "t": time_filter,
        "limit": 100,
        "restrict_sr": (
            "true" if subreddit != "all"
            else "false"
        ),
        "type": "link",
    }
    if after:
        params["after"] = after

    try:
        _time.sleep(REDDIT_RATE_DELAY)

        resp = requests.get(
            url,
            params=params,
            headers=REDDIT_HEADERS,
            timeout=15,
        )

        if resp.status_code == 429:
            print(
                "    Rate limited, waiting 10s..."
            )
            _time.sleep(10)
            resp = requests.get(
                url,
                params=params,
                headers=REDDIT_HEADERS,
                timeout=15,
            )

        if resp.status_code != 200:
            return [], None

        data = resp.json()
        children = (
            data.get("data", {})
            .get("children", [])
        )

        posts = []
        for child in children:
            d = child.get("data", {})
            permalink = d.get("permalink", "")
            if not permalink:
                continue

            full_url = (
                f"https://www.reddit.com"
                f"{permalink}"
            )
            if not _is_valid_post_url(full_url):
                continue

            created_utc = d.get(
                "created_utc", 0
            )
            created_at = (
                datetime.fromtimestamp(
                    created_utc,
                    tz=timezone.utc,
                ).isoformat()
                if created_utc else ""
            )

            posts.append({
                "url": full_url,
                "post_id": _get_post_id(
                    full_url
                ),
                "title": d.get("title", ""),
                "author": d.get("author", ""),
                "subreddit": d.get(
                    "subreddit", ""
                ),
                "created_utc": created_utc,
                "created_at": created_at,
                "score": d.get("score", 0),
                "num_comments": d.get(
                    "num_comments", 0
                ),
                "selftext": d.get(
                    "selftext", ""
                )[:500],
            })

        next_after = (
            data.get("data", {}).get("after")
        )

        return posts, next_after

    except Exception as e:
        print(
            f"    Reddit API error "
            f"({subreddit}/{sort}/{time_filter})"
            f": {e}"
        )
        return [], None
