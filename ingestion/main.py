import asyncio
import sys

from ingestion.search import (
    search_reddit_urls,
    search_all_topics,
    DEFAULT_TOPICS,
)
from ingestion.scraper import RedditScraper
from ingestion.parser import RedditParser


def _parse_flag(prefix: str) -> str | None:
    for arg in sys.argv:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    return None


def _parse_topics() -> list[str] | None:
    raw = _parse_flag("--topics=")
    if raw:
        return [
            t.strip() for t in raw.split(",")
            if t.strip()
        ]
    return None


def _parse_year() -> int | None:
    raw = _parse_flag("--year=")
    if raw and raw.isdigit():
        return int(raw)
    return None


async def main():
    custom_topics = _parse_topics()
    year_filter = _parse_year()

    if (
        "--all" in sys.argv
        or custom_topics
        or year_filter
    ):
        topics = custom_topics or DEFAULT_TOPICS
        await run_all_topics(topics, year_filter)
    else:
        await run_single_query()


async def run_single_query():
    query = input("Enter your search query: ")

    print(
        f"\n[1/4] Searching Reddit URLs..."
    )
    results = await search_reddit_urls(
        query,
        num_results=50,
    )
    print(f"      Found {len(results)} URLs")

    if not results:
        print(
            "No Reddit URLs found. "
            "Try a different query."
        )
        return

    for r in results:
        print(f"  - {r['url']}")

    urls = [r["url"] for r in results]

    print(
        f"\n[2/4] Scraping {len(urls)} URLs..."
    )
    scraper = RedditScraper()
    scraped = await scraper.scrape_urls(
        urls, search_metadata=results
    )

    print("\n[3/4] Saving raw data...")
    raw_file = scraper.save_results(
        query, scraped
    )

    print(
        "\n[4/4] Parsing into structured data..."
    )
    parser = RedditParser()
    parsed = parser.parse_scraped_file(raw_file)
    parser.save_parsed(parsed, raw_file)

    print(f"\nDone. {len(parsed)} posts parsed.")


async def run_all_topics(
    topics: list[str],
    year_filter: int = None,
):
    year_label = (
        f" (year={year_filter})"
        if year_filter else ""
    )
    print(
        f"\n[1/4] Searching {len(topics)} "
        f"topics{year_label}..."
    )
    for t in topics:
        print(f"  - {t}")

    results = await search_all_topics(
        topics=topics,
        use_time_variants=True,
        year_filter=year_filter,
    )

    if not results:
        print("No Reddit URLs found.")
        return

    urls = [r["url"] for r in results]

    print(
        f"\n[2/4] Scraping {len(urls)} URLs..."
    )
    scraper = RedditScraper()
    scraped = await scraper.scrape_urls(
        urls, search_metadata=results
    )

    print("\n[3/4] Saving raw data...")
    slug = "all_topics"
    if year_filter:
        slug = f"all_topics_{year_filter}"
    raw_file = scraper.save_results(
        slug, scraped
    )

    print(
        "\n[4/4] Parsing into structured data..."
    )
    parser = RedditParser()
    parsed = parser.parse_scraped_file(raw_file)
    parser.save_parsed(parsed, raw_file)

    print(f"\nDone. {len(parsed)} posts parsed.")


if __name__ == "__main__":
    asyncio.run(main())
