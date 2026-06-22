import asyncio
import sys

from ingestion.search import (
    search_reddit_urls,
    search_all_topics,
)
from ingestion.scraper import RedditScraper
from ingestion.parser import RedditParser


async def main():
    engine = "tavily"
    if "--google" in sys.argv:
        engine = "google_cse"

    if "--all" in sys.argv:
        await run_all_topics(engine)
    else:
        await run_single_query(engine)


async def run_single_query(engine: str):
    query = input("Enter your search query: ")

    print(
        f"\n[1/4] Searching Reddit URLs "
        f"({engine})..."
    )
    results = await search_reddit_urls(
        query,
        num_results=50,
        engine=engine,
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


async def run_all_topics(engine: str):
    print(
        f"\n[1/4] Searching all topics "
        f"({engine})..."
    )
    results = await search_all_topics(
        num_results=50,
        engine=engine,
        use_time_variants=True,
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
    raw_file = scraper.save_results(
        "all_topics", scraped
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
