import json
import os
from datetime import datetime

from crawl4ai import AsyncWebCrawler


OUTPUT_DIR = "data/raw"


class RedditScraper:

    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    async def scrape_urls(
        self,
        urls: list[str],
        search_metadata: list[dict] = None,
    ) -> list[dict]:
        meta_lookup = {}
        if search_metadata:
            for entry in search_metadata:
                meta_lookup[entry["url"]] = entry

        results = []

        async with AsyncWebCrawler() as crawler:
            for url in urls:
                meta = meta_lookup.get(url, {})

                try:
                    result = await crawler.arun(
                        url=url
                    )
                    data = {
                        "url": url,
                        "markdown": result.markdown,
                        "success": result.success,
                        "scraped_at": (
                            datetime.now().isoformat()
                        ),
                        "search_topic": meta.get(
                            "search_topic"
                        ),
                        "search_engine": meta.get(
                            "search_engine"
                        ),
                        "post_id": meta.get(
                            "post_id"
                        ),
                    }
                    results.append(data)
                    print(f"[+] Scraped: {url}")

                except Exception as e:
                    print(
                        f"[-] Failed: {url} — {e}"
                    )
                    results.append({
                        "url": url,
                        "markdown": None,
                        "success": False,
                        "error": str(e),
                        "scraped_at": (
                            datetime.now().isoformat()
                        ),
                        "search_topic": meta.get(
                            "search_topic"
                        ),
                        "search_engine": meta.get(
                            "search_engine"
                        ),
                        "post_id": meta.get(
                            "post_id"
                        ),
                    })

        return results

    def save_results(
        self,
        query: str,
        results: list[dict],
    ) -> str:
        slug = (
            query.lower()
            .replace(" ", "_")[:50]
        )
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        filename = (
            f"{OUTPUT_DIR}/{slug}_{timestamp}.jsonl"
        )

        with open(
            filename, "w", encoding="utf-8"
        ) as f:
            for result in results:
                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"[+] Saved {len(results)} results "
            f"to {filename}"
        )
        return filename
