import json
import os
import sys

from preprocessing.cleaner import clean_post
from preprocessing.canonicalize import (
    canonicalize_topics,
)
from preprocessing.normalizer import Normalizer


PARSED_DIR = "data/parsed"
CLEANED_DIR = "data/cleaned"


def preprocess_file(
    filepath: str,
    use_llm: bool = False,
    llm_name: str = "deepseek",
) -> list[dict]:
    os.makedirs(CLEANED_DIR, exist_ok=True)

    with open(
        filepath, "r", encoding="utf-8"
    ) as f:
        posts = json.load(f)

    cleaned = []
    skipped = 0

    for post in posts:
        result = clean_post(post)
        if (
            not result["title"]
            and not result["body"]
        ):
            skipped += 1
            continue

        result["topics"] = canonicalize_topics(
            result.get("topics", [])
        )

        cleaned.append(result)

    if use_llm:
        from preprocessing.llm_extractor import (
            enrich_post,
        )
        print(
            f"[+] LLM extracting entities "
            f"from {len(cleaned)} posts "
            f"({llm_name})..."
        )
        for i, post in enumerate(cleaned):
            print(
                f"  [{i+1}/{len(cleaned)}] "
                f"{post.get('title', '')[:50]}..."
            )
            enrich_post(post, llm_name)

    basename = os.path.basename(filepath)
    output_file = f"{CLEANED_DIR}/{basename}"

    with open(
        output_file, "w", encoding="utf-8"
    ) as f:
        json.dump(
            cleaned,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"[+] Cleaned {len(cleaned)} posts, "
        f"skipped {skipped} "
        f"-> {output_file}"
    )
    return cleaned


def preprocess_all(
    use_llm: bool = False,
    llm_name: str = "deepseek",
):
    if not os.path.exists(PARSED_DIR):
        print(f"No parsed data in {PARSED_DIR}")
        return

    files = [
        f
        for f in os.listdir(PARSED_DIR)
        if f.endswith(".json")
    ]

    if not files:
        print(f"No JSON files in {PARSED_DIR}")
        return

    print(
        f"[+] Found {len(files)} files "
        f"to preprocess\n"
    )

    all_cleaned = []

    for filename in files:
        filepath = f"{PARSED_DIR}/{filename}"
        cleaned = preprocess_file(
            filepath, use_llm, llm_name
        )
        all_cleaned.extend(cleaned)

    print(
        f"\n[+] Normalizing "
        f"{len(all_cleaned)} posts..."
    )

    normalizer = Normalizer()
    posts, comments = normalizer.normalize(
        all_cleaned
    )
    normalizer.save(posts, comments)
    normalizer.quality_report(posts, comments)


if __name__ == "__main__":
    use_llm = "--llm-extract" in sys.argv
    llm_name = "deepseek"

    for arg in sys.argv:
        if arg.startswith("--llm="):
            llm_name = arg.split("=")[1]

    args = [
        a for a in sys.argv[1:]
        if not a.startswith("--")
    ]

    if args:
        cleaned = preprocess_file(
            args[0], use_llm, llm_name
        )
        normalizer = Normalizer()
        posts, comments = normalizer.normalize(
            cleaned
        )
        normalizer.save(posts, comments)
        normalizer.quality_report(
            posts, comments
        )
    else:
        preprocess_all(use_llm, llm_name)
