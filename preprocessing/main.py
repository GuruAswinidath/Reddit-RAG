import json
import os
import sys

from preprocessing.cleaner import clean_post
from preprocessing.normalizer import Normalizer


PARSED_DIR = "data/parsed"
CLEANED_DIR = "data/cleaned"


def preprocess_file(filepath: str) -> list[dict]:
    os.makedirs(CLEANED_DIR, exist_ok=True)

    with open(
        filepath, "r", encoding="utf-8"
    ) as f:
        posts = json.load(f)

    cleaned = []
    skipped = 0

    for post in posts:
        result = clean_post(post)
        if not result["title"] and not result["body"]:
            skipped += 1
            continue
        cleaned.append(result)

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


def preprocess_all():
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
        cleaned = preprocess_file(filepath)
        all_cleaned.extend(cleaned)

    print(f"\n[+] Normalizing {len(all_cleaned)} "
          f"posts...")

    normalizer = Normalizer()
    posts, comments = normalizer.normalize(
        all_cleaned
    )
    normalizer.save(posts, comments)
    normalizer.quality_report(posts, comments)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cleaned = preprocess_file(sys.argv[1])
        normalizer = Normalizer()
        posts, comments = normalizer.normalize(
            cleaned
        )
        normalizer.save(posts, comments)
        normalizer.quality_report(posts, comments)
    else:
        preprocess_all()
