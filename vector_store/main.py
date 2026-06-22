import json
import sys

from vector_store.embeddings import (
    get_embedding_model,
)
from vector_store.store import VectorStore


POSTS_FILE = "data/normalized/posts.jsonl"
COMMENTS_FILE = "data/normalized/comments.jsonl"


def load_jsonl(filepath: str) -> list[dict]:
    items = []
    with open(
        filepath, "r", encoding="utf-8"
    ) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def ingest(embedding_name: str):
    print(
        f"[1/4] Loading embedding model "
        f"({embedding_name})..."
    )
    model = get_embedding_model(embedding_name)
    store = VectorStore(model)

    print(f"[2/4] Loading normalized data...")
    posts = load_jsonl(POSTS_FILE)
    comments = load_jsonl(COMMENTS_FILE)
    print(
        f"      {len(posts)} posts, "
        f"{len(comments)} comments"
    )

    print(f"[3/4] Embedding and storing posts...")
    posts_added = store.add_posts(posts)

    print(
        f"[4/4] Embedding and storing comments..."
    )
    comments_added = store.add_comments(comments)

    stats = store.stats()
    print(f"\nDone.")
    print(f"  Posts in store:    {stats['posts']}")
    print(
        f"  Comments in store: {stats['comments']}"
    )


def search(
    query: str,
    embedding_name: str,
    n: int = 5,
):
    model = get_embedding_model(embedding_name)
    store = VectorStore(model)

    print(f"Query: {query}\n")

    print("--- Post Results ---")
    results = store.query_posts(
        query, n_results=n
    )
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        score = 1 - dist
        print(
            f"\n[{i+1}] score={score:.3f} "
            f"r/{meta.get('subreddit','')} "
            f"({meta.get('time_window','')})"
        )
        print(f"    {doc[:150]}...")

    print("\n--- Comment Results ---")
    results = store.query_comments(
        query, n_results=n
    )
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        score = 1 - dist
        print(
            f"\n[{i+1}] score={score:.3f} "
            f"u/{meta.get('author','')} "
            f"depth={meta.get('depth',0)}"
        )
        print(f"    {doc[:150]}...")


if __name__ == "__main__":
    embedding = "sentence-transformer"

    for arg in sys.argv:
        if arg.startswith("--embed="):
            embedding = arg.split("=")[1]

    if "--search" in sys.argv:
        query = input("Enter search query: ")
        search(query, embedding)
    else:
        ingest(embedding)
