import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


POSTS_FILE = "data/normalized/posts.jsonl"
COMMENTS_FILE = "data/normalized/comments.jsonl"


@dataclass
class RetrievedDoc:
    id: str
    text: str
    score: float
    metadata: dict
    source: str


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> list[RetrievedDoc]:
        pass


def load_corpus() -> tuple[list[dict], list[dict]]:
    posts = []
    comments = []

    if os.path.exists(POSTS_FILE):
        with open(
            POSTS_FILE, "r", encoding="utf-8"
        ) as f:
            for line in f:
                post = json.loads(line)
                title = post.get("title") or ""
                body = post.get("body") or ""
                text = f"{title}\n\n{body}".strip()
                if not text:
                    continue
                posts.append({
                    "id": post.get("post_id", ""),
                    "text": text,
                    "source": "post",
                    "metadata": {
                        "post_id": (
                            post.get("post_id") or ""
                        ),
                        "subreddit": (
                            post.get("subreddit") or ""
                        ),
                        "author": (
                            post.get("author") or ""
                        ),
                        "time_window": (
                            post.get("time_window")
                            or ""
                        ),
                        "created_at": (
                            post.get("created_at")
                            or ""
                        ),
                        "url": (
                            post.get("url") or ""
                        ),
                        "topics": ", ".join(
                            post.get("topics", [])
                        ),
                        "comment_count": (
                            post.get("comment_count")
                            or 0
                        ),
                    },
                })

    if os.path.exists(COMMENTS_FILE):
        with open(
            COMMENTS_FILE, "r", encoding="utf-8"
        ) as f:
            for line in f:
                comment = json.loads(line)
                body = comment.get("body") or ""
                if len(body) < 10:
                    continue
                comments.append({
                    "id": (
                        comment.get("comment_id")
                        or ""
                    ),
                    "text": body,
                    "source": "comment",
                    "metadata": {
                        "comment_id": (
                            comment.get("comment_id")
                            or ""
                        ),
                        "post_id": (
                            comment.get("post_id")
                            or ""
                        ),
                        "parent_id": (
                            comment.get("parent_id")
                            or ""
                        ),
                        "author": (
                            comment.get("author")
                            or ""
                        ),
                        "created_at": (
                            comment.get("created_at")
                            or ""
                        ),
                        "time_window": (
                            comment.get("time_window")
                            or ""
                        ),
                        "depth": (
                            comment.get("depth", 0)
                        ),
                    },
                })

    return posts, comments


def apply_where_filter(
    items: list[dict],
    where: dict | None,
) -> list[dict]:
    if not where:
        return items

    filtered = []
    for item in items:
        meta = item.get("metadata", {})
        match = all(
            meta.get(k) == v
            for k, v in where.items()
        )
        if match:
            filtered.append(item)

    return filtered
