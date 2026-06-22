from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Comment:
    author: Optional[str]
    body: str


@dataclass
class ParsedPost:
    post_id: Optional[str]
    subreddit: Optional[str]
    url: str
    title: Optional[str]
    body: str
    comments: list[Comment] = field(
        default_factory=list
    )
    comment_count: int = 0
    content_markdown: str = ""
    scraped_at: str = ""
