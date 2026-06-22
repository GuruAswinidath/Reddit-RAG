import html
import re

from preprocessing.patterns import (
    HTML_TAG_PATTERN,
    HTML_ENTITY_PATTERN,
    MARKDOWN_BOLD_PATTERN,
    MARKDOWN_ITALIC_PATTERN,
    MARKDOWN_STRIKETHROUGH_PATTERN,
    MARKDOWN_HEADING_PATTERN,
    MARKDOWN_BLOCKQUOTE_PATTERN,
    MARKDOWN_CODE_BLOCK_PATTERN,
    MARKDOWN_INLINE_CODE_PATTERN,
    MARKDOWN_IMAGE_PATTERN,
    MARKDOWN_HR_PATTERN,
    DELETED_PATTERN,
    BOT_AUTHOR_PATTERN,
    AWARD_PATTERN,
    VOTE_NOISE_PATTERN,
    TIMESTAMP_NOISE_PATTERN,
    AD_PATTERN,
    AD_DOMAINS,
    NOISE_KEYWORDS,
    REDDIT_ASSET_PATTERNS,
    EXCESS_NEWLINES_PATTERN,
    EXCESS_SPACES_PATTERN,
    UNICODE_JUNK_PATTERN,
    EMOJI_PATTERN,
)

from preprocessing.extractor import (
    extract_all,
    extract_timestamp,
    assign_time_window,
    extract_topics,
)

FLAIR_PATTERN = re.compile(
    r"^(?:Discussion|Question\s*\|?\s*Help|"
    r"Resources|News|Tutorial|Project|"
    r"Showcase|Research|Funny|Meme|"
    r"Question|Help|OC|Rumor|Opinion)\s*\n",
    flags=re.IGNORECASE | re.MULTILINE,
)

MARKDOWN_LINK_PATTERN_SPACED = re.compile(
    r"\[([^\]]*)\]\([^\)]+\)"
)

CURLY_QUOTES = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
    "…": "...",
    "�": "'",
}


def _fix_encoding(text: str) -> str:
    for bad, good in CURLY_QUOTES.items():
        text = text.replace(bad, good)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = _fix_encoding(text)

    # Replace HTML tags with space to
    # prevent word glueing
    text = re.sub(r"><", "> <", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = HTML_ENTITY_PATTERN.sub(" ", text)

    text = MARKDOWN_IMAGE_PATTERN.sub(" ", text)
    text = MARKDOWN_CODE_BLOCK_PATTERN.sub(
        " ", text
    )
    text = MARKDOWN_INLINE_CODE_PATTERN.sub(
        r" \1 ", text
    )

    # Wrap content in spaces to prevent glueing
    text = MARKDOWN_BOLD_PATTERN.sub(
        r" \1 ", text
    )
    text = MARKDOWN_ITALIC_PATTERN.sub(
        r" \1 ", text
    )
    text = MARKDOWN_STRIKETHROUGH_PATTERN.sub(
        r" \1 ", text
    )
    text = MARKDOWN_HEADING_PATTERN.sub("", text)
    text = MARKDOWN_BLOCKQUOTE_PATTERN.sub(
        "", text
    )
    text = MARKDOWN_HR_PATTERN.sub("", text)

    text = MARKDOWN_LINK_PATTERN_SPACED.sub(
        r" \1 ", text
    )

    text = VOTE_NOISE_PATTERN.sub("", text)
    text = TIMESTAMP_NOISE_PATTERN.sub("", text)
    text = AWARD_PATTERN.sub("", text)

    text = AD_PATTERN.sub("", text)
    text = _remove_ad_domains(text)
    text = _remove_reddit_assets(text)
    text = _remove_noise_lines(text)

    text = FLAIR_PATTERN.sub("", text)

    text = UNICODE_JUNK_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)

    # Collapse whitespace LAST
    text = EXCESS_SPACES_PATTERN.sub(" ", text)
    text = EXCESS_NEWLINES_PATTERN.sub(
        "\n\n", text
    )

    return text.strip()


def _remove_ad_domains(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if any(
            domain in line.lower()
            for domain in AD_DOMAINS
        ):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _remove_reddit_assets(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if any(
            asset in line
            for asset in REDDIT_ASSET_PATTERNS
        ):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _remove_noise_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if any(
            noise in stripped
            for noise in NOISE_KEYWORDS
        ):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def is_deleted(text: str) -> bool:
    return bool(
        DELETED_PATTERN.match(text.strip())
    )


def is_bot(author: str) -> bool:
    if not author:
        return False
    return bool(BOT_AUTHOR_PATTERN.match(author))


def clean_post(post: dict) -> dict:
    scraped_at = post.get("scraped_at", "")
    raw_body = post.get("body") or ""
    raw_title = post.get("title") or ""
    raw_content = post.get(
        "content_markdown", ""
    )

    title = clean_text(raw_title)
    body = clean_text(raw_body)

    # Try timestamp from content markdown
    created_at = extract_timestamp(
        raw_content, scraped_at
    )
    time_window = assign_time_window(created_at)

    post_author = _extract_post_author(
        raw_content
    )
    if not post_author:
        post_author = "Unknown"

    body_extracted = extract_all(
        raw_body, scraped_at
    )

    full_text = f"{title} {body}"
    topics = extract_topics(full_text)

    cleaned_comments = []
    for i, comment in enumerate(
        post.get("comments", [])
    ):
        author = comment.get("author")
        comment_body = comment.get("body", "")

        if is_bot(author):
            continue
        if is_deleted(comment_body):
            continue

        cleaned_body = clean_text(comment_body)
        if len(cleaned_body) < 5:
            continue

        # Use time_text from parser if available
        time_text = comment.get("time_text", "")
        comment_created = extract_timestamp(
            time_text or comment_body, scraped_at
        )

        comment_extracted = extract_all(
            comment_body, scraped_at
        )

        post_id = (
            post.get("post_id") or "unknown"
        )

        comment_id = comment.get(
            "comment_id",
            f"{post_id}_c{i}",
        )
        parent_id = comment.get(
            "parent_id", post_id
        )
        depth = comment.get("depth", 0)

        cleaned_comments.append({
            "comment_id": comment_id,
            "post_id": post_id,
            "parent_id": parent_id,
            "author": author or "Unknown",
            "body": cleaned_body,
            "created_at": comment_created,
            "depth": depth,
            "extracted": comment_extracted,
        })

    return {
        "post_id": post.get("post_id"),
        "subreddit": post.get("subreddit"),
        "url": post.get("url"),
        "permalink": post.get("url"),
        "author": post_author,
        "title": title,
        "body": body,
        "created_at": created_at,
        "time_window": time_window,
        "topics": topics,
        "comments": cleaned_comments,
        "comment_count": len(cleaned_comments),
        "extracted": body_extracted,
        "scraped_at": scraped_at,
    }


def _extract_post_author(
    content: str,
) -> str | None:
    if not content:
        return None

    match = re.search(
        r"(?:posted|submitted|by)\s+/?u/(\w+)",
        content,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)

    match = re.search(r"u/(\w+)", content)
    if match:
        return match.group(1)

    return None
