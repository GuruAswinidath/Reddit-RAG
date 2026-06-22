import json
import os
import re

OUTPUT_DIR = "data/parsed"

NOISE_KEYWORDS = [
    "Log In", "Sign Up", "Get the Reddit app",
    "Reddit Inc", "User Agreement",
    "Privacy Policy", "Content Policy",
    "Cookie Notice", "Advertise on Reddit",
    "Download the official Reddit app",
    "Open sort options", "Best Comments",
    "Top Comments", "Home", "Popular",
    "Explore", "Terms", "View in App",
    "Continue with Google", "Continue with Email",
    "Continue with Apple", "Accessibility",
    "All rights reserved", "Reddit Premium",
    "Reddit Coins", "Back to Top",
    "About Community", "Community Details",
]

# Lines containing these are avatar/asset noise
ASSET_NOISE = [
    "redditstatic.com",
    "redditmedia.com",
    "styles.redditmedia",
    "preview.redd.it",
    "avatar",
]

# Reddit post flair tags to strip from body
FLAIR_PATTERN = re.compile(
    r"^(?:Discussion|Question\s*\|?\s*Help|"
    r"Resources|News|Tutorial|Project|"
    r"Showcase|Research|Funny|Meme|"
    r"Question|Help|OC|Rumor|Opinion)\s*\n",
    flags=re.IGNORECASE | re.MULTILINE,
)

USERNAME_LINK_PATTERN = re.compile(
    r"\[\s*([\w_-]+)\s*\]\s*\("
    r"https?://(?:www\.)?reddit\.com/user/"
)

COMMENT_ID_PATTERN = re.compile(
    r"comment/(\w+)/"
)

RELATIVE_TIME_IN_LINK = re.compile(
    r"\[\s*(\d+\s*(?:mo|yr|hr|min|sec|"
    r"months?|years?|hours?|minutes?|"
    r"days?|weeks?|seconds?|"
    r"d|w|h|m|y)\s*ago)\s*\]",
    flags=re.IGNORECASE,
)

MORE_REPLIES_PATTERN = re.compile(
    r"[Mm]ore\s+repl(?:y|ies)"
)

AD_MARKERS = [
    "Promoted", "Learn More",
    "Sponsored", "Ad •",
]


class RedditParser:

    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def parse_scraped_file(
        self, filepath: str
    ) -> list[dict]:
        results = []

        with open(
            filepath, "r", encoding="utf-8"
        ) as f:
            for line in f:
                entry = json.loads(line)
                if (
                    not entry.get("success")
                    or not entry.get("markdown")
                ):
                    print(
                        f"[-] Skipped (no content):"
                        f" {entry.get('url')}"
                    )
                    continue

                parsed = self._parse_entry(entry)
                if parsed:
                    results.append(parsed)

        return results

    def _parse_entry(self, entry: dict) -> dict:
        url = entry["url"]
        markdown = entry["markdown"]

        subreddit, post_id = self._parse_url(url)
        cleaned = self._clean_markdown(markdown)
        title, body, comments_raw = (
            self._split_content(cleaned)
        )

        body = FLAIR_PATTERN.sub("", body).strip()

        comments = self._parse_comments(
            comments_raw, post_id or "unknown"
        )

        return {
            "post_id": post_id,
            "subreddit": subreddit,
            "url": url,
            "title": title,
            "body": body,
            "comments": comments,
            "comment_count": len(comments),
            "content_markdown": cleaned,
            "scraped_at": entry.get("scraped_at"),
        }

    def _parse_url(self, url: str) -> tuple:
        match = re.search(
            r"reddit\.com/r/(\w+)"
            r"/comments/(\w+)",
            url,
        )
        if match:
            return (
                match.group(1),
                match.group(2),
            )

        match = re.search(
            r"reddit\.com/r/(\w+)", url
        )
        if match:
            return match.group(1), None

        return None, None

    def _clean_markdown(
        self, markdown: str
    ) -> str:
        lines = markdown.split("\n")
        cleaned = []

        for line in lines:
            stripped = line.strip()

            if not cleaned and not stripped:
                continue

            if any(
                noise in stripped
                for noise in NOISE_KEYWORDS
            ):
                continue

            if re.match(
                r"^\[?(Log In|Sign Up|Get app)",
                stripped,
            ):
                continue

            if any(
                ad in stripped
                for ad in AD_MARKERS
            ):
                continue

            cleaned.append(line)

        text = "\n".join(cleaned)
        text = re.sub(
            r"\n{4,}", "\n\n\n", text
        )
        return text.strip()

    def _split_content(
        self, markdown: str
    ) -> tuple[str, str, str]:
        lines = markdown.split("\n")

        title = None
        title_idx = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip(
                    "#"
                ).strip()
                title_idx = i
                break

        comments_idx = len(lines)
        comment_markers = [
            r"\d+\s+comments?",
            r"sort(ed)?\s+by",
            r"add a comment",
            r"comment as",
        ]

        for i in range(
            title_idx + 1, len(lines)
        ):
            stripped = lines[i].strip().lower()
            for marker in comment_markers:
                if re.search(marker, stripped):
                    comments_idx = i
                    break
            if comments_idx != len(lines):
                break

        body = "\n".join(
            lines[title_idx + 1 : comments_idx]
        ).strip()

        comments_raw = "\n".join(
            lines[comments_idx:]
        )

        return title, body, comments_raw

    def _parse_comments(
        self,
        raw: str,
        post_id: str,
    ) -> list[dict]:
        if not raw.strip():
            return []

        lines = raw.split("\n")
        comments = []
        close_markers = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Skip asset/avatar lines
            if any(
                asset in line
                for asset in ASSET_NOISE
            ):
                i += 1
                continue

            # Check for "more replies" marker
            if MORE_REPLIES_PATTERN.search(line):
                cid = COMMENT_ID_PATTERN.search(
                    line
                )
                if cid:
                    close_markers.append((
                        len(comments),
                        cid.group(1),
                    ))
                i += 1
                continue

            # Check for username link
            user_match = (
                USERNAME_LINK_PATTERN.search(line)
            )
            if not user_match:
                i += 1
                continue

            author = user_match.group(1)
            comment_id = None
            time_text = None

            # Scan next few lines for
            # comment ID and timestamp
            scan_end = min(i + 4, len(lines))
            body_start = i + 1

            for j in range(i + 1, scan_end):
                cid = COMMENT_ID_PATTERN.search(
                    lines[j]
                )
                if cid and not comment_id:
                    comment_id = cid.group(1)
                    body_start = j + 1

                tm = RELATIVE_TIME_IN_LINK.search(
                    lines[j]
                )
                if tm and not time_text:
                    time_text = tm.group(1)
                    body_start = max(
                        body_start, j + 1
                    )

            # Collect body lines until next
            # comment or marker
            body_lines = []
            for j in range(
                body_start, len(lines)
            ):
                check = lines[j]

                if USERNAME_LINK_PATTERN.search(
                    check
                ):
                    break
                if MORE_REPLIES_PATTERN.search(
                    check
                ):
                    break
                if any(
                    asset in check
                    for asset in ASSET_NOISE
                ):
                    continue

                stripped = check.strip()
                if stripped:
                    body_lines.append(stripped)

            body = " ".join(body_lines).strip()

            if body and len(body) > 5:
                comments.append({
                    "author": author,
                    "body": body,
                    "comment_id": (
                        comment_id
                        or f"{post_id}_c{len(comments)}"
                    ),
                    "time_text": time_text,
                    "depth": 0,
                    "parent_id": post_id,
                })

            i += 1

        self._assign_hierarchy(
            comments, close_markers, post_id
        )

        return comments

    def _assign_hierarchy(
        self,
        comments: list[dict],
        close_markers: list[tuple],
        post_id: str,
    ) -> None:
        """Use 'More replies' markers to build
        the comment tree.

        When we see 'More replies for X' at
        position P, all comments between X and P
        are descendants of X.
        """
        id_to_idx = {}
        for i, c in enumerate(comments):
            cid = c.get("comment_id")
            if cid:
                id_to_idx[cid] = i

        for close_pos, parent_id in close_markers:
            if parent_id not in id_to_idx:
                continue

            parent_idx = id_to_idx[parent_id]
            parent_depth = comments[
                parent_idx
            ].get("depth", 0)

            for j in range(
                parent_idx + 1,
                min(close_pos, len(comments)),
            ):
                c = comments[j]
                if c["depth"] <= parent_depth:
                    c["depth"] = parent_depth + 1
                    c["parent_id"] = parent_id

    def save_parsed(
        self,
        results: list[dict],
        source_file: str,
    ) -> str:
        basename = os.path.splitext(
            os.path.basename(source_file)
        )[0]
        output_file = (
            f"{OUTPUT_DIR}/{basename}.json"
        )

        with open(
            output_file, "w", encoding="utf-8"
        ) as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[+] Parsed {len(results)} posts "
            f"-> {output_file}"
        )
        return output_file
