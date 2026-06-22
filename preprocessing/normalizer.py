import json
import os


OUTPUT_DIR = "data/normalized"


class Normalizer:

    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def normalize(
        self,
        cleaned_posts: list[dict],
        drop_out_of_range: bool = True,
    ) -> tuple[list[dict], list[dict]]:
        posts = []
        comments = []
        seen_post_ids = set()

        for post in cleaned_posts:
            tw = post.get(
                "time_window", "UNKNOWN"
            )
            if drop_out_of_range and (
                tw == "OUT_OF_RANGE"
            ):
                continue

            post_id = post.get("post_id")
            if post_id and post_id in seen_post_ids:
                continue
            if post_id:
                seen_post_ids.add(post_id)

            normalized_post = {
                "post_id": post_id,
                "author": post.get("author"),
                "subreddit": post.get("subreddit"),
                "title": post.get("title"),
                "body": post.get("body"),
                "created_at": post.get(
                    "created_at"
                ),
                "time_window": post.get(
                    "time_window"
                ),
                "url": post.get("url"),
                "permalink": post.get("permalink"),
                "topics": post.get("topics", []),
                "comment_count": post.get(
                    "comment_count", 0
                ),
                "extracted": post.get(
                    "extracted", {}
                ),
            }
            posts.append(normalized_post)

            seen_comment_ids = set()
            for comment in post.get(
                "comments", []
            ):
                cid = comment.get("comment_id")
                if cid and cid in seen_comment_ids:
                    continue
                if cid:
                    seen_comment_ids.add(cid)

                normalized_comment = {
                    "comment_id": cid,
                    "post_id": comment.get(
                        "post_id"
                    ),
                    "parent_id": comment.get(
                        "parent_id"
                    ),
                    "author": comment.get(
                        "author"
                    ),
                    "body": comment.get("body"),
                    "created_at": comment.get(
                        "created_at"
                    ),
                    "depth": comment.get(
                        "depth", 0
                    ),
                    "extracted": comment.get(
                        "extracted", {}
                    ),
                }
                comments.append(
                    normalized_comment
                )

        return posts, comments

    def save(
        self,
        posts: list[dict],
        comments: list[dict],
    ) -> tuple[str, str]:
        posts_file = f"{OUTPUT_DIR}/posts.jsonl"
        comments_file = (
            f"{OUTPUT_DIR}/comments.jsonl"
        )

        with open(
            posts_file, "w", encoding="utf-8"
        ) as f:
            for post in posts:
                f.write(
                    json.dumps(
                        post, ensure_ascii=False
                    )
                    + "\n"
                )

        with open(
            comments_file, "w", encoding="utf-8"
        ) as f:
            for comment in comments:
                f.write(
                    json.dumps(
                        comment,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"[+] Normalized {len(posts)} posts "
            f"-> {posts_file}"
        )
        print(
            f"[+] Normalized {len(comments)} "
            f"comments -> {comments_file}"
        )

        return posts_file, comments_file

    def quality_report(
        self,
        posts: list[dict],
        comments: list[dict],
    ) -> dict:
        authors = set()
        subreddits = set()
        missing_timestamps = 0
        missing_authors = 0
        time_windows = {}
        all_companies = set()
        all_models = set()
        all_topics = {}
        nested_comments = 0

        for post in posts:
            if (
                post.get("author")
                and post["author"] != "Unknown"
            ):
                authors.add(post["author"])
            else:
                missing_authors += 1

            if not post.get("created_at"):
                missing_timestamps += 1

            if post.get("subreddit"):
                subreddits.add(post["subreddit"])

            tw = post.get(
                "time_window", "UNKNOWN"
            )
            time_windows[tw] = (
                time_windows.get(tw, 0) + 1
            )

            for topic in post.get("topics", []):
                all_topics[topic] = (
                    all_topics.get(topic, 0) + 1
                )

            extracted = post.get("extracted", {})
            for c in extracted.get(
                "mentioned_companies", []
            ):
                all_companies.add(c)
            for m in extracted.get(
                "mentioned_models", []
            ):
                all_models.add(m)

        for comment in comments:
            if (
                comment.get("author")
                and comment["author"] != "Unknown"
            ):
                authors.add(comment["author"])
            else:
                missing_authors += 1

            if not comment.get("created_at"):
                missing_timestamps += 1

            if comment.get("depth", 0) > 0:
                nested_comments += 1

        report = {
            "posts": len(posts),
            "comments": len(comments),
            "nested_comments": nested_comments,
            "flat_comments": (
                len(comments) - nested_comments
            ),
            "unique_authors": len(authors),
            "subreddits": sorted(subreddits),
            "subreddit_count": len(subreddits),
            "missing_timestamps": missing_timestamps,
            "missing_authors": missing_authors,
            "time_windows": time_windows,
            "topics": dict(
                sorted(
                    all_topics.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
            "companies_found": sorted(
                all_companies
            ),
            "models_found": sorted(all_models),
        }

        report_file = (
            f"{OUTPUT_DIR}/quality_report.json"
        )
        with open(
            report_file, "w", encoding="utf-8"
        ) as f:
            json.dump(
                report,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n{'=' * 40}")
        print("DATA QUALITY REPORT")
        print(f"{'=' * 40}")
        print(f"Posts:              {report['posts']}")
        print(f"Comments:          {report['comments']}")
        print(f"  Nested:          {report['nested_comments']}")
        print(f"  Flat:            {report['flat_comments']}")
        print(f"Unique authors:    {report['unique_authors']}")
        print(f"Subreddits:        {report['subreddit_count']}")
        print(f"Missing timestamps: {report['missing_timestamps']}")
        print(f"Missing authors:   {report['missing_authors']}")
        print(f"Time windows:      {report['time_windows']}")
        print(f"Topics:            {report['topics']}")
        print(f"Companies:         {report['companies_found']}")
        print(f"Models:            {report['models_found']}")
        print(f"{'=' * 40}")
        print(f"Report -> {report_file}")

        return report
