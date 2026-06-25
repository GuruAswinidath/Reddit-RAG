import json
import os
from datetime import datetime

from neo4j import GraphDatabase

from knowledge_graph.schema import (
    CONSTRAINTS, INDEXES,
)
from knowledge_graph.sentiment import (
    analyze_sentiment,
)
from preprocessing.extractor import extract_topics


POSTS_FILE = "data/normalized/posts.jsonl"
COMMENTS_FILE = "data/normalized/comments.jsonl"
BATCH_SIZE = 200


def _get_quarter(created_at: str) -> str | None:
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(created_at)
        q = (dt.month - 1) // 3 + 1
        return f"Q{q}_{dt.year}"
    except (ValueError, TypeError):
        return None


class GraphBuilder:

    def __init__(
        self, uri: str, user: str, password: str
    ):
        self._driver = GraphDatabase.driver(
            uri, auth=(user, password)
        )
        self._driver.verify_connectivity()

        self._posts = []
        self._comments = []

    def close(self):
        self._driver.close()

    # -----------------------------------------
    # Public API
    # -----------------------------------------

    def build(self):
        self._load_data()
        self._apply_schema()
        self._create_nodes()
        self._create_relationships()
        self._run_algorithms()
        self.stats()

    def _run_algorithms(self):
        from knowledge_graph.algorithms import (
            run_all,
        )
        run_all(self._driver)

    def clear(self):
        print("[!] Clearing all graph data...")
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n) DETACH DELETE n "
                "RETURN count(n) AS deleted"
            )
            count = result.single()["deleted"]
        print(f"    Deleted {count} nodes")

    def stats(self):
        print(f"\n{'=' * 40}")
        print("GRAPH STATISTICS")
        print(f"{'=' * 40}")

        with self._driver.session() as s:
            for label in [
                "User", "Subreddit", "Post",
                "Comment", "Topic", "Company",
                "Model", "TimePeriod",
                "SentimentSnapshot",
                "TrendSnapshot",
            ]:
                result = s.run(
                    f"MATCH (n:{label}) "
                    f"RETURN count(n) AS c"
                )
                print(
                    f"  {label:15s} "
                    f"{result.single()['c']}"
                )

            print()

            for rel in [
                "AUTHORED", "POSTED_IN",
                "COMMENTS_ON", "REPLIES_TO",
                "DISCUSSES", "MENTIONS",
                "OCCURRED_IN",
                "HAS_SENTIMENT", "IN_PERIOD",
                "HAS_TREND",
            ]:
                result = s.run(
                    f"MATCH ()-[r:{rel}]->() "
                    f"RETURN count(r) AS c"
                )
                print(
                    f"  {rel:15s} "
                    f"{result.single()['c']}"
                )

        print(f"{'=' * 40}")

    # -----------------------------------------
    # Load & enrich data
    # -----------------------------------------

    def _load_data(self):
        print("[1/5] Loading normalized data...")

        self._posts = _load_jsonl(POSTS_FILE)
        self._comments = _load_jsonl(COMMENTS_FILE)

        print(
            f"      {len(self._posts)} posts, "
            f"{len(self._comments)} comments"
        )

        print("[2/5] Computing sentiment scores...")

        for post in self._posts:
            text = (
                f"{post.get('title', '')} "
                f"{post.get('body', '')}"
            )
            label, score = analyze_sentiment(text)
            post["sentiment"] = label
            post["sentiment_score"] = score
            post["quarter"] = _get_quarter(
                post.get("created_at")
            )

        for comment in self._comments:
            body = comment.get("body", "")
            label, score = analyze_sentiment(body)
            comment["sentiment"] = label
            comment["sentiment_score"] = score
            comment["quarter"] = _get_quarter(
                comment.get("created_at")
            )
            comment["topics"] = extract_topics(
                body
            )

        pos = sum(
            1 for p in self._posts
            if p["sentiment"] == "positive"
        )
        neg = sum(
            1 for p in self._posts
            if p["sentiment"] == "negative"
        )
        neu = len(self._posts) - pos - neg
        print(
            f"      Posts: {pos} positive, "
            f"{neg} negative, {neu} neutral"
        )

    # -----------------------------------------
    # Schema
    # -----------------------------------------

    def _apply_schema(self):
        print("[3/5] Applying constraints "
              "and indexes...")

        with self._driver.session() as s:
            for cypher in CONSTRAINTS + INDEXES:
                s.run(cypher)

    # -----------------------------------------
    # Nodes
    # -----------------------------------------

    def _create_nodes(self):
        print("[4/5] Creating nodes...")
        self._create_users()
        self._create_subreddits()
        self._create_topics()
        self._create_companies()
        self._create_models()
        self._create_time_periods()
        self._create_posts()
        self._create_comments()

    def _create_users(self):
        users = set()
        for p in self._posts:
            a = p.get("author")
            if a and a != "Unknown":
                users.add(a)
        for c in self._comments:
            a = c.get("author")
            if a and a != "Unknown":
                users.add(a)

        batch = [
            {"username": u} for u in users
        ]
        self._run_batch(
            "UNWIND $batch AS item "
            "MERGE (:User {username: "
            "item.username})",
            batch,
        )
        print(f"      Users: {len(batch)}")

    def _create_subreddits(self):
        subs = set()
        for p in self._posts:
            s = p.get("subreddit")
            if s:
                subs.add(s)

        batch = [{"name": s} for s in subs]
        self._run_batch(
            "UNWIND $batch AS item "
            "MERGE (:Subreddit {name: item.name})",
            batch,
        )
        print(f"      Subreddits: {len(batch)}")

    def _create_topics(self):
        topics = set()
        for p in self._posts:
            for t in p.get("topics", []):
                topics.add(t)
        for c in self._comments:
            for t in c.get("topics", []):
                topics.add(t)

        batch = [{"name": t} for t in topics]
        self._run_batch(
            "UNWIND $batch AS item "
            "MERGE (:Topic {name: item.name})",
            batch,
        )
        print(f"      Topics: {len(batch)}")

    def _create_companies(self):
        companies = set()
        for p in self._posts:
            for c in (
                p.get("extracted", {})
                .get("mentioned_companies", [])
            ):
                companies.add(c)
        for cm in self._comments:
            for c in (
                cm.get("extracted", {})
                .get("mentioned_companies", [])
            ):
                companies.add(c)

        batch = [{"name": c} for c in companies]
        self._run_batch(
            "UNWIND $batch AS item "
            "MERGE (:Company {name: item.name})",
            batch,
        )
        print(f"      Companies: {len(batch)}")

    def _create_models(self):
        models = set()
        for p in self._posts:
            for m in (
                p.get("extracted", {})
                .get("mentioned_models", [])
            ):
                models.add(m)
        for cm in self._comments:
            for m in (
                cm.get("extracted", {})
                .get("mentioned_models", [])
            ):
                models.add(m)

        batch = [{"name": m} for m in models]
        self._run_batch(
            "UNWIND $batch AS item "
            "MERGE (:Model {name: item.name})",
            batch,
        )
        print(f"      Models: {len(batch)}")

    def _create_time_periods(self):
        periods = set()
        for p in self._posts:
            q = p.get("quarter")
            if q:
                periods.add(q)
        for c in self._comments:
            q = c.get("quarter")
            if q:
                periods.add(q)

        batch = [{"period": p} for p in periods]
        self._run_batch(
            "UNWIND $batch AS item "
            "MERGE (:TimePeriod "
            "{period: item.period})",
            batch,
        )
        print(
            f"      TimePeriods: {len(batch)} "
            f"({sorted(periods)})"
        )

    def _create_posts(self):
        batch = []
        for p in self._posts:
            pid = p.get("post_id")
            if not pid:
                continue
            batch.append({
                "post_id": pid,
                "title": p.get("title") or "",
                "body": (
                    p.get("body", "")[:2000]
                ),
                "created_at": (
                    p.get("created_at") or ""
                ),
                "url": p.get("url") or "",
                "sentiment": p.get("sentiment"),
                "sentiment_score": (
                    p.get("sentiment_score")
                ),
            })

        self._run_batch(
            "UNWIND $batch AS item "
            "MERGE (p:Post "
            "{post_id: item.post_id}) "
            "SET p.title = item.title, "
            "p.body = item.body, "
            "p.created_at = item.created_at, "
            "p.url = item.url, "
            "p.sentiment = item.sentiment, "
            "p.sentiment_score = "
            "item.sentiment_score",
            batch,
        )
        print(f"      Posts: {len(batch)}")

    def _create_comments(self):
        batch = []
        for c in self._comments:
            cid = c.get("comment_id")
            if not cid:
                continue
            batch.append({
                "comment_id": cid,
                "body": (
                    c.get("body", "")[:2000]
                ),
                "created_at": (
                    c.get("created_at") or ""
                ),
                "depth": c.get("depth", 0),
                "sentiment": c.get("sentiment"),
                "sentiment_score": (
                    c.get("sentiment_score")
                ),
            })

        self._run_batch(
            "UNWIND $batch AS item "
            "MERGE (c:Comment "
            "{comment_id: item.comment_id}) "
            "SET c.body = item.body, "
            "c.created_at = item.created_at, "
            "c.depth = item.depth, "
            "c.sentiment = item.sentiment, "
            "c.sentiment_score = "
            "item.sentiment_score",
            batch,
        )
        print(f"      Comments: {len(batch)}")

    # -----------------------------------------
    # Relationships
    # -----------------------------------------

    def _create_relationships(self):
        print("[5/5] Creating relationships...")
        self._link_post_authored()
        self._link_comment_authored()
        self._link_posted_in()
        self._link_comments_on()
        self._link_replies_to()
        self._link_post_discusses()
        self._link_comment_discusses()
        self._link_post_mentions_company()
        self._link_comment_mentions_company()
        self._link_post_mentions_model()
        self._link_comment_mentions_model()
        self._link_post_occurred_in()
        self._link_comment_occurred_in()

    def _link_post_authored(self):
        batch = [
            {
                "author": p["author"],
                "post_id": p["post_id"],
            }
            for p in self._posts
            if p.get("author")
            and p["author"] != "Unknown"
            and p.get("post_id")
        ]
        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (u:User "
            "{username: item.author}) "
            "MATCH (p:Post "
            "{post_id: item.post_id}) "
            "MERGE (u)-[:AUTHORED]->(p)",
            batch,
        )
        print(f"      AUTHORED (post): {len(batch)}")

    def _link_comment_authored(self):
        batch = [
            {
                "author": c["author"],
                "comment_id": c["comment_id"],
            }
            for c in self._comments
            if c.get("author")
            and c["author"] != "Unknown"
            and c.get("comment_id")
        ]
        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (u:User "
            "{username: item.author}) "
            "MATCH (c:Comment "
            "{comment_id: item.comment_id}) "
            "MERGE (u)-[:AUTHORED]->(c)",
            batch,
        )
        print(
            f"      AUTHORED (comment): {len(batch)}"
        )

    def _link_posted_in(self):
        batch = [
            {
                "post_id": p["post_id"],
                "subreddit": p["subreddit"],
            }
            for p in self._posts
            if p.get("post_id")
            and p.get("subreddit")
        ]
        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (p:Post "
            "{post_id: item.post_id}) "
            "MATCH (s:Subreddit "
            "{name: item.subreddit}) "
            "MERGE (p)-[:POSTED_IN]->(s)",
            batch,
        )
        print(f"      POSTED_IN: {len(batch)}")

    def _link_comments_on(self):
        batch = [
            {
                "comment_id": c["comment_id"],
                "post_id": c["post_id"],
            }
            for c in self._comments
            if c.get("comment_id")
            and c.get("post_id")
        ]
        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (c:Comment "
            "{comment_id: item.comment_id}) "
            "MATCH (p:Post "
            "{post_id: item.post_id}) "
            "MERGE (c)-[:COMMENTS_ON]->(p)",
            batch,
        )
        print(f"      COMMENTS_ON: {len(batch)}")

    def _link_replies_to(self):
        post_ids = {
            p["post_id"]
            for p in self._posts
            if p.get("post_id")
        }

        batch = [
            {
                "comment_id": c["comment_id"],
                "parent_id": c["parent_id"],
            }
            for c in self._comments
            if c.get("comment_id")
            and c.get("parent_id")
            and c["parent_id"] not in post_ids
        ]
        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (c:Comment "
            "{comment_id: item.comment_id}) "
            "MATCH (p:Comment "
            "{comment_id: item.parent_id}) "
            "MERGE (c)-[:REPLIES_TO]->(p)",
            batch,
        )
        print(f"      REPLIES_TO: {len(batch)}")

    def _link_post_discusses(self):
        batch = []
        for p in self._posts:
            pid = p.get("post_id")
            if not pid:
                continue
            for topic in p.get("topics", []):
                batch.append({
                    "post_id": pid,
                    "topic": topic,
                })

        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (p:Post "
            "{post_id: item.post_id}) "
            "MATCH (t:Topic "
            "{name: item.topic}) "
            "MERGE (p)-[:DISCUSSES]->(t)",
            batch,
        )
        print(
            f"      DISCUSSES (post): {len(batch)}"
        )

    def _link_comment_discusses(self):
        batch = []
        for c in self._comments:
            cid = c.get("comment_id")
            if not cid:
                continue
            for topic in c.get("topics", []):
                batch.append({
                    "comment_id": cid,
                    "topic": topic,
                })

        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (c:Comment "
            "{comment_id: item.comment_id}) "
            "MATCH (t:Topic "
            "{name: item.topic}) "
            "MERGE (c)-[:DISCUSSES]->(t)",
            batch,
        )
        print(
            f"      DISCUSSES (comment): "
            f"{len(batch)}"
        )

    def _link_post_mentions_company(self):
        batch = []
        for p in self._posts:
            pid = p.get("post_id")
            if not pid:
                continue
            for company in (
                p.get("extracted", {})
                .get("mentioned_companies", [])
            ):
                batch.append({
                    "post_id": pid,
                    "company": company,
                })

        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (p:Post "
            "{post_id: item.post_id}) "
            "MATCH (co:Company "
            "{name: item.company}) "
            "MERGE (p)-[:MENTIONS]->(co)",
            batch,
        )
        print(
            f"      MENTIONS company (post): "
            f"{len(batch)}"
        )

    def _link_comment_mentions_company(self):
        batch = []
        for c in self._comments:
            cid = c.get("comment_id")
            if not cid:
                continue
            for company in (
                c.get("extracted", {})
                .get("mentioned_companies", [])
            ):
                batch.append({
                    "comment_id": cid,
                    "company": company,
                })

        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (c:Comment "
            "{comment_id: item.comment_id}) "
            "MATCH (co:Company "
            "{name: item.company}) "
            "MERGE (c)-[:MENTIONS]->(co)",
            batch,
        )
        print(
            f"      MENTIONS company (comment): "
            f"{len(batch)}"
        )

    def _link_post_mentions_model(self):
        batch = []
        for p in self._posts:
            pid = p.get("post_id")
            if not pid:
                continue
            for model in (
                p.get("extracted", {})
                .get("mentioned_models", [])
            ):
                batch.append({
                    "post_id": pid,
                    "model": model,
                })

        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (p:Post "
            "{post_id: item.post_id}) "
            "MATCH (m:Model "
            "{name: item.model}) "
            "MERGE (p)-[:MENTIONS]->(m)",
            batch,
        )
        print(
            f"      MENTIONS model (post): "
            f"{len(batch)}"
        )

    def _link_comment_mentions_model(self):
        batch = []
        for c in self._comments:
            cid = c.get("comment_id")
            if not cid:
                continue
            for model in (
                c.get("extracted", {})
                .get("mentioned_models", [])
            ):
                batch.append({
                    "comment_id": cid,
                    "model": model,
                })

        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (c:Comment "
            "{comment_id: item.comment_id}) "
            "MATCH (m:Model "
            "{name: item.model}) "
            "MERGE (c)-[:MENTIONS]->(m)",
            batch,
        )
        print(
            f"      MENTIONS model (comment): "
            f"{len(batch)}"
        )

    def _link_post_occurred_in(self):
        batch = [
            {
                "post_id": p["post_id"],
                "quarter": p["quarter"],
            }
            for p in self._posts
            if p.get("post_id")
            and p.get("quarter")
        ]
        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (p:Post "
            "{post_id: item.post_id}) "
            "MATCH (tp:TimePeriod "
            "{period: item.quarter}) "
            "MERGE (p)-[:OCCURRED_IN]->(tp)",
            batch,
        )
        print(
            f"      OCCURRED_IN (post): {len(batch)}"
        )

    def _link_comment_occurred_in(self):
        batch = [
            {
                "comment_id": c["comment_id"],
                "quarter": c["quarter"],
            }
            for c in self._comments
            if c.get("comment_id")
            and c.get("quarter")
        ]
        self._run_batch(
            "UNWIND $batch AS item "
            "MATCH (c:Comment "
            "{comment_id: item.comment_id}) "
            "MATCH (tp:TimePeriod "
            "{period: item.quarter}) "
            "MERGE (c)-[:OCCURRED_IN]->(tp)",
            batch,
        )
        print(
            f"      OCCURRED_IN (comment): "
            f"{len(batch)}"
        )

    # -----------------------------------------
    # Helpers
    # -----------------------------------------

    def _run_batch(
        self,
        cypher: str,
        data: list[dict],
    ):
        if not data:
            return

        with self._driver.session() as session:
            for i in range(
                0, len(data), BATCH_SIZE
            ):
                batch = (
                    data[i: i + BATCH_SIZE]
                )
                session.run(cypher, batch=batch)


def _load_jsonl(filepath: str) -> list[dict]:
    items = []
    if not os.path.exists(filepath):
        return items
    with open(
        filepath, "r", encoding="utf-8"
    ) as f:
        for line in f:
            items.append(json.loads(line))
    return items
