import os
import re
from datetime import datetime

from dotenv import load_dotenv

from retrieval.base import (
    BaseRetriever,
    RetrievedDoc,
)
from preprocessing.extractor import (
    extract_topics,
    extract_companies,
    extract_models,
)

load_dotenv()

WINDOW_TO_QUARTERS = {
    "W1": ["Q1_2025", "Q2_2025"],
    "W2": ["Q3_2025", "Q4_2025"],
    "W3": ["Q1_2026", "Q2_2026"],
}

ALL_QUARTERS = [
    "Q1_2025", "Q2_2025", "Q3_2025",
    "Q4_2025", "Q1_2026", "Q2_2026",
]

QUARTER_PATTERN = re.compile(
    r"Q([1-4])\s*_?\s*(20\d{2})", re.IGNORECASE
)
YEAR_PATTERN = re.compile(r"\b(202[4-7])\b")


class GraphRetriever(BaseRetriever):

    def __init__(self):
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")

        if not uri or not user or not password:
            raise ValueError(
                "NEO4J_URI, NEO4J_USER, "
                "NEO4J_PASSWORD required in .env"
            )

        self._driver = GraphDatabase.driver(
            uri, auth=(user, password)
        )
        self._driver.verify_connectivity()
        self._analytics_cache = ""

    def close(self):
        self._driver.close()

    @property
    def analytics(self) -> str:
        return self._analytics_cache

    # -----------------------------------------
    # Main retrieve — graph-native
    # -----------------------------------------

    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> list[RetrievedDoc]:
        entities = extract_query_entities(query)
        periods = self._resolve_periods(where)
        subreddit = (
            where.get("subreddit")
            if where else None
        )

        docs = []

        topics = entities["topics"]
        companies = entities["companies"]
        models = entities["models"]

        if topics:
            docs.extend(self._by_topic(
                topics, periods, subreddit,
                n_results,
            ))
            docs.extend(self._thread_hubs(
                topics, periods, n_results // 2,
            ))
            docs.extend(
                self._influential_user_posts(
                    topics, periods, n_results // 2,
                )
            )

        if companies:
            docs.extend(self._by_entity(
                companies, "Company", periods,
                subreddit, n_results,
            ))

        if models:
            docs.extend(self._by_entity(
                models, "Model", periods,
                subreddit, n_results,
            ))

        if topics or companies or models:
            docs.extend(
                self._entity_network(
                    topics, companies, models,
                    periods, n_results // 2,
                )
            )

        if not docs:
            docs = self._general(
                query, periods, subreddit,
                n_results,
            )

        docs = _dedupe(docs, n_results)

        self._analytics_cache = (
            self._build_analytics(entities)
        )

        return docs

    # -----------------------------------------
    # Document queries
    # -----------------------------------------

    def _by_topic(
        self, topics, periods, subreddit, limit
    ):
        cypher = (
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic) "
            "WHERE t.name IN $topics "
            + _period_clause(periods)
            + _sub_clause(subreddit)
            + " OPTIONAL MATCH "
            "(p)<-[:AUTHORED]-(u:User) "
            "OPTIONAL MATCH "
            "(p)-[:POSTED_IN]->(s:Subreddit) "
            "OPTIONAL MATCH "
            "(p)-[:OCCURRED_IN]->(tp:TimePeriod)"
            " RETURN p.post_id AS id, "
            "p.title AS title, p.body AS text, "
            "p.created_at AS created_at, "
            "p.sentiment AS sentiment, "
            "p.sentiment_score AS ss, "
            "p.url AS url, "
            "u.username AS author, "
            "s.name AS subreddit, "
            "tp.period AS period "
            "LIMIT $limit"
        )
        records = self._run(
            cypher, topics=topics, limit=limit
        )
        return _records_to_docs(
            records, "post", "topic"
        )

    def _by_entity(
        self, names, label, periods,
        subreddit, limit
    ):
        cypher = (
            f"MATCH (p:Post)-[:MENTIONS]->"
            f"(e:{label}) "
            "WHERE e.name IN $names "
            + _period_clause(periods)
            + _sub_clause(subreddit)
            + " OPTIONAL MATCH "
            "(p)<-[:AUTHORED]-(u:User) "
            "OPTIONAL MATCH "
            "(p)-[:POSTED_IN]->(s:Subreddit) "
            "OPTIONAL MATCH "
            "(p)-[:OCCURRED_IN]->(tp:TimePeriod)"
            " RETURN p.post_id AS id, "
            "p.title AS title, p.body AS text, "
            "p.created_at AS created_at, "
            "p.sentiment AS sentiment, "
            "p.sentiment_score AS ss, "
            "p.url AS url, "
            "u.username AS author, "
            "s.name AS subreddit, "
            "tp.period AS period "
            "LIMIT $limit"
        )
        records = self._run(
            cypher, names=names, limit=limit
        )
        return _records_to_docs(
            records, "post", f"{label.lower()}"
        )

    def _thread_hubs(
        self, topics, periods, limit
    ):
        cypher = (
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic) "
            "WHERE t.name IN $topics "
            + _period_clause(periods)
            + " OPTIONAL MATCH "
            "(c:Comment)-[:COMMENTS_ON]->(p) "
            "WITH p, count(c) AS replies "
            "ORDER BY replies DESC "
            "LIMIT $limit "
            "OPTIONAL MATCH "
            "(p)<-[:AUTHORED]-(u:User) "
            "OPTIONAL MATCH "
            "(p)-[:POSTED_IN]->(s:Subreddit) "
            "OPTIONAL MATCH "
            "(p)-[:OCCURRED_IN]->(tp:TimePeriod)"
            " RETURN p.post_id AS id, "
            "p.title AS title, p.body AS text, "
            "p.created_at AS created_at, "
            "p.sentiment AS sentiment, "
            "p.sentiment_score AS ss, "
            "p.url AS url, "
            "u.username AS author, "
            "s.name AS subreddit, "
            "tp.period AS period, "
            "replies"
        )
        records = self._run(
            cypher, topics=topics, limit=limit
        )
        return _records_to_docs(
            records, "post", "thread_hub"
        )

    def _influential_user_posts(
        self, topics, periods, limit
    ):
        cypher = (
            "MATCH (u:User)-[:AUTHORED]->"
            "(p:Post)-[:DISCUSSES]->"
            "(t:Topic) "
            "WHERE t.name IN $topics "
            "WITH u, count(DISTINCT p) AS pc "
            "OPTIONAL MATCH "
            "(u)-[:AUTHORED]->(c:Comment) "
            "WITH u, pc, "
            "count(DISTINCT c) AS cc "
            "OPTIONAL MATCH "
            "(u)-[:AUTHORED]->"
            "(c2:Comment)<-[:REPLIES_TO]-"
            "(:Comment) "
            "WITH u, pc, cc, "
            "count(DISTINCT c2) AS rr "
            "WITH u, pc*3 + cc + rr*2 "
            "AS influence "
            "ORDER BY influence DESC "
            "LIMIT 3 "
            "MATCH (u)-[:AUTHORED]->(p:Post) "
            "OPTIONAL MATCH "
            "(p)-[:POSTED_IN]->(s:Subreddit) "
            "OPTIONAL MATCH "
            "(p)-[:OCCURRED_IN]->(tp:TimePeriod)"
            " RETURN p.post_id AS id, "
            "p.title AS title, p.body AS text, "
            "p.created_at AS created_at, "
            "p.sentiment AS sentiment, "
            "p.sentiment_score AS ss, "
            "p.url AS url, "
            "u.username AS author, "
            "s.name AS subreddit, "
            "tp.period AS period, "
            "influence "
            "LIMIT $limit"
        )
        records = self._run(
            cypher, topics=topics, limit=limit
        )
        return _records_to_docs(
            records, "post", "influential_author"
        )

    def _entity_network(
        self, topics, companies, models,
        periods, limit,
    ):
        all_entities = companies + models
        if not topics or not all_entities:
            return []

        cypher = (
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic), "
            "(p)-[:MENTIONS]->(e) "
            "WHERE t.name IN $topics "
            "AND e.name IN $entities "
            + _period_clause(periods)
            + " WITH p, "
            "count(DISTINCT e) AS entity_hits, "
            "collect(DISTINCT e.name) AS matched "
            "ORDER BY entity_hits DESC "
            "LIMIT $limit "
            "OPTIONAL MATCH "
            "(p)<-[:AUTHORED]-(u:User) "
            "OPTIONAL MATCH "
            "(p)-[:POSTED_IN]->(s:Subreddit) "
            "OPTIONAL MATCH "
            "(p)-[:OCCURRED_IN]->(tp:TimePeriod)"
            " RETURN p.post_id AS id, "
            "p.title AS title, p.body AS text, "
            "p.created_at AS created_at, "
            "p.sentiment AS sentiment, "
            "p.sentiment_score AS ss, "
            "p.url AS url, "
            "u.username AS author, "
            "s.name AS subreddit, "
            "tp.period AS period, "
            "entity_hits, matched"
        )
        records = self._run(
            cypher, topics=topics,
            entities=all_entities, limit=limit,
        )
        return _records_to_docs(
            records, "post", "entity_network"
        )

    def _general(
        self, query, periods, subreddit, limit
    ):
        cypher = (
            "MATCH (p:Post) "
            "OPTIONAL MATCH "
            "(c:Comment)-[:COMMENTS_ON]->(p) "
            "WITH p, count(c) AS replies "
            + _period_clause(periods)
            + _sub_clause(subreddit)
            + " ORDER BY replies DESC "
            "LIMIT $limit "
            "OPTIONAL MATCH "
            "(p)<-[:AUTHORED]-(u:User) "
            "OPTIONAL MATCH "
            "(p)-[:POSTED_IN]->(s:Subreddit) "
            "OPTIONAL MATCH "
            "(p)-[:OCCURRED_IN]->(tp:TimePeriod)"
            " RETURN p.post_id AS id, "
            "p.title AS title, p.body AS text, "
            "p.created_at AS created_at, "
            "p.sentiment AS sentiment, "
            "p.sentiment_score AS ss, "
            "p.url AS url, "
            "u.username AS author, "
            "s.name AS subreddit, "
            "tp.period AS period, "
            "replies"
        )
        records = self._run(
            cypher, limit=limit
        )
        return _records_to_docs(
            records, "post", "most_discussed"
        )

    # -----------------------------------------
    # Analytics — graph traversal insights
    # -----------------------------------------

    def _build_analytics(self, entities):
        sections = []
        topics = entities["topics"]

        if topics:
            for fn in [
                self._sentiment_snapshots,
                self._trend_snapshots,
                self._influential_users,
                self._related_entities,
                self._topic_connections,
                self._subreddit_breakdown,
            ]:
                s = fn(topics)
                if s:
                    sections.append(s)

        return "\n\n".join(sections)

    def _sentiment_snapshots(self, topics):
        records = self._run(
            "MATCH (ss:SentimentSnapshot) "
            "WHERE ss.topic IN $topics "
            "RETURN ss.topic AS topic, "
            "ss.period AS period, "
            "ss.post_count AS posts, "
            "ss.avg_sentiment AS avg_sent, "
            "ss.positive_count AS pos, "
            "ss.negative_count AS neg, "
            "ss.neutral_count AS neu "
            "ORDER BY ss.period, ss.topic",
            topics=topics,
        )
        if not records:
            return self._sentiment_over_time(
                topics
            )
        lines = [
            "### Graph: Sentiment Snapshots"
        ]
        for r in records:
            lines.append(
                f"  {r['period']} | "
                f"{r['topic']}: "
                f"{r['posts']} posts, "
                f"avg={r['avg_sent']}, "
                f"+{r['pos']}/-{r['neg']}"
                f"/{r['neu']}neu"
            )
        return "\n".join(lines)

    def _trend_snapshots(self, topics):
        records = self._run(
            "MATCH (ts:TrendSnapshot) "
            "WHERE ts.topic IN $topics "
            "RETURN ts.topic AS topic, "
            "ts.from_period AS from_p, "
            "ts.to_period AS to_p, "
            "ts.from_count AS from_c, "
            "ts.to_count AS to_c, "
            "ts.growth_pct AS growth, "
            "ts.status AS status "
            "ORDER BY ts.from_period",
            topics=topics,
        )
        if not records:
            return ""
        lines = [
            "### Graph: Topic Trends"
        ]
        for r in records:
            lines.append(
                f"  {r['topic']}: "
                f"{r['from_p']}->{r['to_p']} "
                f"({r['from_c']}->{r['to_c']}, "
                f"{r['growth']:+.1f}%, "
                f"{r['status']})"
            )
        return "\n".join(lines)

    def _sentiment_over_time(self, topics):
        records = self._run(
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic), "
            "(p)-[:OCCURRED_IN]->"
            "(tp:TimePeriod) "
            "WHERE t.name IN $topics "
            "RETURN tp.period AS period, "
            "t.name AS topic, "
            "count(p) AS posts, "
            "round(avg(p.sentiment_score)*100)"
            "/100.0 AS avg_sentiment "
            "ORDER BY tp.period, t.name",
            topics=topics,
        )
        if not records:
            return ""
        lines = ["### Graph: Sentiment Over Time"]
        for r in records:
            lines.append(
                f"  {r['period']} | "
                f"{r['topic']}: "
                f"{r['posts']} posts, "
                f"avg_sentiment="
                f"{r['avg_sentiment']}"
            )
        return "\n".join(lines)

    def _influential_users(self, topics):
        records = self._run(
            "MATCH (u:User)-[:AUTHORED]->"
            "(p:Post)-[:DISCUSSES]->"
            "(t:Topic) "
            "WHERE t.name IN $topics "
            "WITH u, count(DISTINCT p) AS pc, "
            "collect(DISTINCT t.name) AS topics "
            "OPTIONAL MATCH "
            "(u)-[:AUTHORED]->(c:Comment) "
            "WITH u, pc, topics, "
            "count(DISTINCT c) AS cc "
            "OPTIONAL MATCH "
            "(u)-[:AUTHORED]->"
            "(c2:Comment)<-[:REPLIES_TO]-"
            "(:Comment) "
            "WITH u, pc, cc, topics, "
            "count(DISTINCT c2) AS rr "
            "WITH u.username AS author, "
            "pc, cc, rr, topics, "
            "pc*3 + cc + rr*2 AS influence "
            "ORDER BY influence DESC "
            "LIMIT 5 "
            "RETURN author, pc AS posts, "
            "cc AS comments, "
            "rr AS replies_received, "
            "influence, topics",
            topics=topics,
        )
        if not records:
            return ""
        lines = [
            "### Graph: Influential Users "
            "(score = posts*3 + comments "
            "+ replies_received*2)"
        ]
        for r in records:
            lines.append(
                f"  u/{r['author']}: "
                f"influence={r['influence']} "
                f"(posts={r['posts']}, "
                f"comments={r['comments']}, "
                f"replies={r['replies_received']}) "
                f"topics: "
                f"{', '.join(r['topics'])}"
            )
        return "\n".join(lines)

    def _related_entities(self, topics):
        records = self._run(
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic), "
            "(p)-[:MENTIONS]->(e) "
            "WHERE t.name IN $topics "
            "AND (e:Company OR e:Model) "
            "RETURN labels(e)[0] AS type, "
            "e.name AS name, "
            "count(p) AS mentions "
            "ORDER BY mentions DESC "
            "LIMIT 10",
            topics=topics,
        )
        if not records:
            return ""
        lines = [
            "### Graph: Related Entities"
        ]
        for r in records:
            lines.append(
                f"  {r['type']}: {r['name']} "
                f"({r['mentions']} co-mentions)"
            )
        return "\n".join(lines)

    def _topic_connections(self, topics):
        records = self._run(
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t1:Topic), "
            "(p)-[:DISCUSSES]->(t2:Topic) "
            "WHERE t1.name IN $topics "
            "AND t1 <> t2 "
            "RETURN t1.name AS source, "
            "t2.name AS related, "
            "count(p) AS co_occurrence "
            "ORDER BY co_occurrence DESC "
            "LIMIT 8",
            topics=topics,
        )
        if not records:
            return ""
        lines = [
            "### Graph: Topic Connections"
        ]
        for r in records:
            lines.append(
                f"  {r['source']} <-> "
                f"{r['related']} "
                f"({r['co_occurrence']} "
                f"shared posts)"
            )
        return "\n".join(lines)

    def _subreddit_breakdown(self, topics):
        records = self._run(
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic), "
            "(p)-[:POSTED_IN]->(s:Subreddit) "
            "WHERE t.name IN $topics "
            "RETURN s.name AS subreddit, "
            "count(p) AS posts, "
            "round(avg(p.sentiment_score)*100)"
            "/100.0 AS avg_sentiment "
            "ORDER BY posts DESC",
            topics=topics,
        )
        if not records:
            return ""
        lines = [
            "### Graph: Community Breakdown"
        ]
        for r in records:
            lines.append(
                f"  r/{r['subreddit']}: "
                f"{r['posts']} posts, "
                f"avg_sentiment="
                f"{r['avg_sentiment']}"
            )
        return "\n".join(lines)

    # -----------------------------------------
    # Temporal graph analytics (Issue 4)
    # -----------------------------------------

    def get_temporal_analytics(
        self, query, windows=None
    ):
        if not windows:
            windows = ["W1", "W2", "W3"]

        entities = extract_query_entities(query)
        periods = []
        for w in windows:
            periods.extend(
                WINDOW_TO_QUARTERS.get(w, [])
            )
        if not periods:
            periods = ALL_QUARTERS

        sections = []
        topics = entities["topics"]

        if topics:
            for fn in [
                self._temporal_sentiment,
                self._topic_evolution,
                self._entity_evolution,
                self._emerging_topics,
            ]:
                s = fn(topics, periods)
                if s:
                    sections.append(s)

        s = self._community_evolution(periods)
        if s:
            sections.append(s)

        s = self._user_evolution(periods)
        if s:
            sections.append(s)

        return "\n\n".join(sections)

    def _temporal_sentiment(self, topics, periods):
        records = self._run(
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic), "
            "(p)-[:OCCURRED_IN]->"
            "(tp:TimePeriod) "
            "WHERE t.name IN $topics "
            "AND tp.period IN $periods "
            "RETURN tp.period AS period, "
            "t.name AS topic, "
            "count(p) AS posts, "
            "round(avg(p.sentiment_score)*100)"
            "/100.0 AS avg_sentiment "
            "ORDER BY tp.period, t.name",
            topics=topics, periods=periods,
        )
        if not records:
            return ""
        lines = [
            "### Temporal: Sentiment Trends"
        ]
        for r in records:
            lines.append(
                f"  {r['period']} | "
                f"{r['topic']}: "
                f"{r['posts']} posts, "
                f"sentiment={r['avg_sentiment']}"
            )
        return "\n".join(lines)

    def _topic_evolution(self, topics, periods):
        records = self._run(
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic), "
            "(p)-[:OCCURRED_IN]->"
            "(tp:TimePeriod) "
            "WHERE tp.period IN $periods "
            "RETURN tp.period AS period, "
            "t.name AS topic, "
            "count(p) AS posts "
            "ORDER BY tp.period, posts DESC",
            periods=periods,
        )
        if not records:
            return ""

        by_period = {}
        for r in records:
            p = r["period"]
            if p not in by_period:
                by_period[p] = []
            by_period[p].append(
                (r["topic"], r["posts"])
            )

        lines = [
            "### Temporal: Topic Evolution"
        ]
        for period in sorted(by_period):
            top = by_period[period][:5]
            ranked = ", ".join(
                f"{t}({c})" for t, c in top
            )
            lines.append(
                f"  {period}: {ranked}"
            )

        first_p = sorted(by_period)[0]
        last_p = sorted(by_period)[-1]
        first_topics = {
            t: c for t, c
            in by_period.get(first_p, [])
        }
        last_topics = {
            t: c for t, c
            in by_period.get(last_p, [])
        }

        growing = []
        declining = []
        for t in set(first_topics) | set(last_topics):
            f = first_topics.get(t, 0)
            l = last_topics.get(t, 0)
            if l > f:
                growing.append(
                    f"{t} ({f}->{l})"
                )
            elif f > l:
                declining.append(
                    f"{t} ({f}->{l})"
                )

        if growing:
            lines.append(
                f"  Growing: "
                f"{', '.join(growing[:5])}"
            )
        if declining:
            lines.append(
                f"  Declining: "
                f"{', '.join(declining[:5])}"
            )

        return "\n".join(lines)

    def _entity_evolution(self, topics, periods):
        records = self._run(
            "MATCH (p:Post)-[:MENTIONS]->(e), "
            "(p)-[:OCCURRED_IN]->"
            "(tp:TimePeriod) "
            "WHERE tp.period IN $periods "
            "AND (e:Company OR e:Model) "
            "RETURN tp.period AS period, "
            "labels(e)[0] AS type, "
            "e.name AS name, "
            "count(p) AS mentions "
            "ORDER BY tp.period, "
            "mentions DESC",
            periods=periods,
        )
        if not records:
            return ""

        by_period = {}
        for r in records:
            p = r["period"]
            if p not in by_period:
                by_period[p] = []
            by_period[p].append(
                f"{r['name']}({r['mentions']})"
            )

        lines = [
            "### Temporal: Entity Evolution"
        ]
        for period in sorted(by_period):
            top = by_period[period][:5]
            lines.append(
                f"  {period}: {', '.join(top)}"
            )
        return "\n".join(lines)

    def _emerging_topics(self, topics, periods):
        if len(periods) < 2:
            return ""

        records = self._run(
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic), "
            "(p)-[:OCCURRED_IN]->"
            "(tp:TimePeriod) "
            "WHERE tp.period IN $periods "
            "WITH t.name AS topic, "
            "collect(DISTINCT tp.period) "
            "AS seen_in, "
            "count(p) AS total "
            "RETURN topic, seen_in, total "
            "ORDER BY total DESC",
            periods=periods,
        )
        if not records:
            return ""

        early = set(periods[:len(periods) // 2])
        late = set(periods[len(periods) // 2:])

        emerged = []
        for r in records:
            seen = set(r["seen_in"])
            if seen & late and not (seen & early):
                emerged.append(
                    f"{r['topic']}({r['total']})"
                )

        if not emerged:
            return ""

        return (
            "### Temporal: Emerging Topics "
            "(new in later periods)\n"
            f"  {', '.join(emerged[:10])}"
        )

    def _community_evolution(self, periods):
        records = self._run(
            "MATCH (p:Post)-[:POSTED_IN]->"
            "(s:Subreddit), "
            "(p)-[:OCCURRED_IN]->"
            "(tp:TimePeriod) "
            "WHERE tp.period IN $periods "
            "RETURN tp.period AS period, "
            "s.name AS subreddit, "
            "count(p) AS posts, "
            "round(avg(p.sentiment_score)*100)"
            "/100.0 AS avg_sentiment "
            "ORDER BY tp.period, posts DESC",
            periods=periods,
        )
        if not records:
            return ""

        by_period = {}
        for r in records:
            p = r["period"]
            if p not in by_period:
                by_period[p] = []
            by_period[p].append(
                f"r/{r['subreddit']}"
                f"({r['posts']}, "
                f"sent={r['avg_sentiment']})"
            )

        lines = [
            "### Temporal: Community Evolution"
        ]
        for period in sorted(by_period):
            lines.append(
                f"  {period}: "
                f"{', '.join(by_period[period])}"
            )
        return "\n".join(lines)

    def _user_evolution(self, periods):
        records = self._run(
            "MATCH (u:User)-[:AUTHORED]->"
            "(p:Post)-[:OCCURRED_IN]->"
            "(tp:TimePeriod) "
            "WHERE tp.period IN $periods "
            "WITH tp.period AS period, "
            "u.username AS author, "
            "count(p) AS posts "
            "ORDER BY period, posts DESC "
            "WITH period, "
            "collect({author: author, "
            "posts: posts})[..3] AS top "
            "RETURN period, top",
            periods=periods,
        )
        if not records:
            return ""
        lines = [
            "### Temporal: Top Authors by Period"
        ]
        for r in records:
            users = ", ".join(
                f"u/{u['author']}({u['posts']})"
                for u in r["top"]
            )
            lines.append(
                f"  {r['period']}: {users}"
            )
        return "\n".join(lines)

    # -----------------------------------------
    # Helpers
    # -----------------------------------------

    def _resolve_periods(self, where):
        if not where:
            return None
        tw = where.get("time_window")
        if tw and tw in WINDOW_TO_QUARTERS:
            return WINDOW_TO_QUARTERS[tw]
        return None

    def _run(self, cypher, **params):
        try:
            with self._driver.session() as s:
                result = s.run(cypher, **params)
                return [
                    dict(r) for r in result
                ]
        except Exception as e:
            print(f"[Graph] Query error: {e}")
            return []


# -----------------------------------------
# Entity extraction
# -----------------------------------------

def extract_query_entities(query: str) -> dict:
    topics = extract_topics(query)
    companies = extract_companies(query)
    models = extract_models(query)

    periods = []
    for m in QUARTER_PATTERN.finditer(query):
        periods.append(
            f"Q{m.group(1)}_{m.group(2)}"
        )
    if not periods:
        for m in YEAR_PATTERN.finditer(query):
            year = m.group(1)
            for q in range(1, 5):
                periods.append(f"Q{q}_{year}")

    return {
        "topics": topics,
        "companies": companies,
        "models": models,
        "periods": periods,
    }


# -----------------------------------------
# Cypher clause builders
# -----------------------------------------

def _period_clause(periods):
    if not periods:
        return ""
    escaped = ", ".join(
        f"'{p}'" for p in periods
    )
    return (
        "AND EXISTS { "
        "MATCH (p)-[:OCCURRED_IN]->"
        f"(tp2:TimePeriod) "
        f"WHERE tp2.period IN [{escaped}]"
        "} "
    )


def _sub_clause(subreddit):
    if not subreddit:
        return ""
    return (
        "AND EXISTS { "
        "MATCH (p)-[:POSTED_IN]->"
        f"(s2:Subreddit {{name: '{subreddit}'}})"
        "} "
    )


# -----------------------------------------
# Record -> RetrievedDoc
# -----------------------------------------

def _records_to_docs(records, source, match_type):
    docs = []
    for rank, r in enumerate(records):
        doc_id = r.get("id") or ""
        if not doc_id:
            continue

        title = r.get("title") or ""
        text = r.get("text") or ""
        full_text = (
            f"{title}\n\n{text}".strip()
            if title else text
        )
        if not full_text:
            continue

        score = round(
            1.0 - (rank * 0.05), 4
        )
        score = max(score, 0.1)

        docs.append(RetrievedDoc(
            id=doc_id,
            text=full_text[:1500],
            score=score,
            metadata={
                "title": title,
                "author": r.get("author") or "",
                "subreddit": (
                    r.get("subreddit") or ""
                ),
                "url": r.get("url") or "",
                "period": r.get("period") or "",
                "created_at": (
                    r.get("created_at") or ""
                ),
                "sentiment": (
                    r.get("sentiment") or ""
                ),
                "sentiment_score": (
                    r.get("ss") or 0
                ),
                "match_type": match_type,
                "retriever": "graph",
            },
            source=source,
        ))
    return docs


def _dedupe(docs, limit):
    seen = {}
    for doc in docs:
        if (
            doc.id not in seen
            or doc.score > seen[doc.id].score
        ):
            seen[doc.id] = doc
    result = sorted(
        seen.values(),
        key=lambda d: d.score,
        reverse=True,
    )
    return result[:limit]
