"""
Post-build graph algorithms:
  - P5: Influence scores on User nodes
  - P4: SentimentSnapshot nodes per topic+period
  - P3: TrendSnapshot nodes with growth metrics
"""


def run_all(driver):
    print("[Algorithms] Computing influence "
          "scores...")
    compute_influence(driver)

    print("[Algorithms] Creating sentiment "
          "snapshots...")
    create_sentiment_snapshots(driver)

    print("[Algorithms] Creating trend "
          "snapshots...")
    create_trend_snapshots(driver)


# -----------------------------------------
# P5: User influence scoring
# -----------------------------------------

def compute_influence(driver):
    _try_pagerank(driver)
    _degree_influence(driver)


def _try_pagerank(driver):
    try:
        with driver.session() as s:
            s.run(
                "CALL gds.graph.project("
                "'user_graph', "
                "['User','Post','Comment'], "
                "['AUTHORED','COMMENTS_ON',"
                "'REPLIES_TO'])"
            )
            s.run(
                "CALL gds.pageRank.write("
                "'user_graph', {"
                "writeProperty: 'pagerank'"
                "})"
            )
            s.run(
                "CALL gds.graph.drop("
                "'user_graph')"
            )
            print("  PageRank computed via GDS")
            return
    except Exception:
        pass


def _degree_influence(driver):
    with driver.session() as s:
        result = s.run(
            "MATCH (u:User) "
            "OPTIONAL MATCH "
            "(u)-[:AUTHORED]->(p:Post) "
            "WITH u, "
            "count(DISTINCT p) AS posts "
            "OPTIONAL MATCH "
            "(u)-[:AUTHORED]->(c:Comment) "
            "WITH u, posts, "
            "count(DISTINCT c) AS comments "
            "OPTIONAL MATCH "
            "(u)-[:AUTHORED]->(p2:Post)"
            "<-[:COMMENTS_ON]-"
            "(eng:Comment) "
            "WITH u, posts, comments, "
            "count(DISTINCT eng) "
            "AS received_comments "
            "OPTIONAL MATCH "
            "(u)-[:AUTHORED]->"
            "(c2:Comment)"
            "<-[:REPLIES_TO]-"
            "(rep:Comment) "
            "WITH u, posts, comments, "
            "received_comments, "
            "count(DISTINCT rep) "
            "AS received_replies "
            "SET u.influence_score = "
            "toFloat("
            "posts * 5 + comments * 1 + "
            "received_comments * 2 + "
            "received_replies * 3"
            ") "
            "RETURN count(u) AS updated"
        )
        count = result.single()["updated"]
        print(
            f"  Influence scores set on "
            f"{count} users"
        )


# -----------------------------------------
# P4: SentimentSnapshot nodes
# -----------------------------------------

def create_sentiment_snapshots(driver):
    with driver.session() as s:
        s.run(
            "MATCH (ss:SentimentSnapshot) "
            "DETACH DELETE ss"
        )

        result = s.run(
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic), "
            "(p)-[:OCCURRED_IN]->"
            "(tp:TimePeriod) "
            "WITH t, tp, "
            "count(p) AS post_count, "
            "round(avg(p.sentiment_score)"
            "*1000)/1000.0 AS avg_sent, "
            "round(stDev(p.sentiment_score)"
            "*1000)/1000.0 AS std_sent, "
            "sum(CASE WHEN "
            "p.sentiment = 'positive' "
            "THEN 1 ELSE 0 END) AS pos, "
            "sum(CASE WHEN "
            "p.sentiment = 'negative' "
            "THEN 1 ELSE 0 END) AS neg, "
            "sum(CASE WHEN "
            "p.sentiment = 'neutral' "
            "THEN 1 ELSE 0 END) AS neu "
            "CREATE (ss:SentimentSnapshot {"
            "topic: t.name, "
            "period: tp.period, "
            "post_count: post_count, "
            "avg_sentiment: avg_sent, "
            "std_sentiment: std_sent, "
            "positive_count: pos, "
            "negative_count: neg, "
            "neutral_count: neu"
            "}) "
            "MERGE (t)-[:HAS_SENTIMENT]->(ss) "
            "MERGE (ss)-[:IN_PERIOD]->(tp) "
            "RETURN count(ss) AS created"
        )
        count = result.single()["created"]
        print(
            f"  Created {count} "
            f"SentimentSnapshot nodes"
        )


# -----------------------------------------
# P3: TrendSnapshot nodes
# -----------------------------------------

def create_trend_snapshots(driver):
    with driver.session() as s:
        s.run(
            "MATCH (ts:TrendSnapshot) "
            "DETACH DELETE ts"
        )

        periods = s.run(
            "MATCH (tp:TimePeriod) "
            "RETURN tp.period AS period "
            "ORDER BY tp.period"
        )
        period_list = [
            r["period"] for r in periods
        ]

        if len(period_list) < 2:
            print("  Not enough periods "
                  "for trends")
            return

        topics = s.run(
            "MATCH (t:Topic) "
            "RETURN t.name AS name"
        )
        topic_list = [
            r["name"] for r in topics
        ]

        counts = {}
        rows = s.run(
            "MATCH (p:Post)-[:DISCUSSES]->"
            "(t:Topic), "
            "(p)-[:OCCURRED_IN]->"
            "(tp:TimePeriod) "
            "RETURN t.name AS topic, "
            "tp.period AS period, "
            "count(p) AS posts"
        )
        for r in rows:
            key = (r["topic"], r["period"])
            counts[key] = r["posts"]

        created = 0
        for topic in topic_list:
            for i in range(
                1, len(period_list)
            ):
                prev = period_list[i - 1]
                curr = period_list[i]
                prev_count = counts.get(
                    (topic, prev), 0
                )
                curr_count = counts.get(
                    (topic, curr), 0
                )

                if (
                    prev_count == 0
                    and curr_count == 0
                ):
                    continue

                if prev_count > 0:
                    growth = round(
                        (curr_count - prev_count)
                        / prev_count * 100, 1
                    )
                elif curr_count > 0:
                    growth = 100.0
                else:
                    growth = 0.0

                if curr_count > 0 and prev_count == 0:
                    status = "emerging"
                elif growth > 20:
                    status = "growing"
                elif growth < -20:
                    status = "declining"
                else:
                    status = "stable"

                s.run(
                    "MATCH (t:Topic "
                    "{name: $topic}) "
                    "CREATE (ts:TrendSnapshot {"
                    "topic: $topic, "
                    "from_period: $prev, "
                    "to_period: $curr, "
                    "from_count: $prev_count, "
                    "to_count: $curr_count, "
                    "growth_pct: $growth, "
                    "status: $status"
                    "}) "
                    "MERGE (t)-[:HAS_TREND]"
                    "->(ts)",
                    topic=topic,
                    prev=prev,
                    curr=curr,
                    prev_count=prev_count,
                    curr_count=curr_count,
                    growth=growth,
                    status=status,
                )
                created += 1

        print(
            f"  Created {created} "
            f"TrendSnapshot nodes"
        )
