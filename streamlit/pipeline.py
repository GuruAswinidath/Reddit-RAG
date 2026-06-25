import sys
import os
import asyncio
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from ingestion.search import search_reddit_urls, search_all_topics, TARGET_SUBREDDITS
from ingestion.scraper import RedditScraper
from ingestion.parser import RedditParser
from preprocessing.cleaner import clean_post
from preprocessing.normalizer import Normalizer
from vector_store.embeddings import get_embedding_model
from vector_store.store import VectorStore


EMBED_MODEL = "sentence-transformer"


# ── Page config ──────────────────────────────

st.set_page_config(
    page_title="Reddit RAG — Data Pipeline",
    page_icon="⚙️",
    layout="wide",
)

# ── Session state ────────────────────────────

if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = None
if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = None
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None
if "normalized_posts" not in st.session_state:
    st.session_state.normalized_posts = None
if "normalized_comments" not in st.session_state:
    st.session_state.normalized_comments = None
if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []


def log(msg: str):
    st.session_state.pipeline_log.append(msg)


# ── Title ────────────────────────────────────

st.title("Reddit RAG — Data Pipeline")
st.markdown("Pull data from Reddit, clean it, and push to vector DB.")

# ── Sidebar ──────────────────────────────────

with st.sidebar:
    st.header("Pipeline Status")

    scraped = st.session_state.scraped_data is not None
    parsed = st.session_state.parsed_data is not None
    cleaned = st.session_state.cleaned_data is not None
    normalized = st.session_state.normalized_posts is not None

    st.markdown(f"{'✅' if scraped else '⬜'} **1. Pull Data**")
    st.markdown(f"{'✅' if parsed else '⬜'} **2. Parse**")
    st.markdown(f"{'✅' if cleaned else '⬜'} **3. Clean**")
    st.markdown(f"{'✅' if normalized else '⬜'} **4. Normalize**")
    st.markdown(f"⬜ **5. Push to Vector DB**")
    st.markdown(f"⬜ **6. Push to Neo4j**")

    st.divider()

    if st.button("Reset Pipeline", use_container_width=True):
        for key in [
            "scraped_data", "parsed_data", "cleaned_data",
            "normalized_posts", "normalized_comments", "pipeline_log",
        ]:
            st.session_state[key] = None if key != "pipeline_log" else []
        st.rerun()

    st.divider()

    if st.session_state.pipeline_log:
        st.subheader("Log")
        for entry in st.session_state.pipeline_log[-20:]:
            st.text(entry)


# ── Step 1: Pull Data ───────────────────────

st.header("1. Pull Data from Reddit")

col1, col2 = st.columns(2)

with col1:
    topic = st.text_input(
        "Search Topics (comma-separated)",
        placeholder="e.g. RAG, Agentic AI, Claude",
    )

with col2:
    year_filter = st.selectbox(
        "Year filter",
        options=[
            None, 2026, 2025, 2024, 2023,
        ],
        index=0,
        format_func=lambda y: (
            "All years" if y is None
            else str(y)
        ),
        help="Filter Reddit API results to "
             "posts from this year only. "
             "'All years' fetches everything.",
    )

restrict_subs = st.checkbox(
    f"Restrict to target subreddits ({', '.join(TARGET_SUBREDDITS)})",
    value=False,
)

if st.button("🔍 Pull Data", type="primary", disabled=not topic):
    st.session_state.scraped_data = None
    st.session_state.parsed_data = None
    st.session_state.cleaned_data = None
    st.session_state.normalized_posts = None
    st.session_state.normalized_comments = None
    st.session_state.pipeline_log = []

    topics = [
        t.strip() for t in topic.split(",")
        if t.strip()
    ]

    with st.status("Pulling data from Reddit...", expanded=True) as status:

        # Search
        st.write(
            f"Searching {len(topics)} topics "
            f"(with time variants)..."
        )
        for t in topics:
            st.write(f"  - {t}")
        log(
            f"Searching: {topics} "
            f"(limit={num_posts}/topic)"
        )

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(
            search_all_topics(
                topics=topics,
                restrict_subreddits=restrict_subs,
                use_time_variants=True,
                year_filter=year_filter,
            )
        )
        log(f"Found {len(results)} URLs")
        st.write(f"Found **{len(results)}** unique Reddit URLs")

        if not results:
            status.update(label="No URLs found. Try a different topic.", state="error")
        else:
            # Scrape
            st.write(f"Scraping {len(results)} posts...")
            log(f"Scraping {len(results)} posts...")

            urls = [r["url"] for r in results]
            scraper = RedditScraper()
            scraped = loop.run_until_complete(
                scraper.scrape_urls(urls, search_metadata=results)
            )

            slug = "_".join(topics[:3])[:40]
            raw_file = scraper.save_results(slug, scraped)
            log(f"Saved raw data to {raw_file}")

            # Parse
            st.write("Parsing scraped data...")
            log("Parsing...")

            parser = RedditParser()
            parsed = parser.parse_scraped_file(raw_file)
            parser.save_parsed(parsed, raw_file)

            st.session_state.scraped_data = scraped
            st.session_state.parsed_data = parsed

            success_count = sum(1 for s in scraped if s.get("success"))
            log(f"Scraped: {success_count}/{len(scraped)} successful")
            log(f"Parsed: {len(parsed)} posts")

            status.update(
                label=f"Pulled {len(parsed)} posts from {success_count} pages",
                state="complete",
            )

        loop.close()

if st.session_state.parsed_data:
    with st.expander(f"📄 Parsed Posts ({len(st.session_state.parsed_data)})", expanded=False):
        for i, post in enumerate(st.session_state.parsed_data[:20]):
            st.markdown(
                f"**{i+1}. [{post.get('title', 'No title')}]({post.get('url', '')})**  \n"
                f"r/{post.get('subreddit', '?')} — "
                f"{post.get('comment_count', 0)} comments"
            )

st.divider()

# ── Step 2: Clean & Normalize ────────────────

st.header("2. Clean & Normalize Data")

can_clean = st.session_state.parsed_data is not None

if st.button(
    "🧹 Clean Data",
    type="primary",
    disabled=not can_clean,
):
    with st.status("Cleaning and normalizing...", expanded=True) as status:
        parsed = st.session_state.parsed_data

        # Clean
        st.write(f"Cleaning {len(parsed)} posts...")
        log(f"Cleaning {len(parsed)} posts...")

        cleaned = []
        skipped = 0
        for post in parsed:
            result = clean_post(post)
            if not result["title"] and not result["body"]:
                skipped += 1
                continue
            cleaned.append(result)

        log(f"Cleaned: {len(cleaned)} posts, skipped {skipped}")
        st.write(f"Cleaned **{len(cleaned)}** posts (skipped {skipped})")

        # Save cleaned
        os.makedirs("data/cleaned", exist_ok=True)
        cleaned_file = "data/cleaned/streamlit_pipeline.json"
        with open(cleaned_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        log(f"Saved cleaned data to {cleaned_file}")

        # Normalize
        st.write("Normalizing into posts and comments...")
        log("Normalizing...")

        normalizer = Normalizer()
        posts, comments = normalizer.normalize(cleaned)
        normalizer.save(posts, comments)
        report = normalizer.quality_report(posts, comments)

        st.session_state.cleaned_data = cleaned
        st.session_state.normalized_posts = posts
        st.session_state.normalized_comments = comments

        log(f"Normalized: {len(posts)} posts, {len(comments)} comments")
        status.update(
            label=f"Cleaned: {len(posts)} posts, {len(comments)} comments",
            state="complete",
        )

if st.session_state.normalized_posts:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Posts", len(st.session_state.normalized_posts))
    with col2:
        st.metric("Comments", len(st.session_state.normalized_comments))
    with col3:
        subreddits = set(
            p.get("subreddit", "") for p in st.session_state.normalized_posts
        )
        st.metric("Subreddits", len(subreddits))

    with st.expander("📊 Data Preview", expanded=False):
        for i, post in enumerate(st.session_state.normalized_posts[:10]):
            topics = ", ".join(post.get("topics", [])) or "none"
            st.markdown(
                f"**{i+1}. {post.get('title', 'No title')}**  \n"
                f"r/{post.get('subreddit', '?')} | "
                f"{post.get('time_window', '?')} | "
                f"Topics: {topics} | "
                f"{post.get('comment_count', 0)} comments"
            )

st.divider()

# ── Step 3: Push to Vector DB ────────────────

st.header("3. Push to Vector DB")

can_push = st.session_state.normalized_posts is not None

if st.button(
    "🚀 Push to ChromaDB",
    type="primary",
    disabled=not can_push,
):
    with st.status("Embedding and storing in ChromaDB...", expanded=True) as status:
        posts = st.session_state.normalized_posts
        comments = st.session_state.normalized_comments

        st.write(f"Loading embedding model ({EMBED_MODEL})...")
        log(f"Loading embedding model: {EMBED_MODEL}")

        model = get_embedding_model(EMBED_MODEL)
        store = VectorStore(model)

        st.write(f"Embedding and storing {len(posts)} posts...")
        log(f"Embedding {len(posts)} posts...")
        posts_added = store.add_posts(posts)

        st.write(f"Embedding and storing {len(comments)} comments...")
        log(f"Embedding {len(comments)} comments...")
        comments_added = store.add_comments(comments)

        stats = store.stats()
        log(f"Vector DB: {stats['posts']} posts, {stats['comments']} comments")

        status.update(
            label=f"Stored {posts_added} post chunks + {comments_added} comment chunks",
            state="complete",
        )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Posts in DB", stats["posts"])
    with col2:
        st.metric("Total Comments in DB", stats["comments"])

st.divider()

# ── Step 4: Push to Neo4j ───────────────────

st.header("4. Push to Neo4j Knowledge Graph")

can_push_neo4j = st.session_state.normalized_posts is not None

col_neo1, col_neo2 = st.columns(2)

with col_neo1:
    clear_graph = st.checkbox(
        "Clear existing graph first",
        value=False,
    )

if st.button(
    "🕸️ Push to Neo4j",
    type="primary",
    disabled=not can_push_neo4j,
):
    import os as _os
    from dotenv import load_dotenv as _ld
    _ld()

    neo_uri = _os.getenv("NEO4J_URI")
    neo_user = _os.getenv("NEO4J_USER")
    neo_pw = _os.getenv("NEO4J_PASSWORD")

    if not neo_uri or not neo_user or not neo_pw:
        st.error(
            "Missing NEO4J_URI, NEO4J_USER, "
            "or NEO4J_PASSWORD in .env"
        )
    else:
        from knowledge_graph.builder import (
            GraphBuilder,
        )

        with st.status(
            "Building knowledge graph...",
            expanded=True,
        ) as status:
            st.write(
                f"Connecting to Neo4j at "
                f"{neo_uri}..."
            )
            log(f"Neo4j: connecting to {neo_uri}")

            builder = GraphBuilder(
                neo_uri, neo_user, neo_pw
            )

            try:
                if clear_graph:
                    st.write(
                        "Clearing existing graph..."
                    )
                    log("Clearing graph...")
                    builder.clear()

                st.write(
                    "Loading data + computing "
                    "sentiment..."
                )
                log("Building knowledge graph...")

                builder.build()

                log("Knowledge graph built")
                status.update(
                    label="Knowledge graph built",
                    state="complete",
                )
            except Exception as e:
                log(f"Neo4j error: {e}")
                status.update(
                    label=f"Error: {e}",
                    state="error",
                )
            finally:
                builder.close()

st.caption(
    "Creates 8 node types (User, Subreddit, "
    "Post, Comment, Topic, Company, Model, "
    "TimePeriod) and 7 relationship types with "
    "VADER sentiment scores."
)
