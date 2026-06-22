# Reddit Temporal Knowledge RAG

A hybrid retrieval system over Reddit discussions about AI/LLM topics (RAG, AI safety, open-source models, agents, etc.). It scrapes Reddit posts and comments across time windows, builds a vector index of the content, and answers natural-language questions — including questions about how a topic changed over time.

> **Build status:** Ingestion → Preprocessing → Vector Index are implemented and working end-to-end. The Neo4j temporal knowledge graph, the hybrid (graph + vector) fusion layer, and `demo.py` are **in progress / not yet implemented**. This README documents what exists today and what's planned next, so it stays honest about current capability. See [Roadmap](#roadmap--whats-left) below.

## Table of Contents
- [Architecture](#architecture)
- [Tech Stack & Reasoning](#tech-stack--reasoning)
- [Data Model](#data-model)
- [Setup (clone → configure → run)](#setup-clone--configure--run)
- [Pipeline Commands](#pipeline-commands)
- [Roadmap / What's Left](#roadmap--whats-left)
- [Known Deviations from the Spec](#known-deviations-from-the-spec)

## Architecture

```
┌─────────────┐     ┌────────────────┐     ┌──────────────────┐
│  Ingestion  │ --> │ Preprocessing  │ --> │   Vector Store    │ --> ┌────────────┐
│ (search +   │     │ (clean, parse  │     │   (ChromaDB)       │     │ Retrieval  │
│  scrape)    │     │  entities,     │     │   [DONE]           │ --> │ (LLM       │
└─────────────┘     │  time windows) │     └──────────────────┘     │  answer +  │
                     └────────────────┘     ┌──────────────────┐     │  citations)│
                                        --> │  Knowledge Graph   │ --> └────────────┘
                                             │  (Neo4j)           │
                                             │  [PLANNED]         │
                                             └──────────────────┘
                                             ┌──────────────────┐
                                             │ Hybrid Fusion      │
                                             │ (RRF: graph+vector)│
                                             │ [PLANNED]          │
                                             └──────────────────┘
```

**`ingestion/`** — Finds Reddit post URLs for a set of AI/LLM topics (with year-qualified query variants to bias results toward different time windows), scrapes each thread, and parses the rendered markdown into structured posts + nested comments.

**`preprocessing/`** — Strips Reddit/markdown chrome, extracts timestamps (relative — "3mo ago" — and absolute), authors, mentioned users/subreddits/companies/models, assigns each post/comment to a time window (W1/W2/W3), tags topics by keyword, deduplicates, and writes a data-quality report.

**`vector_store/`** — Chunks post/comment text, embeds it (pluggable embedding backend), and upserts into ChromaDB with metadata (subreddit, author, time window, topics, created_at) so results can be filtered before/after the similarity search.

**`retrieval/`** — Takes a question, queries the vector store (posts + comments in parallel), builds a context block with inline citation markers, and asks an LLM to synthesize an answer. Also supports per-time-window querying and a temporal-comparison mode that retrieves separately per window and asks the LLM to contrast them.

**Knowledge Graph + Hybrid Fusion (planned)** — see [Roadmap](#roadmap--whats-left).

## Tech Stack & Reasoning

| Concern | Choice | Why |
|---|---|---|
| Reddit data source | Tavily / Google CSE search → [crawl4ai](https://github.com/unclecode/crawl4ai) scrape of the rendered thread page | See [Known Deviations](#known-deviations-from-the-spec) — this is a deliberate departure from the PRAW constraint, explained there. |
| Vector database | **ChromaDB** (local, persistent, embedded) | Zero infra to stand up, runs entirely on disk (`./chroma_db`), supports per-field metadata filtering (`where={"time_window": "W2"}`) which the temporal/subreddit/author queries depend on, and is more than sufficient at this dataset's scale (hundreds–low thousands of chunks). A managed vector DB (Pinecone/Qdrant Cloud) would be the right call past that scale, but is unjustified infra for this assignment. |
| Embeddings | Pluggable: `sentence-transformers/all-MiniLM-L6-v2` (default, local, free), OpenAI `text-embedding-3-small`, Gemini `text-embedding-004`, or BGE-small via HF Inference | Default is local so the whole pipeline runs with zero paid API keys; swapping in a hosted model is a one-line flag (`--embed=openai`) when higher embedding quality is wanted. |
| LLM (extraction, retrieval, generation) | Pluggable: DeepSeek-R1 via HF Inference (default, free tier), OpenAI `gpt-4o-mini`, or Gemini `2.0-flash` | Same reasoning as embeddings — default to a free-tier model so reviewers can run the demo without paid keys, swap via `--llm=` flag. |
| Graph database (planned) | **Neo4j** | Cypher's pattern-matching is a natural fit for "who is talking about X and how does that connect to Y over time" traversal queries (multi-hop: Author → Post → Mentions → Entity → other Posts mentioning it). Native support for relationship properties means every edge can carry `created_at` directly, which a property-graph alternative like a plain SQL adjacency table would make far more awkward to query. Neo4j AuraDB's free tier also means no local infra requirement, matching the ChromaDB choice. |
| LLM framework | None (vanilla SDKs behind a small `ABC` per model type) | The actual logic here — chunking, prompt assembly, fusion — is straightforward enough that a framework (LangChain/LlamaIndex) would add abstraction overhead without saving meaningful code, and would obscure exactly what's being sent to each API for review purposes. |

## Data Model

**Time windows** (defined in [`ingestion/utils.py`](ingestion/utils.py) and [`preprocessing/extractor.py`](preprocessing/extractor.py)):
- `W1`: Jan 1 – Jun 30, 2025
- `W2`: Jul 1 – Dec 31, 2025
- `W3`: Jan 1 – Jun 30, 2026

**Current (vector) representation** — each post and comment carries:
`post_id` / `comment_id`, `subreddit`, `author`, `title`/`body`, `created_at`, `time_window`, `topics`, `url`, `parent_id` + `depth` (comments, preserving thread structure), and `extracted` (mentioned URLs, users, subreddits, companies, models).

**Planned graph representation** — entity-relationship model to mirror the same content:
```
(Author)-[:POSTED {created_at}]->(Post)
(Author)-[:COMMENTED {created_at}]->(Comment)
(Comment)-[:REPLIES_TO {created_at}]->(Post|Comment)
(Post|Comment)-[:POSTED_IN]->(Subreddit)
(Post|Comment)-[:MENTIONS {created_at}]->(Entity {type: Model|Company|Topic})
(Post|Comment)-[:HAS_SENTIMENT {score, created_at}]->(Sentiment)
```
Every node and edge gets a `created_at` (and, where applicable, `edited_at`) so the same Cypher traversal can be scoped to a time range — e.g. "entities co-mentioned with RAG in W3 that weren't co-mentioned in W2."

## Setup (clone → configure → run)

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd reddit-rag

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
crawl4ai-setup                # one-time browser setup for crawl4ai

# 4. Configure secrets
cp .env.example .env
# fill in at least TAVILY_API_KEY (or GOOGLE_CSE_ID) to scrape new data

# 5. Run the pipeline (or skip to step 6 if you already have data/ from a teammate)
python -m ingestion.main --all        # scrape across topics + time windows
python -m preprocessing.main          # clean, extract, normalize, quality report
python -m vector_store.main           # embed + index into ChromaDB

# 6. Ask questions
python -m retrieval.main
```

All steps after `.env` is filled in run unattended; the only manual input is typing a question at the `retrieval.main` prompt (or a search query at `ingestion.main` if not using `--all`).

## Pipeline Commands

```bash
# Ingestion
python -m ingestion.main                  # single ad-hoc query (Tavily)
python -m ingestion.main --google         # single query via Google CSE
python -m ingestion.main --all            # all topics × time-qualified queries (Tavily)
python -m ingestion.main --all --google   # same, via Google CSE

# Preprocessing
python -m preprocessing.main              # process everything in data/parsed/
python -m preprocessing.main <file.json>  # process a single parsed file

# Vector store
python -m vector_store.main                         # embed + store (sentence-transformer)
python -m vector_store.main --embed=openai           # ...with OpenAI embeddings
python -m vector_store.main --embed=gemini           # ...with Gemini embeddings
python -m vector_store.main --search                 # interactive similarity search only

# Retrieval (vector-only today; will route through graph + fusion once that lands)
python -m retrieval.main                  # interactive Q&A (DeepSeek LLM, default)
python -m retrieval.main --llm=openai     # ...with GPT-4o-mini
python -m retrieval.main --llm=gemini     # ...with Gemini 2.0 Flash
python -m retrieval.main --temporal       # interactive temporal-comparison mode
```

Inside `retrieval.main`'s interactive prompt: `/temporal <question>` compares across W1/W2/W3, `/subreddit <name> <question>` filters to one subreddit, `/quit` exits.

## Roadmap / What's Left

The vector half of the system (ingestion → preprocessing → ChromaDB → LLM answer with citations) is complete and is what `retrieval.main` runs today. Still to build:

- [ ] **Neo4j knowledge graph ingestion** — LLM-driven entity/relationship/sentiment extraction over the same normalized posts/comments, written to Neo4j with temporal properties on every node and edge.
- [ ] **Graph retriever** — Cypher traversal queries (e.g. influential-author lookup, entity co-mention paths, subreddit-leadership-over-time) parallel to the existing vector retriever.
- [ ] **Hybrid fusion** — Reciprocal Rank Fusion across the graph and vector result lists into one ranked, deduplicated list, plus the query-routing step that decides which retriever(s) a given question needs.
- [ ] **`demo.py`** — runs the four required example queries (semantic, graph-traversal, hybrid, time-series comparison) and prints graph-only / vector-only / fused results side by side for each.
- [ ] **`.env.example`** — added alongside this README update; will gain `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` once the graph layer lands.

## Known Deviations from the Spec

- **Reddit access via search + scrape, not PRAW.** The assignment calls for the official Reddit API. This project instead searches Tavily/Google CSE for Reddit thread URLs restricted to a fixed subreddit list, then scrapes the rendered page with crawl4ai. Reasoning: PRAW's listing endpoints (`new`, `top`, `search`) don't let you reliably target arbitrary historical windows months apart — Reddit's API returns recent/relevance-ranked results, not an arbitrary date-range query. Qualifying search queries by year (`"RAG 2025"`, `"RAG 2026"`) was a more direct way to bias results toward distinct time windows for this assignment's "multiple time windows" requirement. The tradeoff: scraping is slower, more fragile to markup changes, and self-reports relative timestamps ("3mo ago") that have to be reconstructed rather than reading a reliable `created_utc` field directly off the API — visible in the `missing_timestamps` accounting in the quality report. If this were going to production, PRAW/AsyncPRAW would be the right call specifically *because* of that reliability.
- **Time windows are fixed 6-month buckets, not a rolling "last 6 months."** W1/W2/W3 are calendar-fixed (Jan–Jun 2025, Jul–Dec 2025, Jan–Jun 2026) rather than computed relative to "now," so they stay stable across re-runs but don't literally answer "the last 6 months" if run much later than originally designed.
