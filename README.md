# Reddit Temporal Knowledge RAG

A hybrid retrieval system over Reddit discussions about AI/LLM topics. It scrapes Reddit posts and comments across time windows, structures the same content as both a **vector index** (ChromaDB) and a **temporal knowledge graph** (Neo4j), and answers natural-language questions by fusing graph traversal with semantic vector search — including questions about how a topic, sentiment, or community changed over time.

## Table of Contents
- [Architecture](#architecture)
- [Tech Stack & Reasoning](#tech-stack--reasoning)
- [Setup (clone → configure → run in under 10 minutes)](#setup-clone--configure--run-in-under-10-minutes)
- [Demo Script](#demo-script)
- [Known Deviations from the Spec](#known-deviations-from-the-spec)

---

## Architecture

The system uses a dual-retrieval architecture: Reddit discussions are indexed both as vectors in ChromaDB and as entities/relationships in Neo4j. Every query triggers both retrieval paths in parallel. An LLM router classifies the query type and adjusts fusion weights, while Reciprocal Rank Fusion combines graph and vector results before answer generation.

```
                    ┌─────────────────────┐
                    │   ingestion/        │
                    │  Tavily + Reddit     │
                    │  JSON API search     │──► data/raw, data/parsed
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  preprocessing/      │
                    │  clean → canonicalize │──► data/normalized/
                    │  → extract → normalize│      posts.jsonl
                    │  (+ optional LLM      │      comments.jsonl
                    │     entity extraction)│
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌─────────────────────┐          ┌──────────────────────┐
   │  vector_store/       │          │  knowledge_graph/      │
   │  chunk → embed →     │          │  VADER sentiment →      │
   │  ChromaDB Cloud/local │          │  Neo4j nodes/edges →     │
   │                       │          │  influence/sentiment/     │
   │                       │          │  trend snapshots          │
   └──────────┬────────────┘          └──────────┬───────────────┘
              │                                  │
              └────────────────┬─────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   retrieval/         │
                    │  Query Router (LLM)  │
                    │  Vector + Graph      │
                    │  retrieval in        │
                    │  parallel, fused     │
                    │  with weighted RRF   │
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
                    │  LLM answer with      │
                    │  citations + graph     │
                    │  analytics             │
                    └─────────────────────┘
```

**Why this shape:** ingestion and preprocessing are a shared pipeline that feeds **two parallel representations** of the same underlying data — a vector index for semantic similarity, and a knowledge graph for relationships, entities, and time-scoped traversal. Neither index is privileged at query time: a **Query Router** (LLM) classifies the question, but per the assignment's "single query triggers both retrievers" requirement, the classification informs *fusion weights*, not whether a retriever runs — vector and graph retrieval always execute in parallel, and their ranked result lists are combined with **Reciprocal Rank Fusion**. This means the system never has to guess wrong and silently skip a useful signal; the worst case is a retriever contributing little to the fused ranking rather than being excluded outright.

The same architectural split (shared ingestion → dual index → routed dual retrieval → fusion → LLM synthesis) is what every module in the repo is organized around:

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `ingestion/` | Find Reddit URLs (Tavily for current/trending, Reddit's own JSON API for historical + paginated + exact-timestamp results), scrape with crawl4ai, parse into structured posts/comments |
| Preprocessing | `preprocessing/` | Strip noise, extract timestamps/entities/topics, canonicalize topic aliases, optionally enrich with LLM extraction, normalize into flat tables |
| Vector index | `vector_store/` | Chunk + embed + store in ChromaDB (Cloud or local) |
| Knowledge graph | `knowledge_graph/` | Sentiment-score every post/comment, build the Neo4j graph (10 node types, 10 relationship types), compute influence scores + sentiment/trend snapshots |
| Retrieval | `retrieval/` | Query router, 6 retrieval methods (vector/tfidf/bm25/multi-query/ensemble/hybrid), RRF fusion, LLM answer generation with citations |
| Evaluation | `evaluation/` | Recall@K / Precision@K / MRR benchmark comparing all retrieval methods |

---

## Tech Stack & Reasoning

| Concern | Choice | Why |
|---|---|---|
| Reddit data source | **Tavily** (current/trending web search) + **Reddit's own JSON API** (`reddit.com/r/.../search.json`, historical + paginated, no auth) + **crawl4ai** (web scraping for full thread content) | I initially tried to use the official Reddit API (PRAW) as the assignment specifies.Due to Reddit developer application registration limitations during development, I used Reddit's public JSON API combined with crawl4ai and Tavily as a fallback. — the app creation page returned a warning requiring acceptance of policies that never resolved, regardless of account age or verification status. After exhausting that path, I pivoted to a two-source approach: **Tavily** finds relevant Reddit thread URLs via web search (good for current/trending content), **Reddit's unauthenticated JSON API** (`/search.json`) provides historical results with exact `created_utc` timestamps and real pagination (which is what makes `--year=2024` an actual date filter, not a keyword hint), and **crawl4ai** scrapes the full thread pages to extract post bodies and nested comments. This combination actually covers more ground than PRAW alone would — Tavily catches threads PRAW's search ranking might miss, the JSON API gives the same timestamp precision PRAW would, and scraping captures the rendered comment hierarchy. |
| Vector database | **ChromaDB** — Cloud or local, auto-selected by whether `CHROMA_API_KEY`/`TENANT`/`DATABASE` are set | I selected ChromaDB because the main challenge in this Reddit application is finding relevant posts and comments based on meaning, not just keywords. ChromaDB makes it easy to store embeddings and perform semantic search, while integrating smoothly with Python and requiring minimal infrastructure setup. In our workflow, Reddit posts and comments are embedded and stored in ChromaDB. When a user asks a question, it retrieves the most relevant discussions, which are then enriched with relationship data from the graph database before being sent to the LLM. For the scale of this project, ChromaDB offered the best balance of simplicity, performance, and cost. |
| Embedding model | **sentence-transformers (`all-MiniLM-L6-v2`)** — local, free, default and only backend wired into the CLI | I have selected all-MiniLM-L6-v2 because it's a lightweight, local embedding model that provides good semantic search quality without requiring a paid API. It's fast enough to run on CPU, integrates easily with Python, and is more than sufficient for the scale of this Reddit dataset. |
| LLM | Pluggable: **DeepSeek via HF Inference** (default, free tier), OpenAI `gpt-4o-mini`, Gemini `2.0-flash` | I made the LLM layer pluggable and defaulted to DeepSeek through Hugging Face because it offers a free way to run the project end-to-end. At the same time, the architecture allows swapping in models like GPT-4o-mini or Gemini when stronger reasoning or answer quality is needed. |
| Graph database | **Neo4j** (AuraDB free tier or local Docker) | I choose Neo4j because the data is naturally connected through relationships between users, posts, comments, topics, and time periods. Neo4j's graph model and Cypher queries make multi-hop traversals and trend analysis much simpler and more intuitive than modeling the same relationships in a relational database. |
| Sentiment | **VADER** (`vaderSentiment`) | I used VADER because it's specifically designed for social media text. Reddit content is often informal, short, and full of slang or punctuation, which VADER handles well. It also runs locally with no API costs, making it efficient for scoring every post and comment. |
| Sparse retrieval | **TF-IDF** (scikit-learn) + **BM25** (`rank_bm25`) | I combined TF-IDF and BM25 because embedding search alone can miss exact terms such as model names, acronyms, or version numbers. BM25 complements semantic search by handling keyword-heavy queries more effectively, giving us stronger overall retrieval performance. |
| Topic canonicalization | Hand-built alias map (70+ entries) + fuzzy-match fallback (`difflib`, 80% similarity threshold) | I used an alias mapping approach with fuzzy matching to ensure different variations of the same concept are treated as a single topic. Without that, terms like 'RAG' and 'Retrieval Augmented Generation' would create separate graph nodes and fragment trend analysis and graph traversal results. |
| Hybrid fusion | **Reciprocal Rank Fusion** (`weight/(k+rank)`, k=60), with route-dependent weights | I chose Reciprocal Rank Fusion because it combines results from different retrieval systems without needing their scores to be directly comparable. Since vector search, graph search, and keyword search all produce different scoring scales, RRF provides a reliable way to merge them into a single ranking. |
| Query routing | LLM classification into VECTOR / GRAPH / HYBRID / TEMPORAL (`retrieval/router.py`) | I used the LLM to classify queries as Vector, Graph, Hybrid, or Temporal because different questions require different retrieval strategies. The classification influences retrieval weighting, allowing the system to emphasize the most relevant source while still using all available retrieval methods. |
| Graph algorithms | Composite **influence score** (`posts*5 + comments + received_comments*2 + received_replies*3`) on User nodes; tries Neo4j GDS PageRank first, falls back to pure-Cypher degree centrality | For influence ranking, I used a composite score based on user activity and engagement, and leveraged Neo4j's PageRank when available. Since AuraDB's free tier doesn't include all graph algorithms, I implemented a fallback approach so the system remains fully functional without paid features. |
| LLM framework | None — vanilla SDKs behind a small `ABC` per model type | I chose not to use a framework like LangChain or LlamaIndex because the workflow in this project is relatively straightforward—query routing, retrieval, fusion, and response generation. Using vanilla SDKs with a small abstract base class for different LLM providers kept the architecture lightweight and transparent. It also gave me full control over prompts, retrieval logic, and API interactions while avoiding an additional abstraction layer that wasn't necessary for this project's complexity. |

---

## Setup (clone → configure → run in under 10 minutes)

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

# 4. Configure secrets — create a .env file with:
TAVILY_API_KEY=...                                  
NEO4J_URI=          
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
HUGGINGFACE_API_KEY=...                              
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...

# 5. Pull data, build both indexes
python -m ingestion.main --all          # scrape default topics across all time
python -m preprocessing.main            # clean, canonicalize, normalize
python -m vector_store.main             # embed + index into ChromaDB
python -m knowledge_graph.main          # build the Neo4j knowledge graph

# 6. Run the demo (4 required query types, graph/vector/fused/answer for each)
python demo.py

# 7. Or ask your own questions interactively
python -m retrieval.main --method=hybrid
```

---

## Demo Script

`python demo.py` runs the 4 required query types through the `hybrid` retriever. For each, it prints **graph-only results**, **vector-only results**, the **fused result** (RRF-merged, with the router's classification and per-retriever document counts), and the **final LLM answer**.

| # | Type | Example question |
|---|---|---|
| A | Semantic (vector-dominant) | "What are people's experiences with RAG in 2025?" |
| B | Relationship (graph-dominant) | "Who are the most influential voices discussing AI safety and what companies do they mention?" |
| C | Hybrid (needs both) | "Which open-source LLMs are being discussed alongside RAG and what do people think about them?" |
| D | Temporal comparison | "How has sentiment around RAG systems changed from early 2025 to mid 2026?" |

Run it with `python demo.py` (or `--llm=openai` to swap the LLM), then paste the actual output below.

### A. Semantic — vector-dominant

**Question:** _What are people's experiences with RAG in 2025?_

**Graph-only results:**
```
(paste output here)
```

**Vector-only results:**
```
(paste output here)
```

**Fused result:**
```
(paste output here)
```

**Final answer:**
```
(paste output here)
```

### B. Relationship — graph-dominant

**Question:** _Who are the most influential voices discussing AI safety and what companies do they mention?_

**Graph-only results:**
```
(paste output here)
```

**Vector-only results:**
```
(paste output here)
```

**Fused result:**
```
(paste output here)
```

**Final answer:**
```
(paste output here)
```

### C. Hybrid — needs both

**Question:** _Which open-source LLMs are being discussed alongside RAG and what do people think about them?_

**Graph-only results:**
```
(paste output here)
```

**Vector-only results:**
```
(paste output here)
```

**Fused result:**
```
(paste output here)
```

**Final answer:**
```
(paste output here)
```

### D. Temporal comparison

**Question:** _How has sentiment around RAG systems changed from early 2025 to mid 2026?_

**Graph-only results (temporal analytics):**
```
(paste output here)
```

**Vector-only results (per time window W1/W2/W3):**
```
(paste output here)
```

**Fused result:**
```
(paste output here)
```

**Final answer:**
```
(paste output here)
```

---

## Known Deviations from the Spec

- **Reddit access via Tavily + Reddit JSON API + crawl4ai, not PRAW.** I attempted to use the official Reddit API (PRAW) as the assignment requires. Over two days, I tried repeatedly to register a Reddit developer application, but the app creation page consistently blocked with a policy acceptance warning that never resolved — regardless of account age, email verification, or browser. After exhausting that path, I built a three-layer alternative: (1) **Tavily** web search finds relevant Reddit thread URLs (current/trending), (2) **Reddit's unauthenticated JSON API** (`/r/.../search.json`) provides historical results with exact `created_utc` timestamps, real pagination, and `--year=` date filtering — the same timestamp precision PRAW would give, (3) **crawl4ai** scrapes the full thread pages to extract post bodies and nested comment hierarchies. This combination actually provides broader coverage than PRAW alone: Tavily catches threads PRAW's search ranking might miss, the JSON API covers historical pagination, and scraping captures the full rendered comment tree. If Reddit's developer app registration were to become accessible again, PRAW/AsyncPRAW would still be worth adding for OAuth-gated endpoints (e.g. private subreddits, vote scores) that the unauthenticated API can't reach.
- **Time windows are fixed 6-month/quarterly buckets, not a rolling "last N months."** W1/W2/W3 (vector store) and their quarterly equivalents `Q1_2025`…`Q2_2026` (knowledge graph) are calendar-fixed, so they stay stable across re-runs but don't literally answer "the last 6 months" if run long after the project's reference dates.
- **Query router classifies but doesn't gate.** The router's VECTOR/GRAPH/HYBRID/TEMPORAL classification is logged on every `hybrid` answer and shapes the RRF fusion weights, but both retrievers always run regardless of classification — a deliberate choice to keep "LLM used for query routing" and "a single query should trigger both retrievers" from contradicting each other.
