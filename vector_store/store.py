import os

from dotenv import load_dotenv

from vector_store.embeddings import (
    EmbeddingModel,
)
from vector_store.chunker import chunk_text
from preprocessing.extractor import (
    assign_time_window,
)

load_dotenv()

POSTS_COLLECTION = "reddit_posts"
COMMENTS_COLLECTION = "reddit_comments"

BATCH_SIZE = 50
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

_chroma_client = None


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    import chromadb

    api_key = os.getenv("CHROMA_API_KEY")
    tenant = os.getenv("CHROMA_TENANT")
    database = os.getenv("CHROMA_DATABASE")

    if api_key and tenant and database:
        _chroma_client = chromadb.CloudClient(
            tenant=tenant,
            database=database,
            api_key=api_key,
        )
    else:
        _chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )

    return _chroma_client


class VectorStore:

    def __init__(
        self, embedding_model: EmbeddingModel
    ):
        self._client = _get_chroma_client()
        self._embedder = embedding_model

        self.posts = (
            self._client.get_or_create_collection(
                name=POSTS_COLLECTION,
                metadata={
                    "hnsw:space": "cosine"
                },
            )
        )

        self.comments = (
            self._client.get_or_create_collection(
                name=COMMENTS_COLLECTION,
                metadata={
                    "hnsw:space": "cosine"
                },
            )
        )

    # -----------------------------------------
    # Ingest
    # -----------------------------------------

    def add_posts(
        self, posts: list[dict]
    ) -> int:
        added = 0

        all_ids = []
        all_docs = []
        all_metas = []

        for post in posts:
            post_id = post.get("post_id")
            if not post_id:
                continue

            title = post.get("title") or ""
            body = post.get("body") or ""
            text = f"{title}\n\n{body}".strip()

            if not text:
                continue

            chunks = chunk_text(
                text, CHUNK_SIZE, CHUNK_OVERLAP
            )

            for ci, chunk in enumerate(chunks):
                chunk_id = (
                    f"{post_id}_chunk_{ci}"
                    if len(chunks) > 1
                    else post_id
                )

                meta = _build_post_metadata(
                    post, ci, len(chunks)
                )

                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_metas.append(meta)

        for i in range(
            0, len(all_ids), BATCH_SIZE
        ):
            batch_ids = (
                all_ids[i : i + BATCH_SIZE]
            )
            batch_docs = (
                all_docs[i : i + BATCH_SIZE]
            )
            batch_metas = (
                all_metas[i : i + BATCH_SIZE]
            )

            embeddings = self._embedder.embed(
                batch_docs
            )

            self.posts.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
            )
            added += len(batch_ids)

            print(
                f"  [+] Posts batch "
                f"{i // BATCH_SIZE + 1}: "
                f"{len(batch_ids)} chunks added"
            )

        return added

    def add_comments(
        self, comments: list[dict]
    ) -> int:
        added = 0

        all_ids = []
        all_docs = []
        all_metas = []

        for comment in comments:
            comment_id = comment.get(
                "comment_id"
            )
            if not comment_id:
                continue

            body = comment.get("body") or ""
            if len(body) < 10:
                continue

            meta = _build_comment_metadata(
                comment
            )

            all_ids.append(comment_id)
            all_docs.append(body)
            all_metas.append(meta)

        for i in range(
            0, len(all_ids), BATCH_SIZE
        ):
            batch_ids = (
                all_ids[i : i + BATCH_SIZE]
            )
            batch_docs = (
                all_docs[i : i + BATCH_SIZE]
            )
            batch_metas = (
                all_metas[i : i + BATCH_SIZE]
            )

            embeddings = self._embedder.embed(
                batch_docs
            )

            self.comments.upsert(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
            )
            added += len(batch_ids)

            print(
                f"  [+] Comments batch "
                f"{i // BATCH_SIZE + 1}: "
                f"{len(batch_ids)} added"
            )

        return added

    # -----------------------------------------
    # Query — single embed, reuse for both
    # -----------------------------------------

    def _embed_query(
        self, query: str
    ) -> list[float]:
        return self._embedder.embed([query])[0]

    def query_posts(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
        query_embedding: list[float] = None,
    ) -> dict:
        if query_embedding is None:
            query_embedding = self._embed_query(
                query
            )

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        return self.posts.query(**kwargs)

    def query_comments(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
        query_embedding: list[float] = None,
    ) -> dict:
        if query_embedding is None:
            query_embedding = self._embed_query(
                query
            )

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where

        return self.comments.query(**kwargs)

    def query_all(
        self,
        query: str,
        n_results: int = 10,
        where: dict = None,
    ) -> dict:
        emb = self._embed_query(query)

        return {
            "posts": self.query_posts(
                query, n_results, where,
                query_embedding=emb,
            ),
            "comments": self.query_comments(
                query, n_results, where,
                query_embedding=emb,
            ),
        }

    # -----------------------------------------
    # Filter helpers
    # -----------------------------------------

    def search_by_time_window(
        self,
        query: str,
        time_window: str,
        n_results: int = 10,
    ) -> dict:
        return self.query_all(
            query, n_results,
            where={"time_window": time_window},
        )

    def search_by_subreddit(
        self,
        query: str,
        subreddit: str,
        n_results: int = 10,
    ) -> dict:
        emb = self._embed_query(query)

        posts = self.query_posts(
            query, n_results,
            where={"subreddit": subreddit},
            query_embedding=emb,
        )
        comments = self.query_comments(
            query, n_results,
            query_embedding=emb,
        )

        return {
            "posts": posts,
            "comments": comments,
        }

    def search_by_author(
        self,
        query: str,
        author: str,
        n_results: int = 10,
    ) -> dict:
        return self.query_all(
            query, n_results,
            where={"author": author},
        )

    # -----------------------------------------
    # Stats
    # -----------------------------------------

    def stats(self) -> dict:
        return {
            "posts": self.posts.count(),
            "comments": self.comments.count(),
        }


# -----------------------------------------
# Metadata builders
# -----------------------------------------

def _build_post_metadata(
    post: dict,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> dict:
    topics = post.get("topics", [])

    return {
        "post_id": (
            post.get("post_id") or ""
        ),
        "subreddit": (
            post.get("subreddit") or ""
        ),
        "author": post.get("author") or "",
        "time_window": (
            post.get("time_window") or ""
        ),
        "created_at": (
            post.get("created_at") or ""
        ),
        "url": post.get("url") or "",
        "topics": ", ".join(topics),
        "search_topic": (
            post.get("search_topic") or ""
        ),
        "comment_count": (
            post.get("comment_count") or 0
        ),
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "type": "post",
        "chunk_type": "post",
    }


def _build_comment_metadata(
    comment: dict,
) -> dict:
    created_at = (
        comment.get("created_at") or ""
    )
    time_window = assign_time_window(
        created_at
    )

    return {
        "comment_id": (
            comment.get("comment_id") or ""
        ),
        "post_id": (
            comment.get("post_id") or ""
        ),
        "parent_id": (
            comment.get("parent_id") or ""
        ),
        "author": (
            comment.get("author") or ""
        ),
        "created_at": created_at,
        "time_window": time_window,
        "depth": comment.get("depth", 0),
        "type": "comment",
        "chunk_type": "comment",
    }
