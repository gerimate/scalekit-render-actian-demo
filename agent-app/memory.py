"""
Per-user VectorAI DB memory.

Isolation contract: every call to this module is scoped to a single
user_id. The app (not VectorAI DB) enforces the boundary — each user
owns exactly one collection named `user-{user_id}-memories` and no
code in this module ever touches another user's collection.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from actian_vectorai import Distance, VectorAIClient, VectorParams
from actian_vectorai.exceptions import (
    VectorAIError,
    CollectionExistsError,
    CollectionNotFoundError,
    CollectionNotReadyError,
)
from actian_vectorai.exceptions import ConnectionError as VAIConnectionError
from langchain_actian_vectorai import ActianVectorAIVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

log = logging.getLogger(__name__)

VECTORAI_URL: str = os.getenv("VECTORAI_DB_URL", "localhost:6574")

# All per-user collections share the same embedding model and dimension so
# that the same vector space is used across the board.
# BAAI/bge-small-en-v1.5: 384-dim, no API key, ~90 MB download on first use.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384
DISTANCE = Distance.Cosine

# ---------------------------------------------------------------------------
# Singletons — connected/loaded once per process.
# ---------------------------------------------------------------------------

_client: VectorAIClient | None = None
_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        log.info("Loading embedding model %s (first request only) …", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_client() -> VectorAIClient:
    global _client
    if _client is None:
        log.info("Connecting to VectorAI DB at %s", VECTORAI_URL)
        _client = VectorAIClient(VECTORAI_URL)
        _client.connect()
    return _client


def _reset_client() -> VectorAIClient:
    """Drop the singleton and reconnect. Call after any connection-level error."""
    global _client
    _client = None
    return _get_client()


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def collection_name(user_id: str) -> str:
    """Return the canonical collection name for a user."""
    return f"user-{user_id}-memories"


def get_or_create_user_collection(user_id: str) -> str:
    """
    Ensure the per-user collection exists and return its name.

    Retries once on connection-level errors (e.g. after a DB restart)
    by resetting the singleton and reconnecting.
    """
    name = collection_name(user_id)
    for attempt in range(2):
        try:
            client = _get_client()
            client.collections.create(
                name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=DISTANCE),
            )
            log.info("Created collection %s", name)
            return name
        except CollectionExistsError:
            log.debug("Collection %s already exists — OK", name)
            return name
        except VAIConnectionError as exc:
            if attempt == 0:
                log.warning("VectorAI connection error, reconnecting: %s", exc)
                _reset_client()
                continue
            raise
        except Exception as exc:
            if "already exists" in str(exc).lower():
                log.debug("Collection %s already exists (string match) — OK", name)
                return name
            raise
    return name


# ---------------------------------------------------------------------------
# Vector store factory
# ---------------------------------------------------------------------------

def get_user_store(user_id: str) -> ActianVectorAIVectorStore:
    """
    Return an ActianVectorAIVectorStore scoped to this user's collection.

    The app calls get_or_create_user_collection first so the collection is
    always present before the store is initialized.
    """
    col = get_or_create_user_collection(user_id)
    return ActianVectorAIVectorStore(
        client=_get_client(),
        collection_name=col,
        embedding=_get_embeddings(),
    )


# ---------------------------------------------------------------------------
# Convenience read/write helpers used by LangGraph nodes
# ---------------------------------------------------------------------------

def recall_memories(user_id: str, query: str, k: int = 5) -> list[str]:
    """Return up to *k* relevant memory strings for this user."""
    if not query.strip():
        return []
    store = get_user_store(user_id)
    docs = store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def remember_turn(user_id: str, human_text: str, ai_text: str) -> list[str]:
    """
    Persist a single conversation turn as one document in the user's collection.

    Returns the list of IDs assigned by VectorAI DB.
    Uses strict=True on delete during tests — here we only add, never delete,
    so strict mode is not applicable. Keep this note so test code can pass
    strict=True to delete_by_ids if needed.
    """
    store = get_user_store(user_id)
    turn = f"User: {human_text}\nAssistant: {ai_text}"
    ids = store.add_texts(
        [turn],
        metadatas=[{"user_id": user_id}],
    )
    log.debug("Stored memory for %s: %s", user_id, ids)
    return ids
