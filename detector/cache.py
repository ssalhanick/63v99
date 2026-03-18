"""
detector/cache.py

TTLCache for query embeddings and ANN results.

Two caches:
  1. embedding_cache  — keyed on hash(context_text)
                        stores the 768-dim L2-normalized query vector
                        avoids re-running legal-bert for repeated queries

  2. ann_cache        — keyed on (embedding_hash, top_k)
                        stores the raw Milvus ANN search results
                        avoids re-querying Milvus for identical (vector, top_k) pairs

Both caches share TTL and max size from config.

Usage:
  from detector.cache import get_cached_embedding, cache_embedding
  from detector.cache import get_cached_ann, cache_ann

  # Embedding cache
  vec = get_cached_embedding(text)
  if vec is None:
      vec = _embed(text)
      cache_embedding(text, vec)

  # ANN cache
  key = (hash(text), top_k)
  hits = get_cached_ann(key)
  if hits is None:
      hits = _dense_search(vec, top_k)
      cache_ann(key, hits)

Notes:
  - cachetools must be installed: pip install cachetools
  - Cache is in-process only — not shared across workers or API restarts
  - TTL and max size controlled by CACHE_TTL and CACHE_MAX_SIZE in config.py
  - During Week 8 benchmark evaluation (1,353+ queries in a loop),
    the embedding cache eliminates redundant legal-bert inference,
    cutting evaluation time significantly
"""

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CACHE_TTL, CACHE_MAX_SIZE

log = logging.getLogger(__name__)

try:
    from cachetools import TTLCache
except ImportError:
    raise ImportError(
        "cachetools not installed. Run: pip install cachetools"
    )

# ---------------------------------------------------------------------------
# Cache instances
# ---------------------------------------------------------------------------

# Stores query vectors: str_hash → list[float] (768-dim)
embedding_cache: TTLCache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)

# Stores ANN results: (str_hash, top_k) → list[tuple[int, float]]
ann_cache: TTLCache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _text_hash(text: str) -> str:
    """Stable hash of a text string for use as cache key."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def get_cached_embedding(text: str) -> list[float] | None:
    """
    Return cached query vector for text, or None if not cached.
    """
    key = _text_hash(text)
    result = embedding_cache.get(key)
    if result is not None:
        log.debug(f"Embedding cache hit: {key[:8]}…")
    return result


def cache_embedding(text: str, vector: list[float]) -> None:
    """
    Store a query vector in the embedding cache.
    """
    key = _text_hash(text)
    embedding_cache[key] = vector
    log.debug(f"Embedding cached: {key[:8]}…  (cache size: {len(embedding_cache)})")


# ---------------------------------------------------------------------------
# ANN result cache
# ---------------------------------------------------------------------------

def get_cached_ann(text: str, top_k: int) -> list[tuple[int, float]] | None:
    """
    Return cached ANN results for (text, top_k), or None if not cached.
    """
    key = (_text_hash(text), top_k)
    result = ann_cache.get(key)
    if result is not None:
        log.debug(f"ANN cache hit: {key[0][:8]}… top_k={top_k}")
    return result


def cache_ann(text: str, top_k: int, hits: list[tuple[int, float]]) -> None:
    """
    Store ANN search results in the ANN cache.
    """
    key = (_text_hash(text), top_k)
    ann_cache[key] = hits
    log.debug(f"ANN cached: {key[0][:8]}… top_k={top_k}  (cache size: {len(ann_cache)})")


# ---------------------------------------------------------------------------
# Cache stats (useful for debugging + Week 8 evaluation)
# ---------------------------------------------------------------------------

def cache_stats() -> dict[str, Any]:
    """
    Return current cache sizes and capacities.
    """
    return {
        "embedding_cache": {
            "size":    len(embedding_cache),
            "maxsize": embedding_cache.maxsize,
            "ttl":     embedding_cache.ttl,
        },
        "ann_cache": {
            "size":    len(ann_cache),
            "maxsize": ann_cache.maxsize,
            "ttl":     ann_cache.ttl,
        },
    }


def clear_caches() -> None:
    """Clear both caches. Useful for testing."""
    embedding_cache.clear()
    ann_cache.clear()
    log.debug("Both caches cleared")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

    test_text = "The Fourth Amendment protects against unreasonable searches and seizures."
    test_vec  = [0.1] * 768   # dummy vector

    print("=== Embedding cache test ===")
    assert get_cached_embedding(test_text) is None, "Expected cache miss"
    print("Miss: ✅")

    cache_embedding(test_text, test_vec)
    result = get_cached_embedding(test_text)
    assert result == test_vec, "Expected cache hit with correct vector"
    print("Hit:  ✅")

    print("\n=== ANN cache test ===")
    assert get_cached_ann(test_text, top_k=5) is None, "Expected cache miss"
    print("Miss: ✅")

    dummy_hits = [(123, 0.95), (456, 0.91), (789, 0.88)]
    cache_ann(test_text, top_k=5, hits=dummy_hits)
    result = get_cached_ann(test_text, top_k=5)
    assert result == dummy_hits, "Expected cache hit with correct hits"
    print("Hit:  ✅")

    print("\n=== Cache stats ===")
    import json
    print(json.dumps(cache_stats(), indent=2))

    print("\n=== Clear test ===")
    clear_caches()
    assert get_cached_embedding(test_text) is None, "Expected empty after clear"
    assert get_cached_ann(test_text, top_k=5) is None, "Expected empty after clear"
    print("Clear: ✅")

    print("\nAll cache tests passed ✅")