"""
detector/semantic_check.py

Layer 2 — Semantic relevance check.

Hybrid search combining:
  - Dense ANN search via Milvus HNSW (cosine similarity)
  - Sparse keyword search via BM25 (rank_bm25)
  - Fused via Reciprocal Rank Fusion (RRF)

Returns the top RRF score and top-k matching corpus cases.
A low RRF score means the citation context is semantically inconsistent
with real Fourth Amendment cases — flagged as SUSPICIOUS.

Usage (standalone test):
  python -m detector.semantic_check

Called by detector/pipeline.py at query time with:
  result = semantic_check(context_text)
  # result.rrf_score  — float, higher = more semantically consistent
  # result.top_matches — list of dicts with case metadata
  # result.is_relevant — bool (rrf_score >= RRF_THRESHOLD)
"""

import logging
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    PROCESSED_DIR,
    MILVUS_URI,
    MILVUS_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    HNSW_EF,
    TOP_K,
    RRF_K,
    BM25_INDEX_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# RRF_THRESHOLD is added to config in Week 8 after tuning on validation set.
# Default here is conservative — tune down if too many false SUSPICIOUS verdicts.
RRF_THRESHOLD = 0.02   # floor below which a citation is flagged SUSPICIOUS

METADATA_PATH = Path(PROCESSED_DIR) / "cases_cleaned.parquet"
BM25_PATH     = Path(BM25_INDEX_PATH)

# Metadata columns loaded into memory for top_matches enrichment
METADATA_COLS = ["case_id", "case_name", "court_id", "date_filed", "cite_count"]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SemanticResult:
    rrf_score:        float
    top_dense_score:  float   # cosine similarity of top dense hit (0.0–1.0)
    is_relevant:      bool
    top_matches:      list[dict] = field(default_factory=list)
    # top_matches entries:
    #   case_id, case_name, court_id, date_filed, cite_count,
    #   dense_score, bm25_score, rrf_score


# ---------------------------------------------------------------------------
# Module-level singletons — loaded once, reused across calls
# ---------------------------------------------------------------------------

_bm25        = None
_bm25_ids    = None   # list[int] — position i → case_id (BM25 corpus order)
_metadata_df = None   # DataFrame keyed on case_id for fast lookup
_embedder    = None   # legal-bert tokenizer + model
_milvus      = None   # MilvusClient


def _load_bm25():
    global _bm25, _bm25_ids
    if _bm25 is not None:
        return
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {BM25_PATH}. "
            "Run embeddings/bm25_index.py first."
        )
    log.info(f"Loading BM25 index from {BM25_PATH}")
    with open(BM25_PATH, "rb") as f:
        payload  = pickle.load(f)
    _bm25     = payload["bm25"]
    _bm25_ids = payload["case_ids"]
    log.info(f"  BM25 index loaded: {len(_bm25_ids):,} documents")


def _load_metadata():
    global _metadata_df
    if _metadata_df is not None:
        return
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}. "
            "Run preprocessing/clean_text.py first."
        )
    log.info(f"Loading case metadata from {METADATA_PATH}")
    df = pd.read_parquet(METADATA_PATH, columns=METADATA_COLS)
    _metadata_df = df.set_index("case_id")
    log.info(f"  Metadata loaded: {len(_metadata_df):,} cases")


def _load_embedder():
    global _embedder
    if _embedder is not None:
        return
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch required. "
            "Run: pip install transformers torch"
        )
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model     = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    _embedder = (tokenizer, model)
    log.info("  Embedder loaded")


def _load_milvus():
    global _milvus
    if _milvus is not None:
        return
    try:
        from pymilvus import MilvusClient
    except ImportError:
        raise ImportError("pymilvus required. Run: pip install pymilvus")
    log.info(f"Connecting to Milvus: {MILVUS_URI}")
    _milvus = MilvusClient(uri=MILVUS_URI)
    _milvus.load_collection(MILVUS_COLLECTION)
    log.info(f"  Milvus collection loaded: {MILVUS_COLLECTION}")


def _load_all():
    """Load all singletons. Called once on first semantic_check() invocation."""
    _load_bm25()
    _load_metadata()
    _load_embedder()
    _load_milvus()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed(text: str) -> list[float]:
    """
    Embed a single text string using legal-bert.
    Returns an L2-normalized 768-dim vector as a Python list.
    Truncates to 512 tokens (legal-bert max).
    """
    import torch

    tokenizer, model = _embedder
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean-pool over token dimension → (1, 768)
    token_embeddings = outputs.last_hidden_state          # (1, seq_len, 768)
    attention_mask   = inputs["attention_mask"]           # (1, seq_len)
    mask_expanded    = attention_mask.unsqueeze(-1).float()
    sum_embeddings   = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask         = mask_expanded.sum(dim=1).clamp(min=1e-9)
    vector           = (sum_embeddings / sum_mask).squeeze(0).numpy()  # (768,)

    # L2 normalize — required for cosine similarity in Milvus
    norm   = np.linalg.norm(vector)
    vector = vector / norm if norm > 1e-9 else vector

    return vector.tolist()


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

def _tokenize_query(text: str) -> list[str]:
    """
    Lightweight query tokenizer — lowercase + split.
    No lemmatization at query time to keep latency low;
    BM25 recall is sufficient with simple tokenization for short queries.
    """
    import re
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def _bm25_search(query_text: str, top_k: int) -> list[tuple[int, float]]:
    """
    Run BM25 search over the corpus.
    Returns list of (case_id, bm25_score) sorted descending, top_k entries.
    """
    query_tokens = _tokenize_query(query_text)
    scores       = _bm25.get_scores(query_tokens)          # ndarray, len = corpus size
    top_indices  = np.argsort(scores)[::-1][:top_k]
    return [(int(_bm25_ids[i]), float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Dense ANN search
# ---------------------------------------------------------------------------

def _dense_search(query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
    """
    Run HNSW ANN search in Milvus.
    Returns list of (case_id, cosine_score) sorted descending, top_k entries.
    """
    results = _milvus.search(
        collection_name = MILVUS_COLLECTION,
        data            = [query_vector],
        anns_field      = "embedding",
        search_params   = {
            "metric_type": "COSINE",
            "params":      {"ef": HNSW_EF},
        },
        limit           = top_k,
        output_fields   = ["case_id"],
    )
    hits = results[0]  # single query → single result list
    return [(hit["entity"]["case_id"], float(hit["distance"])) for hit in hits]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense_hits: list[tuple[int, float]],
    sparse_hits: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Fuse dense and sparse ranked lists via Reciprocal Rank Fusion.

    rrf_score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))

    Documents appearing in only one list get 0 for the missing rank component.
    Returns list of (case_id, rrf_score) sorted descending.
    """
    # Build rank maps (1-indexed)
    dense_rank  = {cid: rank for rank, (cid, _) in enumerate(dense_hits,  start=1)}
    sparse_rank = {cid: rank for rank, (cid, _) in enumerate(sparse_hits, start=1)}

    all_ids = set(dense_rank) | set(sparse_rank)
    scores  = {}
    for cid in all_ids:
        dr = dense_rank.get(cid,  len(dense_hits)  + k)   # penalty if absent
        sr = sparse_rank.get(cid, len(sparse_hits) + k)
        scores[cid] = 1.0 / (k + dr) + 1.0 / (k + sr)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

def _enrich(
    fused: list[tuple[int, float]],
    dense_hits: list[tuple[int, float]],
    sparse_hits: list[tuple[int, float]],
    top_k: int,
) -> list[dict]:
    """
    Attach case metadata to top-k fused results.
    Returns list of dicts with case fields + individual scores.
    """
    dense_scores  = dict(dense_hits)
    sparse_scores = dict(sparse_hits)

    matches = []
    for cid, rrf_score in fused[:top_k]:
        entry = {
            "case_id":    cid,
            "rrf_score":  round(rrf_score, 6),
            "dense_score":  round(dense_scores.get(cid,  0.0), 4),
            "bm25_score":   round(sparse_scores.get(cid, 0.0), 4),
            "case_name":  None,
            "court_id":   None,
            "date_filed": None,
            "cite_count": None,
        }
        if cid in _metadata_df.index:
            row = _metadata_df.loc[cid]
            entry["case_name"]  = str(row["case_name"])
            entry["court_id"]   = str(row["court_id"])
            entry["date_filed"] = str(row["date_filed"])
            entry["cite_count"] = int(row["cite_count"]) if pd.notna(row["cite_count"]) else None
        matches.append(entry)

    return matches

def _case_specific_similarity(case_id: int, query_vector: list[float]) -> float:
    """
    Fetch the stored embedding for a specific case and compute
    cosine similarity against the query vector.
    Returns float in [0, 1], or 0.0 if case not found in Milvus.
    """
    try:
        result = _milvus.query(
            collection_name=MILVUS_COLLECTION,
            filter=f"case_id == {case_id}",
            output_fields=["embedding"],
            limit=1
        )
        if not result:
            log.warning("case_id %d not found in Milvus — defaulting to 0.0", case_id)
            return 0.0

        case_vector = np.array(result[0]["embedding"])
        query_array = np.array(query_vector)

        # Cosine similarity (vectors are already L2-normalized)
        similarity = float(np.dot(query_array, case_vector))
        return max(0.0, similarity)  # clamp negatives to 0

    except Exception as e:
        log.error("Milvus query error for case_id %d: %s", case_id, e)
        return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# Case-specific similarity threshold
CASE_SIMILARITY_THRESHOLD = 0.80  # tune if needed

def semantic_check(context_text: str, top_k: int = TOP_K, case_id: int = None) -> SemanticResult:
    _load_all()

    # 1. Embed context
    query_vector = _embed(context_text)

    # 2. Dense ANN search (still needed for top_matches + RRF)
    dense_hits = _dense_search(query_vector, top_k=top_k)

    # 3. BM25 sparse search
    sparse_hits = _bm25_search(context_text, top_k=top_k)

    # 4. RRF fusion
    fused = _rrf_fuse(dense_hits, sparse_hits)

    # 5. Top RRF score (kept for logging/debugging)
    top_rrf_score   = fused[0][1] if fused else 0.0
    top_dense_score = dense_hits[0][1] if dense_hits else 0.0

    # 6. Case-specific relevance — compare proposition against THIS case's embedding
    if case_id is not None:
        case_sim    = _case_specific_similarity(case_id, query_vector)
        is_relevant = case_sim >= CASE_SIMILARITY_THRESHOLD
        log.info(
            "Layer 2 case-specific similarity: %.4f (threshold=%.2f) → %s",
            case_sim, CASE_SIMILARITY_THRESHOLD, is_relevant
        )
    else:
        # Fallback to RRF if no case_id provided
        case_sim    = None
        is_relevant = top_rrf_score >= RRF_THRESHOLD

    # 7. Enrich top-k results with metadata
    top_matches = _enrich(fused, dense_hits, sparse_hits, top_k)

    return SemanticResult(
        rrf_score       = top_rrf_score,
        top_dense_score = top_dense_score,
        is_relevant     = is_relevant,
        top_matches     = top_matches,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with a real Fourth Amendment context
    test_queries = [
        # Real doctrine — should score high
        (
            "The Fourth Amendment protects individuals from unreasonable searches "
            "and seizures. Under Terry v. Ohio, an officer may briefly detain a "
            "person based on reasonable articulable suspicion that criminal activity "
            "is afoot, even without probable cause for a full arrest."
        ),
        # Gibberish — should score low
        (
            "The defendant argued that the rainbow protocol under Zephyr v. Cloudbase "
            "established a new standard for quantum warrant exceptions in 2031."
        ),
    ]

    labels = ["Real Fourth Amendment doctrine", "Gibberish / hallucinated"]

    for label, query in zip(labels, test_queries):
        print(f"\n{'='*60}")
        print(f"Query type: {label}")
        print(f"Text: {query[:80]}...")
        result = semantic_check(query)
        print(f"RRF score:    {result.rrf_score:.6f}")
        print(f"Dense score:  {result.top_dense_score:.4f}")
        print(f"Is relevant:  {result.is_relevant}")
        print(f"Top matches:")
        for i, m in enumerate(result.top_matches, 1):
            print(
                f"  #{i}  {m['case_name']}  "
                f"({m['court_id']}, {m['date_filed']})  "
                f"rrf={m['rrf_score']:.4f}  dense={m['dense_score']:.4f}  "
                f"bm25={m['bm25_score']:.2f}"
            )