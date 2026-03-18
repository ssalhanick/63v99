"""
embeddings/bm25_index.py

Builds a BM25 sparse index over the tokenized corpus and serializes it to disk.

The BM25 index is the sparse half of Week 5 hybrid search. At query time,
semantic_check.py loads this index and runs keyword search alongside Milvus
ANN search, fusing both ranked lists via Reciprocal Rank Fusion.

Input:  data/processed/cases_tokenized.parquet  (case_id | tokens)
Output: data/processed/bm25_index.pkl           (BM25Okapi + case_id list)

Run:
  python -m embeddings.bm25_index

Notes:
  - rank_bm25 must be installed: pip install rank-bm25
  - The pkl file stores a dict with keys: 'bm25' and 'case_ids'
  - case_ids list preserves the corpus order that BM25 uses internally —
    this mapping is required at query time to convert BM25 rank → case_id
  - Index is small (~50MB for 1,353 cases) and loads in under 1 second
"""

import logging
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, BM25_INDEX_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

INPUT_PATH  = Path(PROCESSED_DIR) / "cases_tokenized.parquet"
OUTPUT_PATH = Path(BM25_INDEX_PATH)


def main() -> None:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.error("rank_bm25 not installed. Run: pip install rank-bm25")
        raise

    # --- Load tokenized corpus ---
    if not INPUT_PATH.exists():
        log.error(
            f"Missing input: {INPUT_PATH}. "
            "Run preprocessing/tokenize_bm25.py first."
        )
        sys.exit(1)

    log.info(f"Loading tokenized corpus from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    log.info(f"  {len(df):,} cases loaded")

    # Preserve corpus order — BM25 returns scores by position, not case_id
    case_ids = df["case_id"].tolist()
    corpus   = [list(t) for t in df["tokens"]]   # numpy array → list of list of str

    # Validate a sample
    sample_tokens = corpus[0]
    if not isinstance(sample_tokens, list) or not isinstance(sample_tokens[0], str):
        log.error(
            "tokens column is not list-of-str. "
            "Re-run preprocessing/tokenize_bm25.py."
        )
        sys.exit(1)

    log.info(f"  Sample: case_id={case_ids[0]}, {len(sample_tokens)} tokens")

    # --- Build BM25 index ---
    log.info("Building BM25Okapi index …")
    bm25 = BM25Okapi(corpus)
    log.info(f"  Index built over {len(corpus):,} documents")

    # --- Serialize ---
    payload = {
        "bm25":     bm25,      # BM25Okapi object
        "case_ids": case_ids,  # list[int] — position i → case_id
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    log.info(f"Saved → {OUTPUT_PATH}  ({size_mb:.1f} MB)")

    # --- Smoke test ---
    log.info("Running smoke test …")
    query_tokens = ["warrant", "probable", "cause", "search", "seizure"]
    scores       = bm25.get_scores(query_tokens)
    top_indices  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    log.info("  Top-3 BM25 hits for ['warrant', 'probable', 'cause', 'search', 'seizure']:")
    for rank, idx in enumerate(top_indices, 1):
        log.info(f"    #{rank}  case_id={case_ids[idx]}  score={scores[idx]:.4f}")

    log.info("Done ✅")


if __name__ == "__main__":
    main()