"""
embeddings/milvus_index.py

Bulk-inserts all L2-normalized embeddings into Milvus Lite and builds an
HNSW index (M=16, ef_construction=200).

CRITICAL ordering:
  1. Insert ALL vectors first
  2. Then call create_index()
  Milvus HNSW quality degrades if indexed incrementally.

Input:  data/processed/embeddings.parquet  (case_id | embedding)
Output: milvus_verit.db  (Milvus Lite local DB)

Run:
  python -m embeddings.milvus_index [--drop-existing]

Flags:
  --drop-existing   Drop the collection if it already exists and rebuild from scratch.
                    Omit to skip cases already in the collection (idempotent upsert).
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    PROCESSED_DIR, MILVUS_DB_PATH, MILVUS_COLLECTION,
    EMBEDDING_DIM, HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH   = Path(PROCESSED_DIR) / "embeddings.parquet"
INSERT_BATCH = 500   # vectors per batch — avoids memory spikes


def _get_collection(client, drop_existing: bool):
    """Return (or create) the Milvus collection. Drops first if requested."""
    from pymilvus import MilvusClient, DataType

    exists = client.has_collection(MILVUS_COLLECTION)

    if exists and drop_existing:
        log.info(f"  Dropping existing collection: {MILVUS_COLLECTION}")
        client.drop_collection(MILVUS_COLLECTION)
        exists = False

    if not exists:
        log.info(f"  Creating collection: {MILVUS_COLLECTION}")
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("case_id",   DataType.INT64,        is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
        client.create_collection(collection_name=MILVUS_COLLECTION, schema=schema)
        log.info(f"  Collection created.")
    else:
        log.info(f"  Collection already exists: {MILVUS_COLLECTION}")

    return client


def _get_existing_ids(client) -> set:
    """Return set of case_ids already in the collection."""
    try:
        result = client.query(
            collection_name=MILVUS_COLLECTION,
            filter="case_id >= 0",
            output_fields=["case_id"],
            limit=100_000,
        )
        return {r["case_id"] for r in result}
    except Exception:
        return set()


def _insert_batches(client, df: pd.DataFrame) -> int:
    """Insert records in batches of INSERT_BATCH. Returns number inserted."""
    total_inserted = 0

    for start in range(0, len(df), INSERT_BATCH):
        batch = df.iloc[start : start + INSERT_BATCH]

        data = [
            {
                "case_id":   int(row["case_id"]),
                "embedding": list(row["embedding"]),
            }
            for _, row in batch.iterrows()
        ]

        client.insert(collection_name=MILVUS_COLLECTION, data=data)
        total_inserted += len(data)
        log.info(f"  Inserted {total_inserted}/{len(df)} vectors …")

    return total_inserted


def _build_index(client) -> None:
    """Build HNSW index after all vectors are inserted."""
    log.info("Building HNSW index …")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name   = "embedding",
        index_type   = "HNSW",
        metric_type  = "COSINE",
        params       = {
            "M":              HNSW_M,
            "efConstruction": HNSW_EF_CONSTRUCTION,
        },
    )
    client.create_index(
        collection_name = MILVUS_COLLECTION,
        index_params    = index_params,
    )
    log.info(f"  HNSW index built (M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})")


def _verify(client) -> None:
    """Spot-check: load collection and run a quick ANN search."""
    client.load_collection(MILVUS_COLLECTION)
    stats = client.get_collection_stats(MILVUS_COLLECTION)
    row_count = stats.get("row_count", "?")
    log.info(f"  Collection loaded — row_count={row_count}")

    # Quick self-query: fetch first case_id and search for it
    sample = client.query(
        collection_name = MILVUS_COLLECTION,
        filter          = "case_id >= 0",
        output_fields   = ["case_id", "embedding"],
        limit           = 1,
    )
    if sample:
        query_vec = sample[0]["embedding"]
        results   = client.search(
            collection_name = MILVUS_COLLECTION,
            data            = [query_vec],
            anns_field      = "embedding",
            search_params   = {"metric_type": "COSINE", "params": {"ef": HNSW_EF}},
            limit           = 3,
            output_fields   = ["case_id"],
        )
        top_ids = [r["entity"]["case_id"] for r in results[0]]
        log.info(f"  Smoke-test ANN search — top-3 case_ids: {top_ids}")
        assert sample[0]["case_id"] == top_ids[0], \
            "Self-search failed: query case_id should be top-1 hit"
        log.info("  Smoke-test passed ✅")


def main(drop_existing: bool = False) -> None:
    try:
        from pymilvus import MilvusClient
    except ImportError:
        log.error("pymilvus not installed. Run: pip install pymilvus --break-system-packages")
        raise

    log.info(f"Loading embeddings from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    log.info(f"  {len(df):,} embeddings loaded")

    # Validate embeddings shape
    sample_vec = df["embedding"].iloc[0]
    if isinstance(sample_vec, list):
        actual_dim = len(sample_vec)
    else:
        actual_dim = np.array(sample_vec).shape[0]

    if actual_dim != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dim mismatch: expected {EMBEDDING_DIM}, got {actual_dim}. "
            "Re-run embed_cases.py."
        )
    log.info(f"  Embedding dim check passed: {actual_dim}")

    # Validate L2 norms (should all be ~1.0 after normalization)
    sample_norms = [
        np.linalg.norm(df["embedding"].iloc[i])
        for i in range(min(20, len(df)))
    ]
    avg_norm = np.mean(sample_norms)
    if not (0.98 < avg_norm < 1.02):
        log.warning(
            f"  Average L2 norm = {avg_norm:.4f} (expected ~1.0). "
            "Vectors may not be normalized — check embed_cases.py."
        )
    else:
        log.info(f"  L2 norm check passed: avg norm = {avg_norm:.4f}")

    # Connect
    log.info(f"Connecting to Milvus Lite: {MILVUS_DB_PATH}")
    client = MilvusClient(uri=str(MILVUS_DB_PATH))

    _get_collection(client, drop_existing)

    # Skip already-inserted cases if not dropping
    if not drop_existing:
        existing_ids = _get_existing_ids(client)
        if existing_ids:
            log.info(f"  {len(existing_ids):,} case_ids already in collection, skipping")
            df = df[~df["case_id"].isin(existing_ids)].reset_index(drop=True)

    if len(df) == 0:
        log.info("  No new vectors to insert.")
    else:
        log.info(f"  Inserting {len(df):,} vectors in batches of {INSERT_BATCH} …")
        n_inserted = _insert_batches(client, df)
        log.info(f"  Insert complete: {n_inserted:,} vectors")

    _build_index(client)
    _verify(client)
    log.info(f"Done. Milvus DB → {MILVUS_DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop the collection and rebuild from scratch",
    )
    args = parser.parse_args()
    main(drop_existing=args.drop_existing)