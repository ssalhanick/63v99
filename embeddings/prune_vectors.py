"""
embeddings/prune_vectors.py

Filters the cleaned corpus to remove cases unsuitable for embedding:
  - null or empty plain_text
  - plain_text shorter than MIN_TEXT_LENGTH (too short to embed meaningfully)

Cases that pass pruning are written to cases_pruned.parquet.
Cases that fail are logged but retained in Neo4j (no graph changes).

Input:  data/processed/cases_cleaned.parquet
Output: data/processed/cases_pruned.parquet

Run:
  python -m embeddings.prune_vectors
"""

import logging
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, MIN_TEXT_LENGTH, MAX_TEXT_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH  = Path(PROCESSED_DIR) / "cases_cleaned.parquet"
OUTPUT_PATH = Path(PROCESSED_DIR) / "cases_pruned.parquet"


def prune(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (kept, dropped).
    Truncates text at MAX_TEXT_LENGTH in the kept frame.
    """
    original_count = len(df)

    # 1. Drop nulls
    mask_null = df["plain_text"].isna()
    log.info(f"  Null plain_text:  {mask_null.sum():,}")

    # 2. Drop too-short
    mask_short = df["plain_text"].fillna("").str.len() < MIN_TEXT_LENGTH
    log.info(f"  Too short (<{MIN_TEXT_LENGTH} chars): {(mask_short & ~mask_null).sum():,}")

    drop_mask = mask_null | mask_short
    kept = df[~drop_mask].copy()
    dropped = df[drop_mask].copy()

    # 3. Truncate long texts (in-place on kept)
    long_mask = kept["plain_text"].str.len() > MAX_TEXT_LENGTH
    log.info(f"  Truncated (>{MAX_TEXT_LENGTH} chars): {long_mask.sum():,}")
    kept.loc[long_mask, "plain_text"] = kept.loc[long_mask, "plain_text"].str[:MAX_TEXT_LENGTH]

    log.info(f"  Original: {original_count:,} → Kept: {len(kept):,} | Dropped: {len(dropped):,}")
    return kept, dropped


def main() -> None:
    log.info(f"Loading {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    log.info(f"Loaded {len(df):,} cases")

    kept, dropped = prune(df)

    if len(dropped) > 0:
        log.info("Dropped case_ids (first 20):")
        for cid in dropped["case_id"].head(20).tolist():
            log.info(f"  {cid}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved pruned corpus → {OUTPUT_PATH} ({len(kept):,} cases)")


if __name__ == "__main__":
    main()