"""
preprocessing/tokenize_bm25.py

Tokenizes cleaned opinion text for BM25 sparse indexing.

Pipeline:
  1. Lowercase all text
  2. Remove spaCy stopwords — but preserve legal terms critical to
     Fourth Amendment context (see LEGAL_PRESERVE below)
  3. Lemmatize — "searched" and "searching" → "search", improving BM25 recall
  4. Strip punctuation and non-alpha tokens (except preserved legal terms)

Input:  data/processed/cases_cleaned.parquet  (case_id | cleaned_text)
Output: data/processed/cases_tokenized.parquet (case_id | tokens)
        tokens column is a list of strings per case

Run:
  python -m preprocessing.tokenize_bm25

Notes:
  - Uses spaCy en_core_web_sm for tokenization + lemmatization
  - cases_cleaned.parquet must exist (run preprocessing/clean_text.py first)
  - cases_tokenized.parquet is consumed by embeddings/bm25_index.py
  - Only non-pruned cases (those in embeddings.parquet) are tokenized,
    so BM25 corpus aligns exactly with Milvus vector index
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, LEGAL_PRESERVE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

INPUT_CLEANED    = Path(PROCESSED_DIR) / "cases_cleaned.parquet"
INPUT_EMBEDDINGS = Path(PROCESSED_DIR) / "embeddings.parquet"
OUTPUT_PATH      = Path(PROCESSED_DIR) / "cases_tokenized.parquet"

# LEGAL_PRESERVE imported from config.py — single source of truth shared with semantic_check.py

# Minimum token count — cases with fewer tokens after processing are skipped.
# Very short tokenized texts produce unreliable BM25 scores.
MIN_TOKENS = 20


def load_spacy_model() -> spacy.Language:
    """Load en_core_web_sm. Disable unused pipeline components for speed."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        log.info("spaCy model loaded: en_core_web_sm")
        return nlp
    except OSError:
        log.error(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )
        raise


def tokenize(text: str, nlp: spacy.Language) -> list[str]:
    """
    Tokenize a single document.

    Steps:
      - Lowercase + lemmatize via spaCy
      - Keep tokens that are alphabetic OR in LEGAL_PRESERVE
      - Remove stopwords unless the lemma is in LEGAL_PRESERVE
      - Drop tokens shorter than 2 characters

    Returns a list of lemmatized string tokens.
    """
    doc = nlp(text.lower()[:1_000_000])  # spaCy hard limit guard

    tokens = []
    for token in doc:
        lemma = token.lemma_.lower().strip()

        # Always keep legal terms regardless of stopword status
        if lemma in LEGAL_PRESERVE:
            tokens.append(lemma)
            continue

        # Drop stopwords, punctuation, whitespace, non-alpha
        if token.is_stop or token.is_punct or token.is_space:
            continue
        if not token.is_alpha:
            continue
        if len(lemma) < 2:
            continue

        tokens.append(lemma)

    return tokens


def main() -> None:
    # --- Load inputs ---
    if not INPUT_CLEANED.exists():
        log.error(f"Missing input: {INPUT_CLEANED}. Run preprocessing/clean_text.py first.")
        sys.exit(1)

    if not INPUT_EMBEDDINGS.exists():
        log.error(f"Missing input: {INPUT_EMBEDDINGS}. Run embeddings/embed_cases.py first.")
        sys.exit(1)

    log.info(f"Loading cleaned text from {INPUT_CLEANED}")
    df_cleaned = pd.read_parquet(INPUT_CLEANED)
    log.info(f"  {len(df_cleaned):,} cases in cleaned parquet")

    log.info(f"Loading embeddings index from {INPUT_EMBEDDINGS}")
    df_embeddings = pd.read_parquet(INPUT_EMBEDDINGS, columns=["case_id"])
    embedded_ids  = set(df_embeddings["case_id"].tolist())
    log.info(f"  {len(embedded_ids):,} case_ids in embeddings.parquet")

    # Align: only tokenize cases that were actually embedded
    # BM25 corpus must match Milvus vector index exactly
    text_col = "cleaned_text" if "cleaned_text" in df_cleaned.columns else "plain_text"
    if text_col not in df_cleaned.columns:
        log.error(
            f"Expected column 'cleaned_text' or 'plain_text' in {INPUT_CLEANED}. "
            f"Found: {df_cleaned.columns.tolist()}"
        )
        sys.exit(1)

    df = df_cleaned[df_cleaned["case_id"].isin(embedded_ids)][["case_id", text_col]].copy()
    df = df.dropna(subset=[text_col])
    log.info(f"  {len(df):,} cases to tokenize (intersection of cleaned + embedded)")

    # --- Load spaCy ---
    nlp = load_spacy_model()

    # --- Tokenize ---
    log.info("Tokenizing corpus …")
    results = []
    skipped = 0

    for i, (_, row) in enumerate(df.iterrows(), 1):
        tokens = tokenize(str(row[text_col]), nlp)

        if len(tokens) < MIN_TOKENS:
            log.debug(f"  Skipping case_id={row['case_id']} — only {len(tokens)} tokens")
            skipped += 1
            continue

        results.append({"case_id": int(row["case_id"]), "tokens": tokens})

        if i % 100 == 0:
            log.info(f"  Tokenized {i}/{len(df)} cases …")

    log.info(f"Tokenization complete: {len(results):,} cases kept, {skipped} skipped (<{MIN_TOKENS} tokens)")

    # --- Save ---
    df_out = pd.DataFrame(results)  # columns: case_id (int64), tokens (list of str)
    df_out.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved → {OUTPUT_PATH}")

    # --- Spot check ---
    sample = df_out.iloc[0]
    log.info(f"  Sample case_id={sample['case_id']} — {len(sample['tokens'])} tokens")
    log.info(f"  First 20 tokens: {sample['tokens'][:20]}")
    log.info("Done ✅")


if __name__ == "__main__":
    main()