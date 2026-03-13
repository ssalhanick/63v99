"""
embeddings/embed_cases.py

Generates 768-dim L2-normalized embeddings for all pruned corpus cases using
legal-bert-base-uncased. Saves incrementally to parquet (crash-safe).

Pipeline per case:
  1. Structure-aware paragraph chunking (512-token ceiling, 1-paragraph overlap)
  2. legal-bert inference in batches of EMBED_BATCH_SIZE
  3. Mean-pool chunk [CLS] token embeddings → 1 vector per case
  4. L2-normalize the pooled vector
  5. Append to running parquet file

Input:  data/processed/cases_pruned.parquet
Output: data/processed/embeddings.parquet  (case_id | embedding)

Run:
  python -m embeddings.embed_cases [--resume]

Flags:
  --resume   Skip cases whose case_id already exists in embeddings.parquet
"""

import argparse
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    PROCESSED_DIR, EMBEDDING_MODEL, EMBEDDING_DIM,
    MAX_TEXT_LENGTH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH  = Path(PROCESSED_DIR) / "cases_pruned.parquet"
OUTPUT_PATH = Path(PROCESSED_DIR) / "embeddings.parquet"

# Tunable constants
MAX_TOKENS       = 512     # legal-bert max sequence length
CHUNK_OVERLAP    = 1       # paragraphs of overlap between consecutive chunks
EMBED_BATCH_SIZE = 16      # cases per inference batch (reduce to 8 if OOM)
SAVE_EVERY       = 50      # write to parquet every N cases (crash recovery)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines; return non-empty paragraphs."""
    paras = [p.strip() for p in text.split("\n\n")]
    return [p for p in paras if p]


def _chunk_paragraphs(
    paragraphs: list[str],
    tokenizer,
    max_tokens: int = MAX_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Greedily pack consecutive paragraphs into chunks that fit within max_tokens.
    Uses overlap paragraphs at chunk boundaries.
    Returns list of chunk strings.
    """
    if not paragraphs:
        return []

    chunks = []
    i = 0
    while i < len(paragraphs):
        current_paras = []
        current_len   = 0
        j = i
        while j < len(paragraphs):
            para_tokens = len(tokenizer.encode(paragraphs[j], add_special_tokens=False))
            # If a single paragraph exceeds max_tokens, truncate it
            if para_tokens > max_tokens - 2:  # -2 for [CLS]/[SEP]
                truncated = tokenizer.decode(
                    tokenizer.encode(paragraphs[j], add_special_tokens=False)[: max_tokens - 2]
                )
                current_paras.append(truncated)
                j += 1
                break
            if current_len + para_tokens + 2 > max_tokens:
                break
            current_paras.append(paragraphs[j])
            current_len += para_tokens
            j += 1

        chunks.append("\n\n".join(current_paras))
        # Advance with overlap: step back `overlap` paragraphs
        step = max(1, (j - i) - overlap)
        i += step

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _mean_pool_cls(model_output, attention_mask) -> np.ndarray:
    """
    Mean-pool the last hidden state over non-padding tokens.
    More robust than CLS-only for longer documents.
    """
    token_embeddings = model_output.last_hidden_state  # (batch, seq, dim)
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed  = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts  = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return (summed / counts).detach().cpu().numpy()  # (batch, dim)


def embed_text_chunks(
    chunks: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    """
    Embed all chunks, return mean-pooled 768-dim vector for the case.
    """
    all_chunk_vectors = []

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)

        vecs = _mean_pool_cls(output, encoded["attention_mask"])  # (batch, 768)
        all_chunk_vectors.append(vecs)

    stacked = np.vstack(all_chunk_vectors)      # (n_chunks, 768)
    case_vector = stacked.mean(axis=0)           # mean-pool across chunks → (768,)
    return case_vector


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Divide vector by its L2 norm. Required for cosine sim == dot product in Milvus."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(resume: bool = False) -> None:
    log.info(f"Loading pruned corpus from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH, columns=["case_id", "plain_text"])
    log.info(f"  {len(df):,} cases to embed")

    # Resume: skip already-embedded cases
    already_done: set = set()
    if resume and OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH, columns=["case_id"])
        already_done = set(existing["case_id"].tolist())
        log.info(f"  Resuming: {len(already_done):,} cases already embedded, skipping")
        df = df[~df["case_id"].isin(already_done)].reset_index(drop=True)
        log.info(f"  Remaining: {len(df):,} cases")

    if len(df) == 0:
        log.info("Nothing to embed. Exiting.")
        return

    # Load model
    log.info(f"Loading tokenizer + model: {EMBEDDING_MODEL}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model     = AutoModel.from_pretrained(EMBEDDING_MODEL).to(device)
    model.eval()
    log.info("  Model loaded.")

    records    = []  # {case_id, embedding}
    total      = len(df)
    start_time = time.time()

    for idx, row in df.iterrows():
        case_id   = row["case_id"]
        text      = row["plain_text"] or ""

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            log.warning(f"  case_id={case_id}: no paragraphs after split, skipping")
            continue

        chunks = _chunk_paragraphs(paragraphs, tokenizer)
        if not chunks:
            log.warning(f"  case_id={case_id}: no chunks produced, skipping")
            continue

        vec = embed_text_chunks(chunks, tokenizer, model, device)
        vec = l2_normalize(vec)

        records.append({"case_id": int(case_id), "embedding": vec.tolist()})

        i = len(records)
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate    = i / elapsed
            eta     = (total - i) / rate if rate > 0 else 0
            log.info(
                f"  [{i}/{total}] case_id={case_id} | "
                f"chunks={len(chunks)} | "
                f"{rate:.1f} cases/s | ETA {eta/60:.1f}m"
            )

        # Incremental save
        if i % SAVE_EVERY == 0:
            _save(records, already_done)

    # Final save
    _save(records, already_done)
    log.info(f"Done. {len(records):,} new embeddings saved → {OUTPUT_PATH}")


def _save(records: list[dict], already_done: set) -> None:
    """Merge new records with any existing embeddings.parquet and write."""
    if not records:
        return

    new_df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists() and already_done:
        existing_df = pd.read_parquet(OUTPUT_PATH)
        combined    = pd.concat([existing_df, new_df], ignore_index=True)
        # De-duplicate in case of retry
        combined    = combined.drop_duplicates(subset=["case_id"], keep="last")
    else:
        combined = new_df

    combined.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"  Checkpoint saved: {len(combined):,} total embeddings → {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip case_ids already present in embeddings.parquet",
    )
    args = parser.parse_args()
    main(resume=args.resume)