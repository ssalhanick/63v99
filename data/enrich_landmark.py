"""
data/enrich_landmarks.py

Fetches landmark case opinions from CourtListener, strips HTML,
embeds using Legal-BERT, and inserts into:
  - Milvus (case_embeddings collection)
  - cases_enriched.parquet (for _CORPUS_INDEX citation resolution)
  - Neo4j (landmark node upsert)

Citations are pulled automatically from the CourtListener cluster endpoint.
No hardcoding required.

Run once:
    py -m data.enrich_landmarks
"""

import re
import time
import logging
import pickle
import numpy as np
import pandas as pd
import requests

from pathlib import Path
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModel
import torch

from config import (
    COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    MILVUS_URI, MILVUS_COLLECTION,
    EMBEDDING_MODEL, PROCESSED_DIR,
    LANDMARK_IDS
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

HEADERS      = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}
PARQUET_PATH = Path(PROCESSED_DIR) / "cases_enriched.parquet"
BM25_PATH    = Path(PROCESSED_DIR) / "bm25_index.pkl"

# Citation types to keep — 1=U.S./F.3d etc, 2=state, 3=specialty
# Exclude 4=U.S.L.W., 6=LEXIS — not resolvable by EyeCite
KEEP_CITATION_TYPES = {1, 2, 3}


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

def strip_html(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)   # html entities
    text = re.sub(r"&#\d+;", " ", text)          # numeric entities
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Citation parsing
# ---------------------------------------------------------------------------

def parse_citations(citations_list: list) -> list[str]:
    """
    Convert CourtListener citation objects to plain reporter strings.
    Keeps types 1 (federal/SCOTUS), 2 (state), 3 (specialty).
    Excludes LEXIS (6), U.S.L.W. (4), and other non-EyeCite reporters.

    Input:  [{'volume': '392', 'reporter': 'U.S.', 'page': '1', 'type': 1}, ...]
    Output: ['392 U.S. 1', '88 S. Ct. 1868', ...]
    """
    result = []
    for c in citations_list:
        if not isinstance(c, dict):
            continue
        if c.get("type") not in KEEP_CITATION_TYPES:
            continue
        volume   = c.get("volume", "").strip()
        reporter = c.get("reporter", "").strip()
        page     = c.get("page", "").strip()
        if volume and reporter and page:
            result.append(f"{volume} {reporter} {page}")
    return result


# ---------------------------------------------------------------------------
# CourtListener fetch
# ---------------------------------------------------------------------------

def fetch_landmark(opinion_id: int) -> dict | None:
    """
    Fetch opinion text and cluster metadata for a landmark case.
    Returns a dict with case_id, case_name, year, court, citations, plain_text.
    Returns None if fetch fails or no text is available.
    """
    # Fetch opinion
    url  = f"{COURTLISTENER_BASE_URL}/opinions/{opinion_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        log.warning("Failed to fetch opinion %d — status %d", opinion_id, resp.status_code)
        return None

    data = resp.json()

    # Extract text — prefer plain_text, fall back to HTML fields
    text = data.get("plain_text", "").strip()
    if not text:
        for field in ["html_with_citations", "html", "html_lawbox", "html_columbia"]:
            raw = data.get(field, "").strip()
            if raw and len(raw) > 100:
                text = strip_html(raw)
                log.info("  Using %s field for opinion text", field)
                break

    if not text:
        log.warning("No text found for opinion %d", opinion_id)
        return None

    # Fetch cluster for metadata + citations
    cluster_url = data.get("cluster", "")
    name      = f"Unknown (opinion {opinion_id})"
    year      = None
    court     = "scotus"
    citations = []

    if cluster_url:
        cr = requests.get(cluster_url, headers=HEADERS, timeout=15)
        if cr.status_code == 200:
            cd         = cr.json()
            name       = cd.get("case_name", name)
            date_filed = cd.get("date_filed", "")
            year       = int(date_filed[:4]) if date_filed else None
            citations  = parse_citations(cd.get("citations", []))
            docket     = cd.get("docket", {})
            court      = docket.get("court_id", "scotus") if isinstance(docket, dict) else "scotus"
        else:
            log.warning("Failed to fetch cluster for opinion %d — status %d",
                        opinion_id, cr.status_code)

    return {
        "case_id":    opinion_id,
        "case_name":  name,
        "year":       year,
        "court":      court,
        "citations":  citations,
        "plain_text": text,
    }


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_embedder():
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model     = AutoModel.from_pretrained(EMBEDDING_MODEL)
    model.eval()
    log.info("Embedder loaded")
    return tokenizer, model


def embed(text: str, tokenizer, model) -> list[float]:
    """
    Embed text using Legal-BERT mean pooling.
    Truncates to 512 tokens. Returns L2-normalized 768-dim vector.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    mask   = inputs["attention_mask"].unsqueeze(-1).float()
    vector = ((outputs.last_hidden_state * mask).sum(1) /
               mask.sum(1).clamp(min=1e-9)).squeeze(0).numpy()
    norm   = np.linalg.norm(vector)
    vector = vector / norm if norm > 1e-9 else vector
    return vector.tolist()


# ---------------------------------------------------------------------------
# Neo4j upsert
# ---------------------------------------------------------------------------

def upsert_neo4j(driver, case: dict):
    with driver.session() as session:
        session.run("""
            MERGE (c:Case {id: $case_id})
            SET c.name     = $name,
                c.year     = $year,
                c.court    = $court,
                c.court_id = $court,
                c.stub     = false,
                c.landmark = true
        """,
        case_id = case["case_id"],
        name    = case["case_name"],
        year    = case["year"],
        court   = case["court"],
        )
    log.info("Neo4j upserted: %s (id=%d)", case["case_name"], case["case_id"])


# ---------------------------------------------------------------------------
# Milvus insert
# ---------------------------------------------------------------------------

def insert_milvus(client, case_id: int, vector: list[float]):
    # Delete existing entry if present to avoid duplicates
    client.delete(
        collection_name=MILVUS_COLLECTION,
        filter=f"case_id == {case_id}",
    )
    client.insert(
        collection_name=MILVUS_COLLECTION,
        data=[{"case_id": case_id, "embedding": vector}],
    )
    log.info("Milvus inserted: case_id=%d", case_id)


# ---------------------------------------------------------------------------
# Parquet update
# ---------------------------------------------------------------------------

def update_parquet(cases: list[dict]):
    df = pd.read_parquet(PARQUET_PATH)

    for case in cases:
        case_id       = case["case_id"]
        citations_str = str(case["citations"]) if case["citations"] else "[]"

        if case_id in df["case_id"].values:
            df.loc[df["case_id"] == case_id, "citations"]  = citations_str
            df.loc[df["case_id"] == case_id, "plain_text"] = case["plain_text"]
            df.loc[df["case_id"] == case_id, "case_name"]  = case["case_name"]
            log.info("Parquet updated existing row: %s (id=%d)",
                     case["case_name"], case_id)
        else:
            new_row = {col: None for col in df.columns}
            new_row.update({
                "case_id":    case_id,
                "case_name":  case["case_name"],
                "citations":  citations_str,
                "plain_text": case["plain_text"],
                "court_id":   case["court"],
                "date_filed": f"{case['year']}-01-01" if case["year"] else None,
            })
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            log.info("Parquet added new row: %s (id=%d)",
                     case["case_name"], case_id)

    df.to_parquet(PARQUET_PATH, index=False)
    log.info("Parquet saved — %d total rows", len(df))


# ---------------------------------------------------------------------------
# BM25 rebuild
# ---------------------------------------------------------------------------

def rebuild_bm25(cases: list[dict]):
    """Append landmark texts to the BM25 index."""
    log.info("Rebuilding BM25 index with landmark texts...")

    with open(BM25_PATH, "rb") as f:
        payload = pickle.load(f)

    existing_ids   = payload["case_ids"]
    existing_texts = payload.get("corpus_tokens")

    if existing_texts is None:
        log.warning(
            "BM25 index has no corpus_tokens key — skipping BM25 rebuild. "
            "Landmarks will not be BM25-searchable until a full reindex is run."
        )
        return

    from rank_bm25 import BM25Okapi

    new_tokens = []
    new_ids    = []
    for case in cases:
        tokens = [t for t in case["plain_text"].lower().split() if len(t) > 1]
        new_tokens.append(tokens)
        new_ids.append(case["case_id"])
        log.info("  BM25 tokens for %s: %d", case["case_name"], len(tokens))

    all_tokens = existing_texts + new_tokens
    all_ids    = existing_ids   + new_ids
    new_bm25   = BM25Okapi(all_tokens)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25":          new_bm25,
            "case_ids":      all_ids,
            "corpus_tokens": all_tokens,
        }, f)

    log.info("BM25 rebuilt — %d total documents", len(all_ids))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Verit — Landmark Enrichment")
    log.info("Processing %d landmark cases...\n", len(LANDMARK_IDS))

    tokenizer, model = load_embedder()
    milvus_client    = MilvusClient(uri=MILVUS_URI)
    milvus_client.load_collection(MILVUS_COLLECTION)
    neo4j_driver     = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    )

    enriched = []
    skipped  = []

    for opinion_id in LANDMARK_IDS:
        log.info("--- Processing opinion_id=%d ---", opinion_id)

        case = fetch_landmark(opinion_id)
        if not case:
            log.warning("Skipping %d — fetch failed or no text", opinion_id)
            skipped.append(opinion_id)
            continue

        log.info("  Case:      %s (%s)", case["case_name"], case["year"])
        log.info("  Court:     %s", case["court"])
        log.info("  Text:      %d chars", len(case["plain_text"]))
        log.info("  Citations: %s", case["citations"])

        # Embed
        vector = embed(case["plain_text"], tokenizer, model)
        log.info("  Embedded:  dim=%d", len(vector))

        # Neo4j
        upsert_neo4j(neo4j_driver, case)

        # Milvus
        insert_milvus(milvus_client, case["case_id"], vector)

        enriched.append(case)
        time.sleep(0.5)

    # Parquet + BM25
    if enriched:
        update_parquet(enriched)
        rebuild_bm25(enriched)

    neo4j_driver.close()

    log.info("\n--- Summary ---")
    log.info("Enriched: %d / %d", len(enriched), len(LANDMARK_IDS))
    if skipped:
        log.warning("Skipped:  %s", skipped)
    log.info("Next steps:")
    log.info("  1. Restart FastAPI and Streamlit to rebuild _CORPUS_INDEX")
    log.info("  2. Verify: py -m tests.landmark_verify")


if __name__ == "__main__":
    main()