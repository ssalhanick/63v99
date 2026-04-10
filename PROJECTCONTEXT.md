# Verit — Project Context

> Paste this at the start of each new Claude session to restore full project context.

---

## Project Overview

**Name:** Verit
**Type:** Legal Citation Hallucination Detector
**Domain:** Fourth Amendment / Search & Seizure federal case law
**Due Date:** May 8, 2026
**Solo project**

### What It Does

Verifies whether citations in AI-generated legal text are real, relevant, and properly
connected using a three-layer detection pipeline:

1. **Existence check** — does the cited case exist as a node in Neo4j?
2. **Semantic relevance check** — is it relevant to the citing text? (Milvus ANN search + cosine similarity)
3. **Graph connectivity check** — does it have citation density overlap with the corpus? (Neo4j)

Results are surfaced through a Streamlit frontend powered by Claude Haiku, which explains
verdicts in plain English and suggests corrections for hallucinated citations using
retrieved corpus cases as RAG context.

---

## Tech Stack

| Component                | Tool                         | Purpose                                              |
| ------------------------ | ---------------------------- | ---------------------------------------------------- |
| Vector Store             | Milvus Lite 2.4+             | Store and search 768-dim case embeddings (HNSW)      |
| Sparse Index             | BM25 (rank_bm25)             | Keyword search over plain_text for hybrid search     |
| Hybrid Fusion            | Reciprocal Rank Fusion       | Merge dense + sparse results into single ranked list |
| Graph Database           | Neo4j 5.15 (Docker)          | Store citation relationships as directed graph       |
| Embedding Model          | legal-bert-base-uncased      | Convert legal text to semantic vectors               |
| Dimensionality Reduction | UMAP                         | 2D visualization of embedding space (Week 9)         |
| Citation Extraction      | EyeCite                      | Parse citation strings from raw text                 |
| API Layer                | FastAPI                      | Expose /check-citation endpoint                      |
| Query Cache              | cachetools TTLCache          | Cache embeddings + ANN results at API layer          |
| Frontend                 | Streamlit                    | User-facing UI for pasting and checking legal text   |
| LLM                      | Claude Haiku (Anthropic API) | Explain verdicts, suggest corrections via RAG        |
| Infrastructure           | Docker + Docker Compose      | Run Neo4j locally                                    |
| Language                 | Python 3.10                  |                                                      |
| IDE                      | VS Code                      |                                                      |

---

## Environment

- **OS:** Windows
- **Python:** 3.10.11
- **Virtual env:** `.venv` (activate with `.venv\Scripts\activate`)
- **Project root:** `C:\Users\ssalh\Grad School\2026\01_Spring\MIS6V99\Verit`
- **Run scripts with:** `python -m folder.script` from project root

---

## Data

| File                         | Location          | Description                                                   |
| ---------------------------- | ----------------- | ------------------------------------------------------------- |
| `batch_2015_present.json`    | `data/raw/`       | 1,500 raw cases from CourtListener (multi-circuit, post-2015) |
| `batch_2010_2015.json`       | `data/raw/`       | 500 raw cases 2010-2015                                       |
| `enriched_2015_present.json` | `data/raw/`       | 1,234 cases with full opinion text                            |
| `enriched_2010_2015.json`    | `data/raw/`       | 183 cases with full opinion text                              |
| `cases_merged.json`          | `data/raw/`       | 1,353 unique deduplicated enriched cases                      |
| `cases_enriched.parquet`     | `data/processed/` | Final dataset — 1,353 cases, 20.23 MB                         |

### Corpus Stats

- **Total unique cases:** 1,353
- **Date range:** 2010 – 2025
- **Courts:** Multi-circuit federal (anchored in 9th Circuit)
- **Domain:** Fourth Amendment, search and seizure, warrant, probable cause
- **Fields:** case_id, cluster_id, case_name, date_filed, court_id, citations,
  plain_text, opinions_cited, cite_count, docket_number, status

---

## Neo4j

- **Container name:** `verit_neo4j`
- **Ports:** 7474 (browser), 7687 (bolt)
- **Credentials:** neo4j / Verit2026! (stored in .env)
- **URI:** bolt://localhost:7687
- **Access:** cypher-shell only (browser auth issues with local Docker on Windows)
- **Start:** `docker-compose up -d` then wait 30 seconds before running any scripts
- **Status:** ✅ Graph loaded and verified

### cypher-shell Access

```powershell
docker exec -it verit_neo4j cypher-shell -u neo4j -p "Verit2026!"
```

### Graph Stats (Week 3 final)

| Metric           | Count  |
| ---------------- | ------ |
| Full Case nodes  | 1,358  |
| Stub nodes       | 14,773 |
| Total Case nodes | 16,131 |
| CITES edges      | 30,806 |
| Landmark nodes   | 5      |

### Landmark Anchor Cases

Loaded via `db/fetch_landmarks.py`. Present in graph as full nodes with `landmark: true`.
These are isolated from the corpus citation network (corpus cases do not cite them by
CourtListener opinion ID) — landmark connectivity is not used in Layer 3 (see design decisions).

| Case                  | Year | CourtListener Opinion ID |
| --------------------- | ---- | ------------------------ |
| Terry v. Ohio         | 1968 | 107729                   |
| Katz v. United States | 1967 | 107564                   |
| Mapp v. Ohio          | 1961 | 106285                   |
| United States v. Leon | 1984 | 111252                   |
| Illinois v. Gates     | 1983 | 110930                   |

---

## Project Structure

```
Verit/
├── .env                          # Credentials (never pushed to GitHub)
├── .env.example                  # Credential template
├── .gitignore
├── config.py                     # Central config — all paths, credentials, constants
├── docker-compose.yml            # Neo4j container
├── requirements.txt
├── README.md
├── PROJECT_CONTEXT.md            # This file
├── data/
│   ├── __init__.py
│   ├── fetch_cases.py            # CourtListener API fetch (argparse: --after, --before, --limit)
│   ├── fetch_all_opinions.py     # Enrich cases with full text (argparse: --batch)
│   ├── merge_batches.py          # Merge enriched batch files, deduplicate on case_id
│   ├── convert_to_parquet.py     # Convert cases_merged.json → cases_enriched.parquet
│   ├── diagnose_batches.py       # Check snippet/PDF coverage across batch files
│   ├── diagnose_text.py          # Check plain_text coverage in enriched files
│   ├── data_check.py             # General data inspection utility
│   ├── raw/                      # Raw and enriched JSON files (in .gitignore)
│   └── processed/                # Parquet files (in .gitignore)
├── db/
│   ├── __init__.py
│   ├── neo4j_client.py           # Neo4j driver, create_case, create_citation, create_landmark
│   ├── graph_loader.py           # ✅ Loads parquet into Neo4j graph (batched, idempotent)
│   ├── fetch_landmarks.py        # ✅ Fetches landmark cases from CourtListener → Neo4j
│   └── verify_landmarks.py       # ✅ Verifies CourtListener opinion IDs before graph load
├── embeddings/
│   ├── __init__.py
│   ├── prune_vectors.py          # Week 4 — filter low-quality embeddings before indexing
│   ├── embed_cases.py            # Week 4 — chunk, embed, mean-pool, L2-normalize, save to parquet
│   ├── milvus_index.py           # Week 4 — bulk insert normalized vectors, build HNSW index
│   └── bm25_index.py             # Week 5 — build BM25 sparse index over tokenized plain_text
├── preprocessing/
│   ├── __init__.py
│   ├── clean_text.py             # Week 4 — strip headers/footers, normalize citations to [CITATION], fix encoding
│   └── tokenize_bm25.py          # Week 5 — lowercase, remove stopwords, lemmatize for BM25 corpus
├── detector/
│   ├── __init__.py
│   ├── eyecite_parser.py         # Week 6 — extract citation strings from raw text via EyeCite
│   ├── existence_check.py        # Week 6 — Layer 1: Neo4j node lookup
│   ├── semantic_check.py         # Week 6 — Layer 2: hybrid search (ANN + BM25 via RRF)
│   ├── connectivity_check.py     # Week 6 — Layer 3: citation density scoring
│   ├── pipeline.py               # Week 6 — orchestrate all three layers, return verdict
│   └── cache.py                  # Week 6 — TTLCache for query embeddings + ANN results
├── api/
│   ├── __init__.py
│   └── main.py                   # Week 6 — FastAPI /check-citation endpoint
├── benchmark/
│   ├── __init__.py
│   ├── generate_benchmark.py     # Week 7 — build balanced real/hallucinated test set
│   └── benchmark.json            # Week 7 — 50/50 real vs hallucinated citations (in .gitignore)
├── frontend/
│   ├── __init__.py
│   ├── app.py                    # Week 7/9 — Streamlit UI: input, verdict display, LLM explanations
│   └── llm.py                    # Week 9 — Claude Haiku integration: explain verdicts, suggest corrections
├── visualization/
│   └── umap_viz.py               # Week 9 — StandardScaler + UMAP + hallucination overlay
└── tests/
    ├── __init__.py
    ├── conftest.py               # Fixtures: raw_cases, merged_cases
    ├── test_data.py              # 13 passing tests ✅
    ├── test_db.py                # 14 passing tests ✅ (Week 3)
    └── test_detector.py          # Empty — to be built in Week 6
```

## Preprocessing Pipeline — Full Order

```
raw plain_text (cases_enriched.parquet)
  │
  ├── [Week 4] preprocessing/clean_text.py
  │     Strip court headers/footers, normalize citations → [CITATION] token,
  │     fix encoding artifacts and excessive whitespace
  │     Output: data/processed/cases_cleaned.parquet
  │
  ├── [Week 4] embeddings/prune_vectors.py
  │     Drop cases with plain_text < MIN_TEXT_LENGTH or null
  │     Truncate at MAX_TEXT_LENGTH before chunking
  │     Output: filtered case_id list
  │
  ├── [Week 4] embeddings/embed_cases.py
  │     Paragraph chunking (512 token ceiling, 1-paragraph overlap)
  │     legal-bert inference in batches of 16-32
  │     Mean-pool chunk embeddings → 768-dim vector per case
  │     L2-normalize each vector (divide by L2 norm)
  │     Output: data/processed/embeddings.parquet
  │
  ├── [Week 4] embeddings/milvus_index.py
  │     Bulk insert all normalized vectors
  │     Then build HNSW index (M=16, ef_construction=200)
  │     Output: milvus_verit.db
  │
  ├── [Week 5] preprocessing/tokenize_bm25.py
  │     Takes cases_cleaned.parquet as input
  │     Lowercase, remove stopwords (preserve legal terms), lemmatize
  │     Output: data/processed/cases_tokenized.parquet
  │
  ├── [Week 5] embeddings/bm25_index.py
  │     Build BM25 sparse index from tokenized corpus
  │     Output: data/processed/bm25_index.pkl
  │
  └── [Week 9] visualization/umap_viz.py
        Load embeddings from embeddings.parquet
        StandardScaler → zero mean, unit variance per dimension
        UMAP reduction to 2D (n_neighbors=15, min_dist=0.1, metric='cosine')
        Overlay: color by circuit, year, or hallucination label
```

---

## config.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Root and common paths
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(ROOT_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
BENCHMARK_DIR = os.path.join(ROOT_DIR, "benchmark")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")

# Neo4j
NEO4J_URI      = os.getenv("NEO4J_URI")       # bolt://localhost:7687
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# CourtListener
COURTLISTENER_TOKEN    = os.getenv("COURTLISTENER_TOKEN")
COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL   = "claude-haiku-4-5-20251001"   # Claude Haiku — LLM layer

# Landmark Fourth Amendment anchor cases (CourtListener opinion IDs)
# Verified and loaded into Neo4j via db/fetch_landmarks.py
# NOTE: These are isolated from the corpus citation network — not used in Layer 3
LANDMARK_IDS = [
    107729,   # Terry v. Ohio (1968)
    107564,   # Katz v. United States (1967)
    106285,   # Mapp v. Ohio (1961)
    111252,   # United States v. Leon (1984)
    110930,   # Illinois v. Gates (1983)
]

# Embedding
EMBEDDING_MODEL   = "nlpaueb/legal-bert-base-uncased"
EMBEDDING_DIM     = 768
MILVUS_URI        = "http://localhost:19530"
# MILVUS_DB_PATH  = os.path.join(ROOT_DIR, "milvus_verit.db")
MILVUS_COLLECTION = "case_embeddings"

# Vector pruning thresholds (Week 4)
MIN_TEXT_LENGTH   = 200    # characters — skip cases with very short plain_text
MAX_TEXT_LENGTH   = 50000  # characters — truncate before embedding

# HNSW index parameters (Week 4 — tune in Week 8 if recall drops below 95%)
HNSW_M                = 16    # bidirectional links per node
HNSW_EF_CONSTRUCTION  = 200   # build-time search width
HNSW_EF               = 50    # query-time search width

# ANN search parameters (Week 5)
TOP_K                  = 5     # candidates returned per query
SIMILARITY_THRESHOLD   = 0.75  # cosine similarity floor — tune on validation set in Week 8

# Hybrid search — Reciprocal Rank Fusion (Week 5)
# RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)
RRF_K                  = 60    # smoothing constant — standard default, rarely needs tuning
BM25_INDEX_PATH        = os.path.join(PROCESSED_DIR, "bm25_index.pkl")

# Connectivity (Layer 3 — Option B)
CITATION_DENSITY_THRESHOLD = 3   # minimum shared citations — tune on validation set in Week 8

# Cache (Week 6)
CACHE_TTL             = 3600   # seconds — TTL for query embedding + ANN result cache
CACHE_MAX_SIZE        = 512    # max entries in LRU cache
```

---

## Timeline Status

| Week | Dates           | Milestone                                                      | Status      |
| ---- | --------------- | -------------------------------------------------------------- | ----------- |
| 1    | Feb 24 – Mar 2  | Environment setup, Docker, Neo4j, first cases                  | ✅ Complete |
| 2    | Mar 3 – Mar 9   | Full data ingestion, Parquet pipeline                          | ✅ Complete |
| 3    | Mar 10 – Mar 16 | Neo4j graph build and verification                             | ✅ Complete |
| 4    | Mar 17 – Mar 23 | BERT embedding pipeline + vector pruning + Milvus              | ✅ Complete |
| 5    | Mar 24 – Mar 30 | ANN search + semantic retrieval layer                          | ✅ Complete |
| 6    | Mar 31 – Apr 6  | Hallucination detector — all three checks                      | ✅ Complete |
| 7    | Apr 7 – Apr 13  | Benchmark dataset construction + Streamlit app scaffold        | ✅ Complete |
| 8    | Apr 14 – Apr 20 | Evaluation — precision, recall, F1 + threshold tuning          | ⬜ Upcoming |
| 9    | Apr 21 – Apr 27 | Error analysis + UMAP visualization + LLM integration (Haiku)  | ⬜ Upcoming |
| 10   | Apr 28 – May 8  | Frontend polish + citation graph visualization + final writeup | ⬜ Upcoming |

---

## Week 4 — Text Cleaning + BERT Embedding Pipeline + Milvus Indexing

### Goals

Clean raw opinion text, generate L2-normalized 768-dimensional embeddings for all
non-pruned corpus cases using `legal-bert-base-uncased`, and index the result in
Milvus Lite using an HNSW index for fast ANN search.

### Text Cleaning (`preprocessing/clean_text.py`)

Run this before embedding. Legal opinions from CourtListener are noisy:

- **Header/footer stripping** — remove court headers, docket numbers, attorney lists.
  These are procedural, not doctrinal, and pull embeddings toward procedural similarity.
- **Citation normalization** — replace citation strings like `Terry v. Ohio, 392 U.S. 1 (1968)`
  with a `[CITATION]` token. Raw citations burn token budget and add noise.
- **Encoding cleanup** — fix unicode artifacts, excessive newlines, OCR errors from
  older PDF conversions common in pre-2015 opinions.
- Output: `data/processed/cases_cleaned.parquet`

### Vector Pruning (`embeddings/prune_vectors.py`)

Before embedding, filter out cases that would produce unreliable vectors:

- **Too short** — `plain_text` under `MIN_TEXT_LENGTH` characters (200). Very short
  opinions are often procedural orders with no substantive Fourth Amendment content.
- **Too long** — truncate at `MAX_TEXT_LENGTH` (50,000 characters) before chunking.
- **Missing text** — cases where `plain_text` is null or empty are dropped entirely.

Pruned cases remain in Neo4j as nodes but are not indexed in Milvus.

### Embedding (`embeddings/embed_cases.py`)

- Input: `cases_cleaned.parquet` (cleaned text)
- Model: `nlpaueb/legal-bert-base-uncased` (768-dim, pretrained on legal corpora)
- Chunking: structure-aware paragraph chunking with 1-paragraph overlap, 512-token ceiling
- Strategy: mean-pool the token embeddings across chunks → one 768-dim vector per case
- **L2 normalization:** divide each vector by its L2 norm before saving
  (`vector = vector / np.linalg.norm(vector)`). Required for cosine similarity
  to be equivalent to dot product in Milvus. Applied after mean-pooling, before
  saving to parquet and before Milvus insertion.
- Output: `data/processed/embeddings.parquet` (case_id + normalized 768-dim vector)
- Save to parquet after each batch — crash recovery without restarting from scratch
- Run with `--resume` flag to skip already-embedded cases after a crash

### Milvus Indexing (`embeddings/milvus_index.py`)

- Index type: **HNSW** (Hierarchical Navigable Small World)
  - Better recall than IVF_FLAT at this corpus size (~1,300 vectors)
  - Fast at query time — no quantization artifacts
  - Parameters: `M=16, ef_construction=200, ef=50` (tune in Week 8)
- Metric: cosine similarity
- Collection schema: `case_id` (int64) + `embedding` (float_vector, dim=768)
- **Critical:** bulk insert all vectors first, then call `create_index` — Milvus builds
  a better HNSW graph when it sees the full dataset at once.
- Insert in batches of 500-1000 to avoid memory spikes.
- Run with `--drop-existing` flag to rebuild from scratch.

### Scripts to Build

| Script                        | Purpose                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `preprocessing/clean_text.py` | Strip headers, normalize citations, fix encoding          |
| `embeddings/prune_vectors.py` | Filter corpus by text quality, output clean list          |
| `embeddings/embed_cases.py`   | Chunk, embed in batches, mean-pool, L2-normalize, parquet |
| `embeddings/milvus_index.py`  | Bulk insert normalized vectors, then build HNSW index     |

---

## Week 5 — Hybrid Search + Semantic Retrieval Layer

### Goals

Implement hybrid search combining dense ANN search (Milvus HNSW) with sparse keyword
search (BM25) fused via Reciprocal Rank Fusion. Add metadata pre-filtering and query
embedding caching. This becomes Layer 2 of the detector.

### Why Hybrid Search

BM25 improves **Layer 2 retrieval quality** — it finds better corpus candidates to compare
against, not whether the cited case itself is real. A cited case name like _"Carpenter v.
United States"_ may not be in your corpus, but the term `Carpenter` likely appears in
corpus opinions that discuss it. BM25 catches that keyword signal that dense vector search
can miss when the embedding space compresses semantically similar cases together.

The distinction matters: BM25 scoring zero on a citation tells you nothing about whether
that case is hallucinated — a real-but-out-of-corpus case and a fabricated case both score
zero. The hallucination detection work is done by the three layers working together:

- **Layer 1** — catches citations that don't exist anywhere in CourtListener
- **Layer 2** — checks whether the context around the citation is semantically consistent
  with real Fourth Amendment cases (hybrid search improves candidate retrieval here)
- **Layer 3** — checks whether the cited case has any citation footprint in the network

Hybrid search makes Layer 2 a better retriever. It does not replace Layers 1 or 3.

### BM25 Tokenization (`preprocessing/tokenize_bm25.py`)

Run before building the BM25 index. Takes `cases_cleaned.parquet` as input:

- Lowercase all text
- Remove stopwords — but preserve legal terms that are meaningful in Fourth Amendment
  context (`unreasonable`, `reasonable`, `warrant`, `probable`, `cause`, `seizure`)
- Lemmatize — `searched` and `searching` map to the same token, improving BM25 recall
- Output: `data/processed/cases_tokenized.parquet`

### BM25 Sparse Index (`embeddings/bm25_index.py`)

- Library: `rank_bm25` (pure Python, no additional infrastructure)
- Corpus: tokenized `plain_text` for all non-pruned cases
- Serialized to `data/processed/bm25_index.pkl` for reuse at query time
- Query: the citation string + surrounding context, same as the dense query

### Reciprocal Rank Fusion (`detector/semantic_check.py`)

RRF merges the dense and sparse ranked lists without requiring weight tuning:

```
rrf_score(d) = 1 / (k + rank_dense(d)) + 1 / (k + rank_sparse(d))
k = 60  (standard smoothing constant)
```

Steps at query time:

1. Embed context text using legal-bert → query vector
2. Run HNSW ANN search in Milvus → top-k dense candidates
3. Run BM25 keyword search → top-k sparse candidates
4. Fuse both ranked lists using RRF → final ranked list
5. Return top result's RRF score — flag as suspicious if below threshold

### Metadata Pre-filtering

Apply metadata filters **before** the vector search, not after — post-filtering kills recall
by discarding candidates before ranking. Milvus supports pre-filtering natively via the
`expr` parameter:

```python
# Example: restrict search to cases from a specific year range
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 50}},
    expr="year >= 2010 and year <= 2025",   # pre-filter
    limit=TOP_K
)
```

Useful filters: `court_id`, `year`, `stub == false` (never return stubs as candidates).

### Query Embedding Cache (`detector/cache.py`)

- Library: `cachetools.TTLCache`
- **Embedding cache:** keyed on `hash(context_text)` — avoids re-running legal-bert
  for repeated queries. Critical during benchmark evaluation loop in Week 8.
- **ANN result cache:** keyed on `(embedding_hash, top_k, expr)` — returns cached
  Milvus results instantly for identical queries.
- TTL: `CACHE_TTL = 3600` seconds, max size: `CACHE_MAX_SIZE = 512` entries

### Scripts to Build

| Script                           | Purpose                                                     |
| -------------------------------- | ----------------------------------------------------------- |
| `preprocessing/tokenize_bm25.py` | Lowercase, remove stopwords, lemmatize for BM25 corpus      |
| `embeddings/bm25_index.py`       | Build and serialize BM25 index over tokenized plain_text    |
| `detector/semantic_check.py`     | Hybrid search: ANN + BM25 fused via RRF, with pre-filtering |
| `detector/cache.py`              | TTLCache for query embeddings and ANN results               |

---

## Week 6 — Hallucination Detector + FastAPI Endpoint

### Goals

Wire the three detection layers into a unified pipeline, with EyeCite as the entry
point for citation extraction from raw text. Expose the full pipeline via a FastAPI
`/check-citation` endpoint. Each layer returns a score and a verdict; the pipeline
combines them into a final `REAL | SUSPICIOUS | HALLUCINATED` label.

### EyeCite Citation Extraction (`detector/eyecite_parser.py`)

EyeCite is used at **query time** to extract structured citation objects from raw
AI-generated text. This is distinct from the graph-building phase (Week 3) which used
CourtListener's `opinions_cited` API field directly.

```
Input:  raw AI-generated paragraph text
Output: list of extracted citations, each with:
          - full citation string (e.g. "392 U.S. 1")
          - reporter, volume, page
          - CourtListener opinion ID (resolved via API lookup)

Steps:
  1. Run EyeCite over input text → list of FoundCitation objects
  2. For each citation, query CourtListener /search/ to resolve to opinion ID
  3. Pass (citation_string, opinion_id, context_text) to three-layer pipeline
```

### Layer 1 — Existence Check (`detector/existence_check.py`)

```
Input:  case_id (integer, resolved from citation string via EyeCite + CourtListener lookup)
Output: exists (bool)

Steps:
  1. Query Neo4j: MATCH (c:Case {id: $id}) RETURN c
  2. Return True if node found, False otherwise
  3. If False → immediate HALLUCINATED verdict, skip Layers 2 and 3
```

### Layer 2 — Semantic Check (`detector/semantic_check.py`)

```
Input:  citation string + surrounding context text
Output: rrf_score (float), top_k_cases (list)

Steps:
  1. Check embedding cache (TTLCache keyed on hash(context_text))
  2. If miss: embed context using legal-bert, store in cache
  3. Run HNSW ANN search in Milvus (pre-filter: stub=false, top-k=5)
  4. Run BM25 search over tokenized corpus
  5. Fuse results via RRF → ranked list
  6. Return top RRF score
  7. Flag as SUSPICIOUS if score < RRF_THRESHOLD
```

### Layer 3 — Connectivity Check (`detector/connectivity_check.py`)

```
Input:  case_id
Output: density_score (int), is_connected (bool)

Steps:
  1. Query Neo4j for shared citations between target and corpus:
     MATCH (target:Case {id: $id})-[:CITES]->(shared)
           <-[:CITES]-(corpus:Case {stub: false})
     RETURN count(DISTINCT shared) AS density
  2. Return density score
  3. Flag as SUSPICIOUS if density < CITATION_DENSITY_THRESHOLD
```

### Pipeline (`detector/pipeline.py`)

```
Input:  raw AI-generated text
Output: list of verdicts, one per extracted citation

Logic per citation:
  citations = eyecite_parser(raw_text)   # extract + resolve to opinion IDs

  for each citation:
    layer1 = existence_check(case_id)
    if not layer1: verdict = HALLUCINATED; continue

    layer2 = semantic_check(context)
    layer3 = connectivity_check(case_id)

    if layer2 and layer3:           verdict = REAL
    elif not layer2 and not layer3: verdict = HALLUCINATED
    else:                           verdict = SUSPICIOUS
```

### FastAPI Endpoint (`api/main.py`)

```
POST /check-citation
Body: { "text": "...raw AI-generated paragraph..." }
Response: {
    "citations": [
        {
            "citation_string": "392 U.S. 1",
            "verdict": "REAL | SUSPICIOUS | HALLUCINATED",
            "existence": bool,
            "semantic_score": float,
            "density_score": int,
            "top_matches": [...]
        }
    ]
}
```

### Scripts to Build

| Script                           | Purpose                                                       |
| -------------------------------- | ------------------------------------------------------------- |
| `detector/eyecite_parser.py`     | Extract citations from raw text, resolve to CourtListener IDs |
| `detector/existence_check.py`    | Layer 1 — Neo4j node lookup                                   |
| `detector/semantic_check.py`     | Layer 2 — hybrid ANN + BM25 search via RRF                    |
| `detector/connectivity_check.py` | Layer 3 — citation density score                              |
| `detector/pipeline.py`           | Orchestrate EyeCite + all three layers, return verdicts       |
| `detector/cache.py`              | TTLCache for embeddings and ANN results                       |
| `api/main.py`                    | FastAPI endpoint, request/response models                     |

---

## Week 7 — Benchmark Dataset Construction + Streamlit Scaffold

### Goals

Build a balanced benchmark of real and hallucinated citations to evaluate the detector
in Week 8. Stand up the Streamlit frontend with basic input and verdict display wired
to the FastAPI backend.

### Benchmark Design

- **Size:** 200 citations total (tune up/down based on time)
- **Split:** 50% real (100), 50% hallucinated (100)
- **Hallucinated subtypes** (3 equal groups of ~33):
  - **Type A — Fabricated entirely** — case name, year, and citation string invented
  - **Type B — Real case, wrong details** — real case name with wrong year or court
  - **Type C — Plausible but nonexistent** — realistic-sounding name in right style
    (e.g., _"United States v. Torres, 9th Cir. 2019"_) that doesn't exist

### Real Citation Sampling

Sample corpus cases with verified `opinions_cited` links. Use EyeCite to extract
the formatted citation string (e.g. `"392 U.S. 1"`) from `plain_text` — this gives
you the citation in the format that would appear in AI-generated text, paired with
a real surrounding context paragraph. Confirm the cited case exists in Neo4j before
including. Use stratified sampling across circuits and years to avoid bias.

### Hallucinated Citation Generation

- **Type A/C:** generate via Claude API with a prompt instructing it to invent
  plausible-sounding Fourth Amendment citations
- **Type B:** take real cases from corpus and corrupt year (+/- 2 years) or
  swap the court ID

### Output

`benchmark/benchmark.json` — list of objects:

```json
{
  "citation": "United States v. Torres, 923 F.3d 1027 (9th Cir. 2019)",
  "context": "...surrounding paragraph text...",
  "label": "REAL | HALLUCINATED",
  "subtype": null | "A" | "B" | "C",
  "case_id": 12345 | null
}
```

### Streamlit App Scaffold (`frontend/app.py`)

Stand up the frontend shell in Week 7 so it can be wired to the full pipeline.
LLM explanations are added in Week 9 — the Week 7 version just displays raw verdicts.

```
UI layout:
  - Text area: paste AI-generated legal text
  - "Check Citations" button → POST to FastAPI /check-citation
  - Results table: one row per citation
      - Citation string
      - Verdict badge (🟢 REAL / 🟡 SUSPICIOUS / 🔴 HALLUCINATED)
      - Semantic score
      - Density score
  - Expandable detail per citation (top matches from corpus)
```

### Scripts to Build

| Script                            | Purpose                                             |
| --------------------------------- | --------------------------------------------------- |
| `benchmark/generate_benchmark.py` | Build balanced real/hallucinated benchmark dataset  |
| `frontend/app.py`                 | Streamlit UI scaffold — verdict display, no LLM yet |

### API Security

#### Current State (Week 6)

- **Input size limit:** requests exceeding 50,000 characters are rejected with HTTP 400.
  Prevents runaway legal-bert inference on oversized payloads.
- **CORS:** restricted to `http://localhost:8501` (Streamlit default port).
- **No authentication:** single-user local deployment, no API keys required.
- **No rate limiting:** not needed for local use.

### Production / Monetization Roadmap

If Verit is deployed as a commercial API, add these in order:

1. **API key authentication** — issue keys per user/tier. FastAPI supports this
   natively via `APIKeyHeader` dependency injection.

2. **Per-key rate limiting** — use `slowapi` (wraps the `limits` library, integrates
   cleanly with FastAPI). Typical SaaS tiers: free (10 req/min), pro (60 req/min),
   enterprise (unlimited). Install with: `pip install slowapi`.

3. **HTTPS** — terminate TLS at a reverse proxy (nginx or Caddy) in front of uvicorn.
   Never expose uvicorn directly to the public internet.

4. **Request logging** — log per-key usage to a database for billing and abuse detection.
   FastAPI middleware makes this straightforward to add without touching endpoint logic.

---
# Week 8 — Evaluation + Threshold Tuning + Layer 4 + Benchmark Expansion
 
## Goals
 
Evaluate the detector against the benchmark, tune thresholds on a held-out validation
set, add Layer 4 metadata validation to catch subtype B hallucinations, expand the
benchmark to 500 entries, and run 10-fold cross-validation to verify score stability.
 
---
 
## Changes to Existing Files
 
### Project Overview — updated
 
The pipeline is now **four layers**, not three. Add Layer 4 to the What It Does list:
 
> 4. **Metadata validation** — does the year and court in the citation string match the Neo4j node properties?
 
### config.py — updated values
 
```python
SIMILARITY_THRESHOLD       = 0.60   # tuned in Week 8 (was 0.75)
RRF_THRESHOLD              = 0.010  # added in Week 8
CITATION_DENSITY_THRESHOLD = 1      # tuned in Week 8 (was 3)
```
 
### Neo4j status — updated
 
`court_id` backfilled from `cases_enriched.parquet` into all Case nodes via
`db/backfill_court_id.py`. After the backfill, 5 landmark nodes still had no
`court_id` — landmarks are fetched via `db/fetch_landmarks.py` and are not
present in the parquet file. All 5 are SCOTUS cases; patched manually with
`court_id = 'scotus'`. `fetch_landmarks.py` now writes `court_id` directly
during the upsert so this does not recur on graph rebuilds. `backfill_court_id.py`
now runs a post-migration verification query and warns if any non-stub nodes
still lack `court_id` after the batch run. Verified at zero after patch:
 
```cypher
MATCH (c:Case {stub: false}) WHERE c.court_id IS NULL RETURN count(c);
// Returns 0
```
 
### Pipeline pseudocode — updated (`detector/pipeline.py`)
 
Layer 4 now runs between Layer 1 and Layers 2/3:
 
```
layer1 = existence_check(case_id)
if not layer1: verdict = HALLUCINATED; continue
 
layer4 = metadata_check(case_id, citation_string)
if layer4 fails: verdict = HALLUCINATED; continue
 
layer2 = semantic_check(context)
layer3 = connectivity_check(case_id)
 
if layer2 and layer3:           verdict = REAL
elif not layer2 and not layer3: verdict = HALLUCINATED
else:                           verdict = SUSPICIOUS
```
 
---
 
## New Scripts
 
| Script | Purpose |
|---|---|
| `detector/metadata_check.py` | Layer 4 — validate year + court in citation string vs Neo4j node |
| `db/backfill_court_id.py` | One-time migration: write court_id from parquet into Neo4j Case nodes |
| `benchmark/evaluate.py` | Stratified 80/20 split, 180-combo threshold sweep on val set, saves `tuned_thresholds.json` |
| `benchmark/report.py` | Final metrics on held-out test set, per-layer and combined, saves `eval_report.json` |
| `benchmark/expand_benchmark.py` | Expand benchmark from 200 → 500 entries, weighted toward Type B |
| `benchmark/cross_validate.py` | 10-fold stratified CV, mean ± std per metric, fold checkpointing, saves `cv_report.json` |
 
### New files in `benchmark/`
 
| File | Description |
|---|---|
| `tuned_thresholds.json` | Best thresholds from val sweep (in .gitignore) |
| `eval_report.json` | Test-set metrics per layer and combined (in .gitignore) |
| `cv_report.json` | Full cross-validation results with per-fold breakdown (in .gitignore) |
| `split_indices.json` | Cached 80/20 split — never delete mid-project |
| `cv_checkpoint.json` | Fold-level CV checkpoint — deleted on successful completion |
 
---
 
## Layer 4 — Metadata Validation (`detector/metadata_check.py`)
 
Added after the first evaluation run showed all 7 subtype B hallucinations passing
Layers 1–3 with full confidence. This was expected — a Type B citation uses a real
`case_id`, so it exists in Neo4j, has a valid semantic footprint, and has strong citation
density. Layers 1–3 have no signal to distinguish it from a genuine citation.
 
Layer 4 extracts the court identifier and year from the citation string and compares
them against the actual properties on the Neo4j Case node. A mismatch → HALLUCINATED.
 
**Court extraction — two strategies, tried in order:**
 
1. **Direct CourtListener ID match** — bare court_id in trailing parenthetical,
   e.g. `"476 U.S. 207 (ca11)"` → extracts `ca11`. Catches the benchmark Type B
   corruption format where a court_id was injected into the citation string.
2. **Alias match** — natural-language court strings in formatted citations,
   e.g. `"923 F.3d 1027 (4th Cir. 2019)"` → `ca4`.
 
Layer 4 is skipped (`is_valid=True`, no penalty) when no year or court can be extracted
from the citation string — pure reporter citations like `"392 U.S. 1"` have no
parenthetical metadata and cannot be validated.
 
**Why `court_id` needed backfilling:** The Neo4j graph was built in Week 3 without
`court_id` as a node property. `db/backfill_court_id.py` reads `court_id` from
`cases_enriched.parquet` (zero nulls across 1,353 cases) and writes it to all
matching Case nodes in batches of 200. Without this, Layer 4's court comparison
always returned `None` and fell through to `is_valid=True`, making it a no-op.
 
---
 
## Benchmark Expansion (`benchmark/expand_benchmark.py`)
 
The original 200-entry benchmark produced F1=1.0 on 40 test entries — a real result
but statistically weak at that sample size. The benchmark was expanded to 500 entries
to reduce variance and stress-test the pipeline more rigorously.
 
**Expansion targets (300 new entries):**
 
| Type | Original | Added | Final | Notes |
|---|---|---|---|---|
| Real | 100 | 150 | 250 | Same stratified EyeCite sampling |
| Type A | 33 | 40 | 73 | Fully fabricated via Claude API |
| Type B | 34 | 70 | 104 | Weighted heavier — hardest subtype |
| Type C | 33 | 40 | 73 | Plausible nonexistent via Claude API |
| **Total** | **200** | **300** | **500** | |
 
Type B expansion uses 60% court corruption / 40% year corruption (vs 50/50 in original)
to stress Layer 4's court extraction path more aggressively.
 
Expansion checkpoints saved to `benchmark/expand_checkpoint_*.json` during generation,
deleted automatically on completion. Uses `random.seed(99)` (different from original
`seed=42`) to ensure different samples are drawn.
 
---
 
## Threshold Tuning
 
`evaluate.py` sweeps 180 combinations (6 × SIM, 6 × RRF, 5 × DENSITY) on the
validation set (400 entries after 80/20 split on 500). Selects highest F1, ties
broken by precision — fewer false alarms preferred in a legal context.
 
The sweep landed at minimum values across all three parameters. This reflects that
Layers 2 and 3 contribute no independent signal on Type B hallucinations (fully handled
by Layer 4) and that no false positives were observed at any threshold level.
 
**Tuned thresholds:**
 
| Parameter | Config Key | Original | Tuned |
|---|---|---|---|
| Cosine similarity floor | `SIMILARITY_THRESHOLD` | 0.75 | 0.60 |
| RRF score floor | `RRF_THRESHOLD` | — | 0.010 |
| Citation density minimum | `CITATION_DENSITY_THRESHOLD` | 3 | 1 |
 
---
 
## Test Set Results (held-out 20% — 100 entries)
 
| Layer | Precision | Recall | F1 | Notes |
|---|---|---|---|---|
| Layer 1 — Existence | 1.000 | 0.580 | 0.734 | Catches Type A + C immediately |
| Layer 2 — Semantic | 0.000 | 0.000 | 0.000 | No independent signal on Type B |
| Layer 3 — Connectivity | 0.000 | 0.000 | 0.000 | No independent signal on Type B |
| Layer 4 — Metadata | 1.000 | 0.952 | 0.976 | 1 FN — corpus year data quality issue |
| **Combined** | **1.000** | **0.980** | **0.990** | Zero FP, one FN (benchmark_id=171) |
 
Subtype F1: A=1.000, B=0.976, C=1.000. Zero SUSPICIOUS verdicts. Zero false positives.
 
**The one FN (benchmark_id=171):** Type B year-corrupted citation. The benchmark
corrupted the year to `2025`; the Neo4j node for that case (`United States v. Moses,
ca3`) also stores `year=2025` due to a corpus data quality issue. Layer 4 compares
`2025 == 2025`, finds no mismatch, and passes it. This is undetectable at the
Layer 4 level — the signal does not exist in the graph. Root cause: corpus year
data quality, not a Layer 4 bug.
 
---
 
## 10-Fold Cross-Validation Results (full 500-entry benchmark)
 
| Layer | Precision | Recall | F1 |
|---|---|---|---|
| Layer 1 — Existence | 1.000 ± 0.000 | 0.584 ± 0.020 | 0.737 ± 0.016 |
| Layer 2 — Semantic | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Layer 3 — Connectivity | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Layer 4 — Metadata | 1.000 ± 0.000 | 0.874 ± 0.106 | 0.929 ± 0.062 |
| **Combined** | **1.000 ± 0.000** | **0.944 ± 0.045** | **0.971 ± 0.024** |
 
Fold F1 range: 0.936 – 1.000. No anomalous folds (threshold: drop > 0.05 below mean).
 
**CV runtime note:** estimated ~125 min, actual ~12 min. The estimate was wrong —
each entry is only evaluated once (in its test fold), not 10 times. Total inference
calls = 500, not 5,000. Model and indexes stay loaded in memory across all folds.
 
---
 
## Honest Assessment of Evaluation Results
 
### On the test-set F1 (0.990)
 
The 0.990 on 100 held-out entries is genuine. One false negative exists (benchmark_id=171),
a Type B year-corrupted citation where the Neo4j node stores the same wrong year the
benchmark injected — a corpus data quality issue, not an architectural gap. All other
hallucination subtypes were correctly classified:
 
- Type A and C are fabricated → Layer 1 catches them by definition
- Type B corruptions inject a court ID or wrong year into the citation string → Layer 4
  catches them when the mismatch is detectable in the graph; the one FN is the
  edge case where the graph itself carries the wrong value
 
A benchmark of real cases cited for propositions they don't support (wrong legal context
rather than wrong metadata) would not score 0.990 on any current layer.
 
### On Layers 2 and 3 showing F1=0.0 in isolation
 
This is not a malfunction. Every hallucinated case reaching Layers 2 and 3 is a Type B,
and Type B cases carry valid semantic scores and strong citation network footprints from
the underlying real case. The isolated metrics honestly show what each layer contributes
individually. Layers 2 and 3 provide redundancy insurance for edge cases not covered by
this benchmark — e.g. corpus cases that exist but were not indexed in Milvus, or cases
with ambiguous CourtListener resolution.
 
### On Layer 4 recall variance (±0.106 across folds)
 
Some folds contain Type B entries that Layer 4 cannot catch — specifically year-corrupted
citations where the Neo4j node carries the same (incorrect) year as the corrupted benchmark
entry, or cases where the corrupted court happens to match the actual node's court_id.
The test-set FN (benchmark_id=171) is a confirmed example of the former. This is the most
meaningful signal from the CV results and the primary limitation to document in the writeup.
 
### Out-of-scope hallucination types
 
The most dangerous real-world hallucination — a real case cited for a proposition it does
not support — is undetectable by any current layer. Detection would require reading and
reasoning about the full opinion text, not just checking existence and metadata. This is
the central limitation to address in the future work section of the writeup.

---

## Week 9 — Error Analysis + UMAP Visualization + LLM Integration

### Goals

Visualize the embedding space to understand where hallucinated citations land relative
to real ones, show citation density distribution across the corpus, and wire Claude Haiku
into the Streamlit frontend to explain verdicts and suggest corrections.

### UMAP Dimensionality Reduction (`visualization/umap_viz.py`)

Reduce 768-dim case embeddings to 2D using UMAP for visualization.

- **StandardScaler first** — before UMAP, apply `sklearn.preprocessing.StandardScaler`
  to zero-mean and unit-variance each of the 768 dimensions. UMAP is sensitive to
  feature scale — without this, high-variance dimensions dominate the projection and
  distort the 2D layout. Applied to the embedding matrix loaded from parquet, not
  to the Milvus vectors.
- **Why UMAP over PCA:** UMAP preserves local neighborhood structure better than PCA,
  which matters here — we want to see how semantically similar cases cluster together.
  PCA is a linear projection and tends to compress legal text embeddings into a single
  dense blob. UMAP reveals substructure (e.g., warrant cases vs. stop-and-frisk cases).
- **Parameters:** `n_neighbors=15, min_dist=0.1, metric='cosine'` — cosine metric
  matches the similarity metric used in Milvus.
- **Overlay:** color points by circuit, year, or hallucination label (real vs. fake)
  to show where hallucinated citations fall in the semantic space.

### Citation Density Visualization

Plot the distribution of citation density scores (Layer 3) across real vs. hallucinated
citations from the benchmark. This should show a clear separation — real cases cluster
with the corpus, hallucinated cases are isolated.

### Visualizations to Produce

| Visualization                   | What It Shows                                           |
| ------------------------------- | ------------------------------------------------------- |
| UMAP of corpus embeddings       | How cases cluster semantically in 2D                    |
| UMAP with hallucination overlay | Where fake citations land vs. real ones                 |
| Citation density histogram      | Score distribution — real vs. hallucinated              |
| Precision-recall curve          | Detector performance across thresholds (Layers 2 and 3) |

### LLM Integration — Claude Haiku (`frontend/llm.py`)

Claude Haiku is called after the three-layer pipeline returns verdicts. Retrieved corpus
cases from Layer 2 (`top_matches`) are passed as RAG context so Haiku's explanations are
grounded in the actual corpus, not just its training data.

```
Input:  citation_string, verdict, semantic_score, density_score, top_matches (list of corpus cases)
Output: plain-English explanation + (if HALLUCINATED) suggested correction

RAG pattern:
  1. Layer 2 retrieves top-k corpus cases most similar to the citation context
  2. Case names + excerpts passed to Haiku as context in the system prompt
  3. Haiku explains why the citation is real/suspicious/hallucinated
  4. If hallucinated, Haiku suggests the closest real case from top_matches

Prompt structure:
  System: "You are a legal citation verification assistant. The following are real
           Fourth Amendment cases from the corpus: [top_matches]. Use them to explain
           the verdict below and suggest a correction if needed."
  User:   "Citation: [citation_string]. Verdict: [verdict]. Semantic score: [score].
           Density score: [density]. Explain this verdict in plain English."
```

Haiku is called per-citation, not per-document, to keep latency and cost low.
Results are streamed into the Streamlit UI using `st.write_stream`.

### Scripts to Build

| Script                      | Purpose                                                            |
| --------------------------- | ------------------------------------------------------------------ |
| `visualization/umap_viz.py` | StandardScaler + UMAP + hallucination overlay                      |
| `frontend/llm.py`           | Claude Haiku integration — explain verdicts, suggest corrections   |
| `frontend/app.py`           | Update Streamlit UI to display LLM explanations alongside verdicts |

---

## Week 10 — Frontend Polish + Citation Graph Visualization + Final Writeup

### Goals

Complete the Streamlit frontend with an interactive citation graph visualization that
directly demonstrates the original project goal: find similar legal cases and visualize
their citation relationships. Write up and submit the final report.

### Interactive Citation Graph (`frontend/app.py`)

When a user submits legal text and gets verdicts back, they can click any citation to
open a graph visualization showing that case's neighborhood in the Neo4j citation network.

```
User flow:
  1. User pastes AI-generated legal text → clicks "Check Citations"
  2. Verdict table appears (REAL / SUSPICIOUS / HALLUCINATED per citation)
  3. User clicks a citation row → graph panel expands below
  4. Graph shows:
       - The cited case as the center node
       - Cases it cites (outbound CITES edges) — 1 hop out
       - Cases that cite it (inbound CITES edges) — 1 hop in
       - Corpus cases that share citations with it (Layer 3 neighbors)
  5. Node color encodes verdict (green/yellow/red)
  6. Node size encodes cite_count (more cited = larger)
  7. Clicking a node in the graph loads that case's details in a sidebar
```

Library: **pyvis** (renders interactive HTML graph inside Streamlit via `st.components.v1.html`).
Alternatively **streamlit-agraph** if pyvis layout is insufficient.

Cypher query to fetch neighborhood for a given case_id:

```cypher
MATCH (target:Case {id: $id})-[r:CITES]-(neighbor)
RETURN target, r, neighbor
LIMIT 50
```

### Final Writeup Sections

| Section                   | Content                                                       |
| ------------------------- | ------------------------------------------------------------- |
| Problem statement         | LLM hallucination in legal citation, stakes for practitioners |
| System architecture       | Three-layer pipeline + RAG + frontend diagram                 |
| Data                      | Corpus stats, Neo4j graph stats, preprocessing decisions      |
| Methods                   | BERT embedding, HNSW, BM25, RRF, Neo4j connectivity           |
| Evaluation                | Precision, recall, F1 on benchmark — per layer and combined   |
| Visualization             | UMAP plots, citation density histograms, graph UI screenshots |
| Limitations + future work | Corpus scope, threshold sensitivity, production scaling       |

### New Dependencies to Add Before Week 10

```
pyvis
```

### Scripts to Build / Update

| Script            | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `frontend/app.py` | Add citation graph panel — pyvis neighborhood visualization |

---

## Key Design Decisions

1. **Multi-circuit corpus** — removed ca9-only filter to get enough cases with plain text
2. **Post-2015 focus** — older cases have poor plain text coverage (~12% vs ~87%)
3. **Layer 3 — citation density (Option B)** — corpus cases do not cite landmark cases
   by CourtListener opinion ID, so landmark-anchored k-hop connectivity is not viable.
   Instead, Layer 3 measures citation overlap: how many cases does the cited case share
   with known corpus cases? A hallucinated case has no footprint in the citation network.
4. **Hybrid search via RRF** — pure vector search can miss relevant corpus cases when
   distinctive legal terms (case names, doctrines) appear in corpus text but compress
   poorly in embedding space. BM25 + HNSW fused via RRF improves Layer 2 candidate
   retrieval quality. BM25 does not detect hallucinations directly — that is the job
   of Layers 1 and 3. RRF chosen over weighted sum because it requires no weight tuning.
5. **Metadata pre-filtering** — filter before ANN search, not after, to preserve recall.
   Post-filtering discards candidates before ranking and degrades results significantly.
6. **Bulk insert then index** — Milvus HNSW quality degrades if index is built incrementally.
   Always insert all vectors first, then call `create_index`.
7. **TTLCache for embeddings + ANN results** — legal-bert inference is slow; caching at the
   API layer makes the benchmark evaluation loop in Week 8 practical.
8. **Structure-aware chunking** — paragraph boundaries with 1-paragraph overlap, 512 token ceiling
9. **Balanced benchmark** — 50/50 real vs hallucinated, hallucinated split into 3 subtypes
10. **Parquet for processed data** — JSON for raw/debugging, Parquet for all pipeline consumption
11. **HNSW over IVF_FLAT** — better recall at this corpus size, no quantization artifacts
12. **UMAP over PCA** — preserves local neighborhood structure in legal embedding space;
    PCA produces a dense blob, UMAP reveals substructure (warrant vs. stop-and-frisk cases)
13. **All thresholds tuned on validation set** — never test set (Week 8)
14. **RAG via top_matches** — Layer 2 top-k corpus cases passed as context to Claude Haiku.
    Haiku explanations are grounded in retrieved corpus cases, not generated from training
    data alone. This keeps explanations factually anchored to the actual corpus.
15. **Haiku over Opus/Sonnet for LLM layer** — explanation and suggestion tasks don't require
    the most powerful model. Haiku keeps latency and API cost minimal for per-citation calls.
16. **Streamlit over React** — pure Python frontend, no separate Node.js project. Sufficient
    for a grad project demo and integrates directly with the existing Python stack.
17. **pyvis for citation graph** — renders interactive HTML graphs inside Streamlit via
    `st.components.v1.html`. Directly fulfills the original project goal of visualizing
    citation relationships. Graph is scoped to 1-hop neighborhood (50 node limit) to keep
    rendering fast and readable.


### Key Design Decisions Added in Week 8
 
18. **Layer 4 added after first evaluation** — metadata validation catches Type B
    hallucinations that pass Layers 1–3 with full confidence. Requires `court_id`
    to be populated on Neo4j nodes — backfilled via `db/backfill_court_id.py`.
 
19. **Benchmark expanded to 500** — original 200-entry benchmark produced statistically
    weak 1.0 on 40 test entries. 300 new entries added, weighted toward Type B (70 new
    vs 40 each for A and C). 10-fold CV on full 500 confirms F1=0.971 ± 0.024 is stable.
 
20. **Citation normalization tradeoff documented** — `[CITATION]` token replacement in
    `clean_text.py` improves BERT embedding quality but removes case name signal from
    the BM25 index, which is built from cleaned text. BM25 keyword recall on case names
    is reduced as a result. Known limitation, not a bug.
 

---

## Known Issues / Things to Watch

- Landmark nodes are isolated in the graph (no CITES edges from corpus) — by design, not a bug
- `test_landmarks_are_reachable` test is skipped — landmark connectivity not used in Layer 3
- EyeCite scoped for Week 6 (query-time extraction) and Week 7 (benchmark real citation sampling) — not needed for graph building (uses CourtListener `opinions_cited` API field instead)
- `eyecite` not yet in requirements.txt — add before Week 6
- `preprocessing/tokenize_bm25.py` not yet built — needed before Week 5 BM25 index
- BM25 index not yet built (Week 5)
- `RRF_THRESHOLD` not yet defined in config — add after Week 5 hybrid search is implemented
- All thresholds (`SIMILARITY_THRESHOLD`, `RRF_THRESHOLD`, `CITATION_DENSITY_THRESHOLD`) not yet tuned (Week 8)
- HNSW parameters (`M`, `ef_construction`, `ef`) may need tuning in Week 8
- `umap-learn` not yet in requirements.txt — add before Week 9
- `pyvis` not yet in requirements.txt — add before Week 10

### Known Issues Added in Week 8
 
- Layer 4 recall varies across CV folds (±0.106) — some Type B year corruptions are
  undetectable when Neo4j node lacks a year property or the corrupted court matches real
- **One test-set FN (benchmark_id=171)** — Type B year-corrupted citation where
  `United States v. Moses (ca3)` has `year=2025` in Neo4j (incorrect); benchmark
  corrupted year to `2025`. Layer 4 sees no mismatch and passes it. Corpus data
  quality issue, not a Layer 4 bug. Undetectable without correcting the graph.
- Landmark nodes were missing `court_id` after backfill — parquet does not include
  landmark nodes (fetched separately). Patched manually to `court_id='scotus'`.
  `fetch_landmarks.py` now writes `court_id` on upsert; `backfill_court_id.py` now
  warns post-run if any non-stub nodes still lack `court_id`.
- Layers 2 and 3 show F1=0.0 in isolation — expected by design, not a bug (see above)
- Citation normalization reduces BM25 case-name signal — known tradeoff, documented

---

## How to Start Each Session

```powershell
# 1. Navigate to project
cd "C:\Users\ssalh\Grad School\2026\01_Spring\MIS6V99\Verit"

# 2. Activate venv
.venv\Scripts\activate

# 3. Start Docker Desktop (from Start menu, wait for whale icon)

# 4. Start Neo4j — wait 30 seconds before running any scripts
docker-compose up -d

# 5. Verify everything
docker ps
docker logs verit_neo4j --tail 3   # confirm "Started."
python -m pytest tests/ -v
```

## Useful cypher-shell Queries

```cypher
// Node counts
MATCH (c:Case) RETURN count(c) AS total;
MATCH (c:Case {stub: false}) RETURN count(c) AS full_nodes;
MATCH (c:Case {stub: true}) RETURN count(c) AS stubs;

// Edge count
MATCH ()-[r:CITES]->() RETURN count(r) AS edges;

// Landmarks
MATCH (c:Case {landmark: true}) RETURN c.name, c.id, c.year;

// Most cited cases in corpus
MATCH (c:Case)<-[:CITES]-(other)
RETURN c.name, count(other) AS cited_by
ORDER BY cited_by DESC LIMIT 10;

// Citation density check for a specific case ID
MATCH (target:Case {id: $id})-[:CITES]->(shared)<-[:CITES]-(corpus:Case {stub: false})
RETURN count(DISTINCT shared) AS shared_citations;
```
