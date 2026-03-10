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

---

## Tech Stack

| Component                | Tool                    | Purpose                                              |
| ------------------------ | ----------------------- | ---------------------------------------------------- |
| Vector Store             | Milvus Lite 2.4+        | Store and search 768-dim case embeddings (HNSW)      |
| Sparse Index             | BM25 (rank_bm25)        | Keyword search over plain_text for hybrid search     |
| Hybrid Fusion            | Reciprocal Rank Fusion  | Merge dense + sparse results into single ranked list |
| Graph Database           | Neo4j 5.15 (Docker)     | Store citation relationships as directed graph       |
| Embedding Model          | legal-bert-base-uncased | Convert legal text to semantic vectors               |
| Dimensionality Reduction | UMAP                    | 2D visualization of embedding space (Week 9)         |
| Citation Extraction      | EyeCite                 | Parse citation strings from raw text                 |
| API Layer                | FastAPI                 | Expose /check-citation endpoint                      |
| Query Cache              | cachetools TTLCache     | Cache embeddings + ANN results at API layer          |
| Infrastructure           | Docker + Docker Compose | Run Neo4j locally                                    |
| Language                 | Python 3.10             |                                                      |
| IDE                      | VS Code                 |                                                      |

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

| Metric              | Count  |
| ------------------- | ------ |
| Full Case nodes     | 1,358  |
| Stub nodes          | 14,773 |
| Total Case nodes    | 16,131 |
| CITES edges         | 30,806 |
| Landmark nodes      | 5      |

### Landmark Anchor Cases

Loaded via `db/fetch_landmarks.py`. Present in graph as full nodes with `landmark: true`.
These are isolated from the corpus citation network (corpus cases do not cite them by
CourtListener opinion ID) — landmark connectivity is not used in Layer 3 (see design decisions).

| Case                      | Year | CourtListener Opinion ID |
| ------------------------- | ---- | ------------------------ |
| Terry v. Ohio             | 1968 | 107729                   |
| Katz v. United States     | 1967 | 107564                   |
| Mapp v. Ohio              | 1961 | 106285                   |
| United States v. Leon     | 1984 | 111252                   |
| Illinois v. Gates         | 1983 | 110930                   |

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
MILVUS_DB_PATH    = os.path.join(ROOT_DIR, "milvus_verit.db")
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

| Week | Dates           | Milestone                                          | Status         |
| ---- | --------------- | -------------------------------------------------- | -------------- |
| 1    | Feb 24 – Mar 2  | Environment setup, Docker, Neo4j, first cases      | ✅ Complete    |
| 2    | Mar 3 – Mar 9   | Full data ingestion, Parquet pipeline                      | ✅ Complete    |
| 3    | Mar 10 – Mar 16 | Neo4j graph build and verification                 | ✅ Complete    |
| 4    | Mar 17 – Mar 23 | BERT embedding pipeline + vector pruning + Milvus  | 🔄 Up Next     |
| 5    | Mar 24 – Mar 30 | ANN search + semantic retrieval layer              | ⬜ Upcoming    |
| 6    | Mar 31 – Apr 6  | Hallucination detector — all three checks          | ⬜ Upcoming    |
| 7    | Apr 7 – Apr 13  | Benchmark dataset construction                     | ⬜ Upcoming    |
| 8    | Apr 14 – Apr 20 | Evaluation — precision, recall, F1                 | ⬜ Upcoming    |
| 9    | Apr 21 – Apr 27 | Error analysis + UMAP visualization                | ⬜ Upcoming    |
| 10   | Apr 28 – May 8  | Final writeup and submission                       | ⬜ Upcoming    |

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

### Embedding Batching

- Process cases through legal-bert in batches of 16-32 (larger batches hit memory limits)
- Save embeddings to parquet after each batch — crash recovery without re-running from scratch

### Scripts to Build

| Script                           | Purpose                                                    |
| -------------------------------- | ---------------------------------------------------------- |
| `preprocessing/clean_text.py`    | Strip headers, normalize citations, fix encoding           |
| `embeddings/prune_vectors.py`    | Filter corpus by text quality, output clean list           |
| `embeddings/embed_cases.py`      | Chunk, embed in batches, mean-pool, L2-normalize, parquet  |
| `embeddings/milvus_index.py`     | Bulk insert normalized vectors, then build HNSW index      |

---

## Week 5 — Hybrid Search + Semantic Retrieval Layer

### Goals

Implement hybrid search combining dense ANN search (Milvus HNSW) with sparse keyword
search (BM25) fused via Reciprocal Rank Fusion. Add metadata pre-filtering and query
embedding caching. This becomes Layer 2 of the detector.

### Why Hybrid Search

BM25 improves **Layer 2 retrieval quality** — it finds better corpus candidates to compare
against, not whether the cited case itself is real. A cited case name like *"Carpenter v.
United States"* may not be in your corpus, but the term `Carpenter` likely appears in
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

| Script                            | Purpose                                                        |
| --------------------------------- | -------------------------------------------------------------- |
| `preprocessing/tokenize_bm25.py`  | Lowercase, remove stopwords, lemmatize for BM25 corpus         |
| `embeddings/bm25_index.py`        | Build and serialize BM25 index over tokenized plain_text       |
| `detector/semantic_check.py`      | Hybrid search: ANN + BM25 fused via RRF, with pre-filtering    |
| `detector/cache.py`               | TTLCache for query embeddings and ANN results                  |

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

    if layer2 and layer3:     verdict = REAL
    elif not layer2 and not layer3: verdict = HALLUCINATED
    else:                     verdict = SUSPICIOUS
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

| Script                           | Purpose                                                         |
| -------------------------------- | --------------------------------------------------------------- |
| `detector/eyecite_parser.py`     | Extract citations from raw text, resolve to CourtListener IDs   |
| `detector/existence_check.py`    | Layer 1 — Neo4j node lookup                                     |
| `detector/semantic_check.py`     | Layer 2 — hybrid ANN + BM25 search via RRF                      |
| `detector/connectivity_check.py` | Layer 3 — citation density score                                |
| `detector/pipeline.py`           | Orchestrate EyeCite + all three layers, return verdicts         |
| `detector/cache.py`              | TTLCache for embeddings and ANN results                         |
| `api/main.py`                    | FastAPI endpoint, request/response models                       |

---

## Week 7 — Benchmark Dataset Construction

### Goals

Build a balanced benchmark of real and hallucinated citations to evaluate the detector
in Week 8. The benchmark is the ground truth for precision, recall, and F1 scoring.

### Benchmark Design

- **Size:** 200 citations total (tune up/down based on time)
- **Split:** 50% real (100), 50% hallucinated (100)
- **Hallucinated subtypes** (3 equal groups of ~33):
  - **Type A — Fabricated entirely** — case name, year, and citation string invented
  - **Type B — Real case, wrong details** — real case name with wrong year or court
  - **Type C — Plausible but nonexistent** — realistic-sounding name in right style
    (e.g., *"United States v. Torres, 9th Cir. 2019"*) that doesn't exist

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

### Scripts to Build

| Script                            | Purpose                                              |
| --------------------------------- | ---------------------------------------------------- |
| `benchmark/generate_benchmark.py` | Build balanced real/hallucinated benchmark dataset   |

---

## Week 8 — Evaluation + Threshold Tuning

### Thresholds to Tune (on validation set only — never test set)

| Parameter                   | Config Key                    | Starting Value | What It Controls                    |
| --------------------------- | ----------------------------- | -------------- | ------------------------------------ |
| Cosine similarity floor     | `SIMILARITY_THRESHOLD`        | 0.75           | Layer 2 — pure vector signal         |
| RRF score floor             | `RRF_THRESHOLD`               | TBD            | Layer 2 — hybrid signal              |
| Citation density minimum    | `CITATION_DENSITY_THRESHOLD`  | 3              | Layer 3                              |
| ANN top-k                   | `TOP_K`                       | 5              | Candidates returned per query        |
| HNSW ef (query time)        | `HNSW_EF`                     | 50             | Recall vs speed tradeoff             |

---

## Week 9 — Error Analysis + UMAP Visualization

### Goals

Visualize the embedding space to understand where hallucinated citations land relative
to real ones, and show citation density distribution across the corpus.

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

| Visualization                  | What It Shows                                              |
| ------------------------------ | ---------------------------------------------------------- |
| UMAP of corpus embeddings      | How cases cluster semantically in 2D                       |
| UMAP with hallucination overlay| Where fake citations land vs. real ones                    |
| Citation density histogram     | Score distribution — real vs. hallucinated                 |
| Precision-recall curve         | Detector performance across thresholds (Layers 2 and 3)   |

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

---

## Known Issues / Things to Watch

- Landmark nodes are isolated in the graph (no CITES edges from corpus) — by design, not a bug
- `test_landmarks_are_reachable` test is skipped — landmark connectivity not used in Layer 3
- EyeCite scoped for Week 6 (query-time extraction) and Week 7 (benchmark real citation sampling) — not needed for graph building (uses CourtListener `opinions_cited` API field instead)
- `eyecite` not yet in requirements.txt — add before Week 6
- `preprocessing/clean_text.py` not yet built — needed before Week 4 embedding can start
- `preprocessing/tokenize_bm25.py` not yet built — needed before Week 5 BM25 index
- BM25 index not yet built (Week 5)
- `RRF_THRESHOLD` not yet defined in config — add after Week 5 hybrid search is implemented
- Benchmark generation script not yet written (Week 7)
- All thresholds (`SIMILARITY_THRESHOLD`, `RRF_THRESHOLD`, `CITATION_DENSITY_THRESHOLD`) not yet tuned (Week 8)
- HNSW parameters (`M`, `ef_construction`, `ef`) may need tuning in Week 8
- `cachetools` not yet in requirements.txt — add before Week 6
- `rank_bm25` not yet in requirements.txt — add before Week 5
- `umap-learn` not yet in requirements.txt — add before Week 9
- `nltk` or `spacy` not yet in requirements.txt — needed for lemmatization in Week 5

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