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
│   ├── embed_cases.py            # Week 4 — generate 768-dim vectors via legal-bert (batched)
│   ├── milvus_index.py           # Week 4 — bulk insert vectors into Milvus, build HNSW index
│   └── bm25_index.py             # Week 5 — build BM25 sparse index over plain_text
├── preprocessing/
│   └── __init__.py
├── detector/
│   ├── __init__.py
│   ├── existence_check.py        # Week 6 — Layer 1: Neo4j node lookup
│   ├── semantic_check.py         # Week 6 — Layer 2: hybrid search (ANN + BM25 via RRF)
│   ├── connectivity_check.py     # Week 6 — Layer 3: citation density scoring
│   └── cache.py                  # Week 6 — TTLCache for query embeddings + ANN results
├── api/
│   └── __init__.py
├── benchmark/
│   └── __init__.py
└── tests/
    ├── __init__.py
    ├── conftest.py               # Fixtures: raw_cases, merged_cases
    ├── test_data.py              # 13 passing tests ✅
    ├── test_db.py                # 14 passing tests ✅ (Week 3)
    └── test_detector.py          # Empty — to be built in Week 6
```

---

## config.py Contents

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

| Week | Dates           | Milestone                                         | Status      |
| ---- | --------------- | ------------------------------------------------- | ----------- |
| 1    | Feb 24 – Mar 2  | Environment setup, Docker, Neo4j, first cases     | ✅ Complete |
| 2    | Mar 3 – Mar 9   | Full data ingestion, EyeCite parsing, Parquet     | ✅ Complete |
| 3    | Mar 10 – Mar 16 | Neo4j graph build and verification                | ✅ Complete |
| 4    | Mar 17 – Mar 23 | BERT embedding pipeline + vector pruning + Milvus | 🔄 Up Next  |
| 5    | Mar 24 – Mar 30 | ANN search + semantic retrieval layer             | ⬜ Upcoming |
| 6    | Mar 31 – Apr 6  | Hallucination detector — all three checks         | ⬜ Upcoming |
| 7    | Apr 7 – Apr 13  | Benchmark dataset construction                    | ⬜ Upcoming |
| 8    | Apr 14 – Apr 20 | Evaluation — precision, recall, F1                | ⬜ Upcoming |
| 9    | Apr 21 – Apr 27 | Error analysis + UMAP visualization               | ⬜ Upcoming |
| 10   | Apr 28 – May 8  | Final writeup and submission                      | ⬜ Upcoming |

---

## Week 4 — BERT Embedding Pipeline + Vector Pruning + Milvus Indexing

### Goals

Generate 768-dimensional embeddings for all 1,353 corpus cases using
`legal-bert-base-uncased`, apply vector pruning to remove low-quality inputs,
and index the result in Milvus Lite using an HNSW index for fast ANN search.

### Vector Pruning (`embeddings/prune_vectors.py`)

Before embedding, filter out cases that would produce unreliable vectors:

- **Too short** — `plain_text` under `MIN_TEXT_LENGTH` characters (200). Very short
  opinions are often procedural orders with no substantive Fourth Amendment content.
  These would cluster artificially close to each other in embedding space.
- **Too long** — truncate at `MAX_TEXT_LENGTH` (50,000 characters) before passing
  to BERT. legal-bert has a 512-token limit; long texts will be chunked or truncated.
- **Missing text** — cases where `plain_text` is null or empty are dropped entirely.

Pruning happens before embedding — pruned cases remain in Neo4j as nodes but are
not indexed in Milvus.

### Embedding (`embeddings/embed_cases.py`)

- Model: `nlpaueb/legal-bert-base-uncased` (768-dim, pretrained on legal corpora)
- Chunking: structure-aware paragraph chunking with 1-paragraph overlap, 512-token ceiling
- Strategy: mean-pool the token embeddings across chunks to produce one vector per case
- Output: saved to `data/processed/embeddings.parquet` (case_id + 768-dim vector)

### Milvus Indexing (`embeddings/milvus_index.py`)

- Index type: **HNSW** (Hierarchical Navigable Small World)
  - Better recall than IVF_FLAT at this corpus size (~1,300 vectors)
  - Fast at query time — no quantization artifacts
  - Parameters: `M=16, ef_construction=200, ef=50` (tune in Week 8)
- Metric: cosine similarity
- Collection schema: `case_id` (int64) + `embedding` (float_vector, dim=768)
- **Critical:** bulk insert all vectors first, then call `create_index` — Milvus builds
  a better HNSW graph when it sees the full dataset at once. Never create index then
  insert incrementally.
- Insert in batches of 500-1000 to avoid memory spikes.

### Embedding Batching (`embeddings/embed_cases.py`)

- Process cases through legal-bert in batches of 16-32 (larger batches hit memory limits)
- Save embeddings to parquet after each batch — crash recovery without re-running from scratch
- Mean-pool token embeddings across chunks to produce one 768-dim vector per case

### Scripts to Build

| Script                        | Purpose                                                  |
| ----------------------------- | -------------------------------------------------------- |
| `embeddings/prune_vectors.py` | Filter corpus by text quality, output clean list         |
| `embeddings/embed_cases.py`   | Generate embeddings in batches of 16-32, save to parquet |
| `embeddings/milvus_index.py`  | Bulk insert into Milvus, then build HNSW index           |

---

## Week 5 — Hybrid Search + Semantic Retrieval Layer

### Goals

Implement hybrid search combining dense ANN search (Milvus HNSW) with sparse keyword
search (BM25) fused via Reciprocal Rank Fusion. Add metadata pre-filtering and query
embedding caching. This becomes Layer 2 of the detector.

### Why Hybrid Search

A hallucinated case like _"United States v. Johnson, 2019"_ might score moderate cosine
similarity (the surrounding legal context sounds Fourth Amendment) but score zero on BM25
(the case name doesn't appear anywhere in the corpus). Hybrid search catches the gap that
pure vector search misses.

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

| Script                       | Purpose                                                     |
| ---------------------------- | ----------------------------------------------------------- |
| `embeddings/bm25_index.py`   | Build and serialize BM25 index over corpus plain_text       |
| `detector/semantic_check.py` | Hybrid search: ANN + BM25 fused via RRF, with pre-filtering |
| `detector/cache.py`          | TTLCache for query embeddings and ANN results               |

---

## Week 8 — Evaluation + Threshold Tuning

### Thresholds to Tune (on validation set only — never test set)

| Parameter                | Config Key                   | Starting Value | What It Controls              |
| ------------------------ | ---------------------------- | -------------- | ----------------------------- |
| Cosine similarity floor  | `SIMILARITY_THRESHOLD`       | 0.75           | Layer 2 — pure vector signal  |
| RRF score floor          | `RRF_THRESHOLD`              | TBD            | Layer 2 — hybrid signal       |
| Citation density minimum | `CITATION_DENSITY_THRESHOLD` | 3              | Layer 3                       |
| ANN top-k                | `TOP_K`                      | 5              | Candidates returned per query |
| HNSW ef (query time)     | `HNSW_EF`                    | 50             | Recall vs speed tradeoff      |

---

## Week 9 — Error Analysis + UMAP Visualization

### Goals

Visualize the embedding space to understand where hallucinated citations land relative
to real ones, and show citation density distribution across the corpus.

### UMAP Dimensionality Reduction

Reduce 768-dim case embeddings to 2D using UMAP for visualization.

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

---

## Key Design Decisions

1. **Multi-circuit corpus** — removed ca9-only filter to get enough cases with plain text
2. **Post-2015 focus** — older cases have poor plain text coverage (~12% vs ~87%)
3. **Layer 3 — citation density (Option B)** — corpus cases do not cite landmark cases
   by CourtListener opinion ID, so landmark-anchored k-hop connectivity is not viable.
   Instead, Layer 3 measures citation overlap: how many cases does the cited case share
   with known corpus cases? A hallucinated case has no footprint in the citation network.
4. **Hybrid search via RRF** — pure vector search misses hallucinated cases with plausible
   semantic context but zero keyword presence. BM25 + HNSW fused via RRF catches both.
   RRF chosen over weighted sum because it requires no weight tuning and is robust by default.
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
- EyeCite parsing not yet implemented — currently using `opinions_cited` URLs from CourtListener API
- BM25 index not yet built (Week 5)
- `RRF_THRESHOLD` not yet defined in config — add after Week 5 hybrid search is implemented
- Benchmark generation script not yet written (Week 7)
- All thresholds (`SIMILARITY_THRESHOLD`, `RRF_THRESHOLD`, `CITATION_DENSITY_THRESHOLD`) not yet tuned (Week 8)
- HNSW parameters (`M`, `ef_construction`, `ef`) may need tuning in Week 8
- `cachetools` not yet in requirements.txt — add before Week 6
- `rank_bm25` not yet in requirements.txt — add before Week 5
- `umap-learn` not yet in requirements.txt — add before Week 9

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
