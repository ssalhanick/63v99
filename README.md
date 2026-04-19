# Verit

### Legal Citation Hallucination Detector

> Automatically verify whether citations in AI-generated legal text are real, relevant, and properly connected — using semantic similarity, vector search, and graph traversal. Focused on Fourth Amendment (Search & Seizure) federal case law.

---

## The Problem

LLMs are increasingly used to draft legal documents, briefs, and opinions. The problem is that LLMs frequently **hallucinate citations**, which has the potential to invent case names that sound real but don't exist, or cite real cases that have nothing to do with the argument being made. This has already resulted in real-world sanctions against lawyers who submitted AI-generated briefs without verifying their citations.

In Fourth Amendment law specifically, LLMs routinely confuse standards across contexts. This presents itself through misattributing the good faith exception, conflating the reasonable expectation of privacy test with the warrant requirement, or citing cases from the wrong circuit or era. Verit is designed to catch exactly these failures automatically.

Existing tools like LexisNexis are research interfaces — they help lawyers _find_ cases. Verit solves a different problem entirely: it verifies citations that **already appear** in AI-generated text, catching three distinct failure modes:

| Hallucination Type      | Description                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------- |
| **Fabricated**          | The cited case does not exist anywhere in the legal corpus                                          |
| **Real but Irrelevant** | The case exists but is semantically unrelated to the argument                                       |
| **Misconnected**        | The case exists and is topically plausible but is not graph-connected to the legal doctrine at hand |

---

## The Solution

Verit runs every citation through a **four-layer** verification pipeline:

```
AI-Generated Legal Text
        │
        ▼
 EyeCite Extraction  →  Parse all citation strings from text
        │
        ▼
 Layer 1: Neo4j      →  Does this case exist as a node in the graph?
        │
        ▼
 Layer 4: Neo4j      →  Do the year and court in the citation match the node?
        │
        ▼
 Layer 2: Milvus     →  Is it semantically relevant? (hybrid ANN + BM25 via RRF)
        │
        ▼
 Layer 3: Neo4j      →  Is it graph-connected to the legal topic?
        │
        ▼
 Verdict + Scores
 { "verdict": "REAL" | "SUSPICIOUS" | "HALLUCINATED",
   "existence": bool, "semantic_score": float, "density_score": int }
```

Each layer catches what the others miss. A citation can pass the existence check and still be semantically irrelevant. A citation can score high on semantic similarity and still be disconnected from the citing document's legal doctrine in the real citation network. All four checks together produce a meaningfully more accurate verdict than any single check alone.

---

## What Makes This Different from LexisNexis

LexisNexis is a **research tool** — it helps lawyers find relevant law. Verit is a **verification tool** — it plugs into an AI writing pipeline and programmatically checks citations the way a linter checks code. Key differences:

- LexisNexis does not address LLM hallucination — it was not designed for this problem
- Shepard's Citations tells you if a case is still good law; Verit tells you if a citation is _appropriate to the argument being made_
- Verit combines semantic similarity and graph topology into a single confidence score — LexisNexis treats these as separate tools
- Verit is API-driven and automatable; LexisNexis is a human-facing interface

---

## Tech Stack

| Component           | Tool                        | Purpose                                                       |
| ------------------- | --------------------------- | ------------------------------------------------------------- |
| Vector Store        | **Milvus Lite**             | Store and search 768-dim case embeddings by cosine similarity |
| Graph Database      | **Neo4j** (Docker)          | Store citation relationships as a directed graph              |
| Embedding Model     | **legal-bert-base-uncased** | Convert legal text to semantic vectors                        |
| Citation Extraction | **EyeCite**                 | Parse and resolve citation strings from raw text              |
| API Layer           | **FastAPI**                 | Expose the `/check-citation` endpoint                         |
| Infrastructure      | **Docker + Docker Compose** | Run Neo4j locally without system install                      |
| IDE                 | **VS Code**                 | Development environment                                       |

---

## Data Sources

### Primary Case Corpus

- **CourtListener** — Federal circuit court opinions filtered to Fourth Amendment, search and seizure, and unreasonable search cases. Pulled across all federal circuits (anchored in 9th Circuit analysis) via the free bulk REST API at `courtlistener.com/api`. Each record includes full opinion text and structured citation metadata used to build the Neo4j graph. Key anchor cases — _Terry v. Ohio_, _Katz v. United States_, _Mapp v. Ohio_, and _United States v. Leon_ — serve as high-citation hub nodes in the graph.

### Benchmark Datasets

- **CLERC** (Hugging Face) — Pre-chunked U.S. legal case retrieval dataset with labeled citation pairs. Used as the "real citation" half of the evaluation benchmark.
- **CaseHOLD** (AI2 / Hugging Face) — Multiple-choice QA benchmark over U.S. legal holdings. Used to evaluate citation relevance classification.

### Hallucinated Citation Generation

The "hallucinated" half of the benchmark is generated by:

1. Taking real legal paragraphs with known correct citations
2. Prompting an LLM to rewrite them with fabricated citations
3. Swapping real citations with real-but-unrelated cases from the corpus
4. Corrupting real citations by altering year, volume, or page number

---

## Project Structure

```
Verit/
├── docker-compose.yml              # Neo4j + Milvus container config
├── requirements.txt
├── config.py                       # Single source of truth for all paths and credentials
├── README.md
│
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI — POST /check-citation, GET /health
│
├── benchmark/
│   ├── __init__.py
│   ├── benchmark.json              # 500-citation evaluation dataset (250 real / 250 hallucinated)
│   ├── cross_validate.py           # 10-fold stratified cross-validation
│   ├── cv_report.json              # Cross-validation results
│   ├── density_histogram.py        # Layer 3 density score distribution plot
│   ├── eval_report.json            # Test-set evaluation results
│   ├── evaluate.py                 # Threshold sweep on validation set
│   ├── expanded_benchmark.py       # Expands benchmark from 200 → 500 entries
│   ├── generate_benchmark.py       # Builds initial 200-citation benchmark
│   ├── report.py                   # Runs inference on held-out test set
│   ├── split_indices.json          # Cached 80/20 val/test split (never reshuffled)
│   └── tuned_thresholds.json       # Best threshold combo from evaluate.py sweep
│
├── data/
│   ├── __init__.py
│   ├── convert_to_parquet.py       # Converts enriched JSON → Parquet
│   ├── data_check.py               # Quick corpus sanity checks
│   ├── diagnose_batches.py         # Diagnose batch merge issues
│   ├── diagnose_text.py            # Inspect plain-text coverage
│   ├── fetch_all_opinions.py       # Fetches full plain_text for each case
│   ├── fetch_cases.py              # CourtListener API ingestion (4th Amendment filter)
│   ├── merge_batches.py            # Deduplicates and merges batch JSONs
│   ├── raw/                        # Raw JSON from CourtListener (not in Git)
│   │   ├── batch_2010_2015.json
│   │   ├── batch_2015_present.json
│   │   ├── cases_merged.json
│   │   ├── enriched_2010_2015.json
│   │   └── enriched_2015_present.json
│   └── processed/                  # Parquet files for pipeline consumption (not in Git)
│       ├── bm25_index.pkl          # Serialized BM25 index
│       ├── cases_cleaned.parquet   # HTML-stripped, citation-normalized text
│       ├── cases_enriched.parquet  # Full corpus with opinion text + metadata
│       ├── cases_pruned.parquet    # Filtered to cases with usable plain text
│       ├── cases_tokenized.parquet # spaCy-tokenized for BM25
│       └── embeddings.parquet      # 768-dim legal-bert vectors per case
│
├── db/
│   ├── __init__.py
│   ├── backfill_court_id.py        # One-time migration: writes court_id to Neo4j nodes
│   ├── fetch_landmarks.py          # Fetches SCOTUS landmark cases from CourtListener
│   ├── graph_loader.py             # Bulk loads corpus into Neo4j (nodes + CITES edges)
│   └── neo4j_client.py             # Neo4j Bolt connection helper
│
├── detector/
│   ├── __init__.py
│   ├── cache.py                    # TTLCache for embeddings and ANN results
│   ├── connectivity_check.py       # Layer 3 — citation density score via Neo4j
│   ├── existence_check.py          # Layer 1 — Neo4j node lookup by cluster ID
│   ├── eyecite_parser.py           # EyeCite extraction + CourtListener cluster ID resolution
│   ├── metadata_check.py           # Layer 4 — year and court validation against Neo4j node
│   ├── pipeline.py                 # Orchestrates all four layers; returns CitationVerdict
│   └── semantic_check.py           # Layer 2 — hybrid ANN (Milvus) + BM25 via RRF
│
├── embeddings/
│   ├── __init__.py
│   ├── bm25_index.py               # Builds and serializes BM25Okapi index
│   ├── embed_cases.py              # Paragraph chunking + legal-bert inference + mean-pool
│   ├── milvus_index.py             # Bulk insert + HNSW index build in Milvus
│   └── prune_vectors.py            # Filters corpus to cases with sufficient plain text
│
├── frontend/
│   ├── __init__.py
│   ├── app.py                      # Streamlit UI — verdict badges, scores, top matches
│   └── llm.py                      # Claude Haiku LLM layer for natural-language explanation
│
├── lib/                            # Vendored JS libraries (vis.js, tom-select, bindings)
│   ├── bindings/
│   ├── tom-select/
│   └── vis-9.1.2/
│
├── preprocessing/
│   ├── __init__.py
│   ├── clean_text.py               # HTML stripping, header/footer removal, [CITATION] normalization
│   └── tokenize_bm25.py            # spaCy lemmatization for BM25 token corpus
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared pytest fixtures
│   ├── coverage_ratio.py           # Reports plain-text coverage across corpus
│   ├── test_data.py                # 13 data validation tests
│   ├── test_db.py                  # 14 Neo4j graph validation tests
│   ├── test_detector.py            # Hallucination detector tests
│   └── test_stubID.py              # Verifies stub node ID integrity
│
└── visualization/
    ├── __init__.py
    ├── graph_viz.py                # pyvis citation graph visualization
    ├── umap_viz.py                 # UMAP dimensionality reduction + Plotly scatter
    ├── density_histogram.png       # Layer 3 density score distribution (exported)
    ├── umap_circuit.html           # UMAP colored by circuit (exported)
    └── umap_year.html              # UMAP colored by year (exported)
```

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Python 3.11+
- VS Code (recommended)
- Node.js (optional, for tooling)

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/63v99.git
cd Verit
```

### 2. Start Neo4j

```bash
docker-compose up -d
```

Neo4j browser will be available at `http://localhost:7474`
Login: `neo4j` / `password123`

### 3. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 4. Pull your first cases

```bash
python data/fetch_cases.py
```

### 5. Run the API

```bash
uvicorn api.main:app --reload
```

---

## Restarting Project

### 1. Navigate to the Verit folder

```bash
cd \2026\01_Spring\MIS6V99\Verit
```

### 2. Activate Virtual Environment

```bash
.venv/Scripts/activate
```

You should see (.venv) at the start of your terminal prompt. If you don't see it, nothing else will work correctly.

### 3. Start Docker Desktop

Open Docker Desktop from your Start menu and wait until the whale icon in the taskbar shows "Docker Desktop is running." This is required before Neo4j can start.

### 4. Start Neo4j

```bash
docker-compose up -d
```

### 5. Verify It is Running

```bash
docker ps
docker logs verit_neo4j --tail 3   # confirm "Started."
```

### 6. Start the FastAPI Backend

Open a terminal and keep it running:

```powershell
uvicorn api.main:app --reload
```

Wait until you see `Uvicorn running on http://127.0.0.1:8000` before proceeding.

### 7. Start the Streamlit Frontend

Open a **second terminal** (FastAPI must stay running in the first):

```powershell
streamlit run frontend/app.py
```

The app will open automatically at `http://localhost:8501`.

> Both terminals must stay open simultaneously. FastAPI handles detection;
> Streamlit is the UI. Closing either will break the other.

---

## API Usage

```bash
POST /check-citation
Content-Type: application/json

{
  "citing_text": "Under the reasonable expectation of privacy standard established in Katz, the warrantless search of the defendant's vehicle was unconstitutional...",
  "citation": "United States v. Harlow, 444 F.3d 1255 (9th Cir. 2006)"
}
```

```json
{
  "verdict": "hallucinated",
  "confidence": 0.88,
  "checks": {
    "exists_in_graph": false,
    "semantic_similarity": 0.38,
    "graph_connected": false
  },
  "evidence": "Case not found in Fourth Amendment federal case corpus."
}
```

---

## Evaluation

Benchmark: 500 citations, 50% real / 50% hallucinated, split 80/20 val/test with stratified
sampling. Thresholds tuned on the 400-entry validation set; reported metrics are on the
held-out 100-entry test set and 10-fold cross-validation over the full 500.

### Test Set Results (held-out 100 entries)

| Layer                  | Precision | Recall    | F1        |
| ---------------------- | --------- | --------- | --------- |
| Layer 1 — Existence    | 1.000     | 0.580     | 0.734     |
| Layer 2 — Semantic     | 0.000     | 0.000     | 0.000     |
| Layer 3 — Connectivity | 0.000     | 0.000     | 0.000     |
| Layer 4 — Metadata     | 1.000     | 0.952     | 0.976     |
| **Combined**           | **1.000** | **0.980** | **0.990** |

Subtype F1: A=1.000, B=0.976, C=1.000. Zero false positives. One FN (benchmark_id=171): Type B year-corrupted citation where the Neo4j node stores the same incorrect year the benchmark injected — corpus data quality issue, undetectable by Layer 4.

### 10-Fold Cross-Validation (500 entries)

| Layer                  | Precision         | Recall            | F1                |
| ---------------------- | ----------------- | ----------------- | ----------------- |
| Layer 1 — Existence    | 1.000 ± 0.000     | 0.584 ± 0.020     | 0.737 ± 0.016     |
| Layer 2 — Semantic     | 0.000 ± 0.000     | 0.000 ± 0.000     | 0.000 ± 0.000     |
| Layer 3 — Connectivity | 0.000 ± 0.000     | 0.000 ± 0.000     | 0.000 ± 0.000     |
| Layer 4 — Metadata     | 1.000 ± 0.000     | 0.874 ± 0.106     | 0.929 ± 0.062     |
| **Combined**           | **1.000 ± 0.000** | **0.944 ± 0.045** | **0.971 ± 0.024** |

Fold F1 range: 0.936 – 1.000. No anomalous folds. Zero false positives across all folds.

---

## Development Timeline

| Week | Dates           | Milestone                                             | Date Completed |
| ---- | --------------- | ----------------------------------------------------- | -------------- |
| 1    | Feb 24 – Mar 2  | Environment setup, first 500 cases from CourtListener | 3-2-26         |
| 2    | Mar 3 – Mar 9   | Full data ingestion, EyeCite parsing, edge list       | 3-4-26         |
| 3    | Mar 10 – Mar 16 | Neo4j graph build and verification                    | 3-16-26        |
| 4    | Mar 17 – Mar 23 | BERT embedding pipeline + Milvus indexing             | 3-23-26        |
| 5    | Mar 24 – Mar 30 | Hybrid search (BM25 + HNSW via RRF)                   | 3-30-26        |
| 6    | Mar 31 – Apr 6  | Hallucination detector — all four layers + FastAPI    | 4-6-26         |
| 7    | Apr 7 – Apr 13  | Benchmark dataset construction + Streamlit scaffold   | 4-9-26         |
| 8    | Apr 14 – Apr 20 | Evaluation, threshold tuning, Layer 4, CV             | 4-10-26        |
| 9    | Apr 21 – Apr 27 | Error analysis + citation graph visualization         | 4-11-26        |
| 10   | Apr 28 – May 8  | Final writeup and submission                          |                |

---

## Domain Scope

This project focuses on **Fourth Amendment (Search & Seizure) federal case law** — a deliberate choice driven by three factors:

1. Fourth Amendment doctrine has the richest and most interconnected citation network in all of federal case law, making the Neo4j graph analysis unusually powerful and visually compelling
2. A small set of landmark anchor cases — _Terry v. Ohio_, _Katz v. United States_, _Mapp v. Ohio_, _United States v. Leon_ — are cited by thousands of downstream opinions, creating clear hub nodes that make graph traversal meaningful and interpretable
3. LLMs are known to confuse Fourth Amendment standards across contexts — mixing up the reasonable expectation of privacy test with the warrant requirement, or misattributing the good faith exception — making this an ideal domain for hallucination detection

Cases are pulled across all federal circuits to capture the full citation network, with the 9th Circuit serving as the primary analytical anchor.

---

## Notes

- Added a parquet layer of ingestion to aid in computational size restrictions.
- Added a test directory to aid in troubleshooting/debugging. This also uncovered that we're not pulling in all of the plain text opions. Which leads me to...
- After running the test enriched cases full text test, we realized that more than half (274) of the opinions pulled did not have full text, but rather a download url to pull PDFs. The two options given were either incorporate a download/extract function in the pipeline or drop them completely and pull in newer cases that were more likely to be digitized. We went with the latter.
- After reworking the import to include batches of after 2015, between 2010 and 2015 and before 2010, I realized that limiting the search params to just the 9th circuit was not good enough, so we expanded the query and dropped the court param.

## Change Log

A week-by-week record of implementation decisions, design choices, and technical
tradeoffs made during the build. Intended as a reference for the final writeup.

---

### Week 1 — Feb 24 – Mar 2

#### Environment Setup, Docker, Neo4j, First Cases

**Completed**

- Python 3.10 virtual environment configured
- Docker Desktop installed; Neo4j 5.15 container (`verit_neo4j`) running via Docker Compose
- CourtListener API access verified with token authentication
- Initial case fetch: first batch of Fourth Amendment cases retrieved
- `config.py` established as single source of truth for all paths and credentials

**Decisions**

- **Neo4j over a relational DB** — citation relationships are inherently a graph problem.
  Traversal queries (k-hops, shared citations) are natural in Cypher and would require
  complex self-joins in SQL.
- **Docker Compose for Neo4j** — keeps the database portable and reproducible without
  a local Neo4j installation. Bolt on 7687, browser on 7474.
- **Parquet for processed data, JSON for raw** — JSON for API responses and debugging
  (human-readable, easy to inspect); Parquet for all pipeline consumption (columnar,
  fast reads, 10x smaller than JSON at this scale).

---

### Week 2 — Mar 3 – Mar 9

#### Full Data Ingestion, Parquet Pipeline

**Completed**

- Full corpus fetch: 2,000 raw cases across two batches (post-2015 and 2010-2015)
- Opinion text enrichment via `fetch_all_opinions.py` — retrieved full `plain_text`
  for cases where available
- Batch merge and deduplication via `merge_batches.py` — 1,353 unique cases
- Converted to `cases_enriched.parquet` (20.23 MB) via `convert_to_parquet.py`
- 13 passing data validation tests in `tests/test_data.py`

**Decisions**

- **Multi-circuit corpus (removed ca9-only filter)** — initial design targeted 9th
  Circuit only, but plain text coverage was too sparse (~12% for pre-2015 cases).
  Expanding to all federal circuits gave enough cases with usable opinion text.
- **Post-2015 focus with 2010 floor** — pre-2015 cases have poor CourtListener plain
  text coverage (~12% vs ~87% post-2015). Including 2010-2015 adds breadth while
  keeping coverage acceptable. Cases before 2010 dropped entirely.
- **Deduplication on case_id** — CourtListener returns overlapping results across
  paginated batches. Deduplication on `case_id` (not `cluster_id`) produced 1,353
  unique cases with no duplicate opinion text.
- **`opinions_cited` URLs instead of EyeCite for graph building** — EyeCite parses
  citation strings from raw text (e.g., `392 U.S. 1`) but CourtListener's API already
  returns structured `opinions_cited` URLs pointing directly to cited opinion IDs.
  Using the API field for graph construction avoids regex parsing errors on OCR'd text
  and gives cleaner, more reliable citation data.
  EyeCite's role is scoped to two later stages: (1) **Week 6** — extracting citation
  strings from raw AI-generated text at query time before passing to the three-layer
  detector, and (2) **Week 7** — extracting real citation strings from corpus
  `plain_text` to use as the "real" half of the benchmark dataset.

---

### Week 3 — Mar 10 – Mar 16

#### Neo4j Graph Build and Verification

**Completed**

- `db/verify_landmarks.py` — verified CourtListener opinion IDs for all 5 landmark cases
- `db/graph_loader.py` — loaded full corpus into Neo4j as directed citation graph
  (1,353 full nodes, 14,773 stub nodes, 30,806 CITES edges); batched writes (500/batch)
  for performance; fully idempotent via MERGE
- `db/fetch_landmarks.py` — fetched 5 landmark SCOTUS cases directly from CourtListener
  and loaded as full nodes with `landmark: true`
- 14 passing graph validation tests in `tests/test_db.py`

**Decisions**

- **Stub nodes for out-of-corpus citations** — cases cited by the corpus but not present
  in the 2010-2025 dataset are stored as minimal stub nodes (`stub: true`, id only).
  This preserves the citation edge rather than discarding it. Stubs are needed for
  Layer 3 citation density scoring — a citation to a real-but-out-of-corpus case is
  still a valid graph signal.
- **MERGE everywhere (idempotent writes)** — all Neo4j writes use MERGE instead of
  CREATE. The graph loader can be re-run safely after crashes or config changes without
  duplicating nodes or edges.
- **Batched writes (500/batch via UNWIND)** — initial implementation wrote one
  transaction per citation edge (~3 cases/second). Switching to batched UNWIND
  transactions reduced load time from ~15 minutes to under 2 minutes.
- **Bulk insert then build HNSW index** — discovered during graph loader work that
  Neo4j (and Milvus) index quality degrades when built incrementally. Established
  pattern: insert all data first, then build indexes.
- **Layer 3 redesign — citation density (Option B)** — original design used k-hop
  paths from corpus cases to canonical landmark cases (Terry, Katz, Mapp, Leon, Gates)
  as the connectivity signal. Investigation revealed the corpus does not cite these
  landmarks by their CourtListener opinion IDs — the landmark nodes are isolated in
  the graph with zero incoming edges from the corpus.
  Two options considered:
  - **Option A:** expand landmark set to cases actually cited by corpus (Coolidge,
    Ornelas, etc.), enrich stubs to build multi-hop paths back to SCOTUS landmarks.
    Preserves k-hop visualization but requires 1,000-5,000 additional API calls and
    anchors are chosen empirically rather than doctrinally.
  - **Option B:** replace landmark connectivity with citation density — measure how
    many citation targets a cited case shares with known corpus cases. Zero additional
    API calls. A hallucinated case has no footprint in the citation network regardless
    of landmark proximity. Loses k-hop visualization but gains a more robust signal
    and a citation density heatmap as an alternative visualization.
    **Option B chosen** — cleaner implementation, no additional data fetching, defensible
    signal, and compatible with the remaining timeline.
- **Neo4j browser inaccessible on Windows/Docker** — browser at `localhost:7474`
  failed to authenticate despite correct credentials. Root cause: WebSocket connection
  issues between Neo4j Desktop/browser and a Docker-hosted instance on Windows.
  Resolution: use `cypher-shell` via `docker exec` for all graph queries. Python Bolt
  connection (`bolt://localhost:7687`) unaffected throughout.
- **`test_landmarks_are_reachable` skipped** — this test asserts at least one landmark
  is reachable via a direct CITES edge from the corpus. With Option B, landmark
  reachability is no longer a requirement. Test kept in suite but marked
  `@pytest.mark.skip` with documented rationale.

### Week 4 — Mar 17 – Mar 23

#### Text Cleaning + BERT Embedding Pipeline + Milvus Indexing

##### Completed

- `preprocessing/clean_text.py` — stripped court headers/footers, normalized citation strings to [CITATION] token, fixed encoding artifacts from PDF-converted opinions. Output: `data/processed/cases_cleaned.parquet`
- `embeddings/prune_vectors.py` — filtered corpus to cases with usable plain text (min 200 chars, max 50,000 chars). Output: `data/processed/cases_pruned.parquet`
- `embeddings/embed_cases.py` — paragraph chunking (512-token ceiling, 1-paragraph overlap), legal-bert inference, mean-pool + L2-normalize → 768-dim vector per case. Output: `data/processed/embeddings.parquet
embeddings/milvus_index.py` — bulk inserted 1,353 normalized vectors, built HNSW index (M=16, ef_construction=200).
- Milvus standalone stack added to Docker Compose (etcd, minio, milvus v2.4.9)
- Dependencies frozen to requirements.txt

##### Decisions

- Milvus Docker standalone over Milvus Lite — milvus-lite has no Windows distribution. Switched to full Milvus standalone via Docker Compose with etcd and minio backing services. Connection updated from local `.db` file path to `http://localhost:19530`. No changes to index logic or collection schema required.
- `torch` **import must precede numpy/pandas on Windows** — encountered WinError 1114 (DLL initialization failure on c10.dll) when numpy or pandas was imported before torch.
  - Root cause: conflicting DLL load order on Windows.
  - Fixed by moving import torch to the top of the imports block in embed_cases.py.
  - Applied same ordering defensively to any future scripts that use both.
- `MAX_TEXT_LENGTH` truncation applied before chunking — initial runs showed ETA of 180+ minutes because `MAX_TEXT_LENGTH` was defined in config but not enforced in the embedding loop. Added text = (row["plain_text"] or "")[:MAX_TEXT_LENGTH] before chunking. Corpus mean text length is 27,684 chars; truncation at 50,000 affects only the longest opinions and keeps chunks-per-case to a manageable range.
- 50,000 char truncation preserves embedding quality — Fourth Amendment opinions front-load doctrinal reasoning. Tail content is typically procedural orders, attorney fee discussions, and appendices. Truncating at 50,000 chars keeps embeddings anchored to substantive legal reasoning and avoids mean-pool dilution from boilerplate.
- `marshmallow` pinned to 3.x — pymilvus 2.4.9 depends on environs, which depends on marshmallow. Installing marshmallow 4.x (the current release) breaks environs with AttributeError: module 'marshmallow' has no attribute '**version_info**'. Pinned to marshmallow==3.23.2 via --force-reinstall.
- setuptools downgraded to 69.x — setuptools 82+ removed pkg_resources, which pymilvus 2.4.9 depends on at import time. Downgraded to setuptools==69.5.1 to restore pkg_resources availability.
- `EMBED_BATCH_SIZE` set to 32 — default of 16 was conservative. CPU inference on legal-bert handles 32-chunk batches without memory issues and reduces per-case overhead.

### Week 5 — Completed (Mar 18, 2026)

#### Scripts Built

- `preprocessing/tokenize_bm25.py` — spaCy lemmatization, legal term preservation,
  outputs `data/processed/cases_tokenized.parquet` (case_id | tokens)
- `embeddings/bm25_index.py` — BM25Okapi index over 1,353 tokenized cases,
  outputs `data/processed/bm25_index.pkl` (5.7 MB)
- `detector/semantic_check.py` — hybrid ANN (Milvus HNSW) + BM25 fused via RRF,
  returns SemanticResult(rrf_score, is_relevant, top_matches)
- `detector/cache.py` — TTLCache for query embeddings and ANN results
  (maxsize=512, ttl=3600)

#### Config Changes

- `MILVUS_DB_PATH` removed, replaced with `MILVUS_URI = "http://localhost:19530"`

#### Infrastructure

- Milvus running in Docker standalone (port 19530), not Milvus Lite
- `verit_milvus` container confirmed running with 1,353 vectors loaded
- spaCy `en_core_web_sm` model installed
- PyTorch reinstalled as CPU-only wheel to resolve Windows DLL initialization error

#### Dependencies Added

- `rank-bm25`
- `spacy`
- `torch` (CPU-only, reinstalled via `https://download.pytorch.org/whl/cpu`)

#### Known Observations

- RRF scores cluster in a narrow band (~0.023–0.025) by design — this is a
  structural property of RRF with TOP_K=5 and RRF_K=60, not a bug
- Layer 2 alone cannot separate real from hallucinated at query time — separation
  comes from Layers 1 and 3 working together; Layer 2's role is retrieval quality
- First sample in `cases_tokenized.parquet` shows residual court header tokens
  (`supreme court state north dakota`) — clean_text.py did not fully strip this
  case's header; low impact on BM25 quality due to IDF weighting, note for
  Week 8 error analysis
- `RRF_THRESHOLD = 0.02` set as placeholder in semantic_check.py — tune on
  validation set in Week 8

### Week 6 — Hallucination Detector + FastAPI Endpoint

_Mar 17 – Mar 23, 2026_

#### Added

- `detector/eyecite_parser.py` — extracts full case citations from raw text via EyeCite; resolves each to a CourtListener cluster ID via `POST /citation-lookup/`; returns `ResolvedCitation` objects with citation string, case name, case ID, and surrounding context window
- `detector/existence_check.py` — Layer 1: Neo4j node lookup by cluster ID; short-circuits to HALLUCINATED if case not found
- `detector/cache.py` — TTLCache for query embeddings and ANN results; keyed on text hash; avoids redundant legal-bert inference during benchmark evaluation loop
- `detector/semantic_check.py` — Layer 2: hybrid ANN + BM25 search fused via Reciprocal Rank Fusion; returns RRF score, top dense cosine score, and top-k corpus matches
- `detector/connectivity_check.py` — Layer 3: citation density score via Neo4j; counts distinct corpus cases sharing outbound citations with the target case
- `detector/pipeline.py` — orchestrates EyeCite parser + all three layers; implements verdict logic (REAL / SUSPICIOUS / HALLUCINATED); returns `CitationVerdict` objects with full layer scores and top corpus matches for RAG
- `api/main.py` — FastAPI service exposing `POST /check-citation` and `GET /health`; wraps pipeline with Pydantic request/response models

#### Fixed

- EyeCite field access updated to use `citation.groups` directly (was `citation.token.volume/reporter/page` which does not exist in installed version)
- CourtListener resolution switched from `GET /search/` (BM25 text search, wrong results) to `POST /citation-lookup/` (exact citation match); cluster ID used directly as Neo4j node key, eliminating redundant opinion lookup call

#### Notes

- Landmark cases (Terry, Katz, Mapp, Leon, Gates) correctly return `density_score: 0` in Layer 3 — landmark isolation from corpus citation network is by design
- `SemanticResult` exposes both `rrf_score` and `top_dense_score`; pipeline logs both for threshold tuning in Week 8
- RRF threshold (`RRF_THRESHOLD = 0.02`) is a placeholder; full threshold tuning deferred to Week 8

### Week 7 — Benchmark Dataset + Streamlit Scaffold

#### New Scripts

- `benchmark/generate_benchmark.py` — builds balanced 200-citation benchmark dataset
  with stratified real citation sampling via EyeCite and three hallucinated subtypes
  generated via Claude API and corpus mutation
- `frontend/app.py` — Streamlit UI scaffold wired to FastAPI `/check-citation` endpoint;
  displays verdict badges, semantic scores, density scores, and expandable top corpus
  matches per citation

#### New Files

- `benchmark/benchmark.json` — 200-citation evaluation dataset (100 real / 100
  hallucinated); hallucinated split into Type A (fabricated), Type B (corrupted real),
  Type C (plausible nonexistent)

#### Dependencies Added

- `anthropic` — Anthropic Python SDK for Claude API calls (added one week early;
  needed for benchmark Type A/C generation and Week 9 LLM layer)
- `streamlit` — frontend framework

#### Config Changes

- `BENCHMARK_DIR` added to `config.py`
- `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` added to `config.py`, `.env`, `.env.example`

#### Notes

- Type B corruption embeds year shift or court swap directly into the citation string
  so the detector sees the corrupted version, not the original
- Real citations stratified across court_id and year; state courts capped at 40% of
  real sample to avoid corpus skew
- Reporter format filter applied to Type B candidates — neutral citations (WestLaw IDs,
  Ohio slip opinions) excluded in favor of volume-reporter-page formatted strings

### Week 8 — Evaluation + Threshold Tuning

#### Goals

Evaluate the hallucination detector against the benchmark dataset built in Week 7,
tune detection thresholds on a held-out validation set, and report final metrics
on a separate test set.

#### Benchmark Split

`benchmark.json` (200 entries, 50% real / 50% hallucinated) was split 80/20 into
a validation set (160 entries) and a held-out test set (40 entries) using stratified
sampling. Stratification preserves both the real/hallucinated balance and the
distribution of hallucination subtypes (A, B, C) across both splits. The split is
cached to `benchmark/split_indices.json` on first run and never reshuffled — this
ensures the test set remains truly held-out across all subsequent runs.

#### New Scripts

| Script                       | Purpose                                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `benchmark/evaluate.py`      | Runs inference on val set, sweeps thresholds, saves best combo to `tuned_thresholds.json` |
| `benchmark/report.py`        | Loads tuned thresholds, runs inference on test set, writes `eval_report.json`             |
| `detector/metadata_check.py` | Layer 4 — validates year and court in citation string against Neo4j node properties       |
| `db/backfill_court_id.py`    | One-time migration: backfills `court_id` from `cases_enriched.parquet` into Neo4j         |

#### Layer 4 — Metadata Validation

The first evaluation run exposed a gap in the original three-layer architecture:
all 7 subtype B hallucinations (real case, corrupted metadata) passed Layers 1–3
with full confidence. This was expected — the underlying case exists in Neo4j,
the citation context is semantically valid Fourth Amendment text, and the citation
network footprint belongs to the real case. No signal available to Layers 1–3
distinguished these from genuine citations.

**Layer 4** was added to close this gap. It extracts the court identifier and year
from the citation string and compares them against the actual properties stored on
the Neo4j Case node. A mismatch flags the citation as HALLUCINATED.

Court extraction uses two strategies in order:

1. **Direct CourtListener ID match** — catches injected court codes in reporter
   citation parentheticals, e.g. `"476 U.S. 207 (ca11)"` where `ca11` appears
   as a bare identifier in the trailing parenthetical.
2. **Alias match** — catches natural-language court strings in formatted citations,
   e.g. `"923 F.3d 1027 (4th Cir. 2019)"` → `ca4`.

Layer 4 is skipped (returns `is_valid=True`) when no year or court can be extracted
from the citation string — pure reporter citations like `"392 U.S. 1"` with no
parenthetical metadata cannot be validated and are not penalized.

**Why `court_id` needed backfilling:** The Neo4j graph was built in Week 3 from
`cases_enriched.parquet` but `court_id` was not included as a node property at
that time. `db/backfill_court_id.py` reads the `court_id` column from the parquet
(zero nulls across 1,353 cases) and writes it to all matching Case nodes in batches.
Without this, Layer 4's court comparison always returned `None` and fell through
to `is_valid=True`, making it a no-op.

#### Threshold Tuning

`evaluate.py` sweeps 180 combinations (6 × SIM, 6 × RRF, 5 × DENSITY) on the
validation set, selecting the combination with the highest F1 (ties broken by
precision — fewer false alarms preferred in a legal context). The sweep landed
at minimum values across all three parameters, reflecting that Layers 2 and 3
contribute no independent signal on subtype B hallucinations — those are fully
handled by Layer 4 — and that no false positives were observed at any threshold
level on this benchmark.

**Tuned thresholds:**

| Parameter                | Config Key                   | Tuned Value |
| ------------------------ | ---------------------------- | ----------- |
| Cosine similarity floor  | `SIMILARITY_THRESHOLD`       | 0.60        |
| RRF score floor          | `RRF_THRESHOLD`              | 0.010       |
| Citation density minimum | `CITATION_DENSITY_THRESHOLD` | 1           |

#### Results — Test Set (40 entries)

| Layer                  | Precision | Recall    | F1        | Notes                                      |
| ---------------------- | --------- | --------- | --------- | ------------------------------------------ |
| Layer 1 — Existence    | 1.000     | 0.650     | 0.788     | Catches Type A + C immediately             |
| Layer 2 — Semantic     | 0.000     | 0.000     | 0.000     | No independent signal on Type B            |
| Layer 3 — Connectivity | 0.000     | 0.000     | 0.000     | No independent signal on Type B            |
| Layer 4 — Metadata     | 1.000     | 1.000     | 1.000     | Catches all 7 Type B cases                 |
| **Combined**           | **1.000** | **1.000** | **1.000** | Zero false positives, zero false negatives |

**Subtype breakdown:**

| Subtype                       | F1    | Description                             |
| ----------------------------- | ----- | --------------------------------------- |
| A — Fully fabricated          | 1.000 | Caught by Layer 1 (case does not exist) |
| B — Real case, wrong metadata | 1.000 | Caught by Layer 4 (court ID mismatch)   |
| C — Plausible but nonexistent | 1.000 | Caught by Layer 1 (case does not exist) |

#### Honest Assessment of the 1.0 Result

The perfect test-set scores are real but should be interpreted carefully:

- **Small test set.** 40 entries is a limited sample. Perfect scores are more
  achievable at this scale than they would be on a larger, harder benchmark.

- **Benchmark scope.** The hallucinated citations were constructed to be detectable
  by the current architecture. Type A and C are fabricated, so Layer 1 catches them
  by definition. Type B corruptions inject a court ID directly into the citation
  string — a pattern Layer 4 was specifically built to catch after observing the
  initial false negatives.

- **Layers 2 and 3 show F1=0.0 in isolation.** This is not a malfunction — it
  accurately reflects that every hallucinated case reaching those layers is a Type B,
  and Type B cases carry valid semantic scores and strong citation network footprints
  from the underlying real case. The isolated metrics honestly show what each layer
  contributes individually.

- **Out-of-scope hallucination types.** The most dangerous real-world hallucination —
  a real case cited for a proposition it does not support — is undetectable by any
  current layer. Detecting contextual misuse would require reading and reasoning about
  the full opinion text, not just checking existence and metadata. This is a known
  limitation documented as future work.

#### Week 8 (REDONE) — Evaluation, Threshold Tuning, and Benchmark Expansion

##### Overview

Week 8 focused on honest evaluation of the hallucination detector. The initial pipeline from Weeks 6–7 was run against the benchmark, results were analyzed, a fourth detection layer was added to close a gap exposed by the evaluation, the benchmark was expanded from 200 to 500 entries, and 10-fold cross-validation was run to verify that strong results were stable rather than lucky.

---

##### What Was Built

###### Layer 4 — Metadata Validation (`detector/metadata_check.py`)

The first evaluation run revealed that all subtype B hallucinations (real case, corrupted metadata) were passing Layers 1–3 with full confidence. This was expected — a Type B citation uses a real `case_id`, so it exists in Neo4j, produces a valid semantic score, and has a strong citation network footprint. Layers 1–3 have no signal to distinguish it from a genuine citation.

Layer 4 was added to close this gap. It extracts the court identifier and year from the citation string and compares them against the actual properties stored on the Neo4j Case node. A mismatch flags the citation as `HALLUCINATED`.

Court extraction uses two strategies in order:

1. **Direct CourtListener ID match** — detects bare court IDs injected into citation parentheticals, e.g. `"476 U.S. 207 (ca11)"` where `ca11` is extracted and compared against the node's `court_id`
2. **Alias match** — detects natural-language court strings in formatted citations, e.g. `"4th Cir."` → `ca4`

Layer 4 is skipped when no year or court can be extracted from the citation string. Pure reporter citations like `"392 U.S. 1"` cannot be validated and are not penalized.

###### Neo4j Court ID Backfill (`db/backfill_court_id.py`)

Layer 4 requires `court_id` to be present on Neo4j Case nodes, but the graph was built in Week 3 without that property. This one-time migration script reads `court_id` from `cases_enriched.parquet` — which has zero nulls across all 1,353 cases — and writes it to every matching Case node in batches of 200.

###### Evaluation Scripts

| Script                  | Purpose                                                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `benchmark/evaluate.py` | Loads benchmark, creates stratified 80/20 val/test split, sweeps 180 threshold combinations on val set, saves best to `tuned_thresholds.json` |
| `benchmark/report.py`   | Loads tuned thresholds, runs inference on held-out test set, writes `eval_report.json` with per-layer and combined metrics                    |

The val/test split is cached to `benchmark/split_indices.json` on first run and never reshuffled, ensuring the test set remains truly held-out across all subsequent runs.

###### Benchmark Expansion (`benchmark/expand_benchmark.py`)

The original 200-entry benchmark produced F1=1.0 on a 40-entry test set — a real result, but statistically weak at that scale. The benchmark was expanded to 500 entries to reduce variance and provide a more meaningful evaluation surface.

| Type                           | Original | Added   | Final   |
| ------------------------------ | -------- | ------- | ------- |
| Real                           | 100      | 150     | 250     |
| Type A — Fabricated            | 33       | 40      | 73      |
| Type B — Corrupted metadata    | 34       | 70      | 104     |
| Type C — Plausible nonexistent | 33       | 40      | 73      |
| **Total**                      | **200**  | **300** | **500** |

Type B was weighted more heavily in the expansion (70 new entries vs 40 for A and C) because it is the hardest subtype for the pipeline to detect. The expansion also shifts Type B corruptions to 60% court / 40% year (vs 50/50 in the original) to more aggressively stress Layer 4.

###### 10-Fold Cross-Validation (`benchmark/cross_validate.py`)

To verify that the strong test-set results were stable rather than a product of a favorable split, 10-fold stratified cross-validation was run on the full 500-entry benchmark. Stratification preserves the real/hallucinated balance and subtype distribution across all folds.

The script includes fold-level checkpointing — completed folds are saved after each fold, so the run is safe to interrupt and resume without restarting from fold 1.

---

##### Threshold Tuning

`evaluate.py` sweeps 180 threshold combinations on the 400-entry validation set. The combination with the highest F1 is selected, with ties broken by precision — fewer false alarms is preferred in a legal context.

| Parameter                | Config Key                   | Before | After |
| ------------------------ | ---------------------------- | ------ | ----- |
| Cosine similarity floor  | `SIMILARITY_THRESHOLD`       | 0.75   | 0.60  |
| RRF score floor          | `RRF_THRESHOLD`              | —      | 0.010 |
| Citation density minimum | `CITATION_DENSITY_THRESHOLD` | 3      | 1     |

The sweep landed at minimum values across all three parameters. This reflects that Layers 2 and 3 contribute no independent signal on Type B hallucinations — those are handled entirely by Layer 4 — and that no false positives were observed at any threshold level on this benchmark.

---

##### Results

###### Test Set (held-out 100 entries)

| Layer                  | Precision | Recall    | F1        |
| ---------------------- | --------- | --------- | --------- |
| Layer 1 — Existence    | 1.000     | 0.580     | 0.734     |
| Layer 2 — Semantic     | 0.000     | 0.000     | 0.000     |
| Layer 3 — Connectivity | 0.000     | 0.000     | 0.000     |
| Layer 4 — Metadata     | 1.000     | 0.952     | 0.976     |
| **Combined**           | **1.000** | **0.980** | **0.990** |

Subtype F1: A=1.000, B=0.976, C=1.000. Zero false positives. One FN (benchmark_id=171):
Type B year-corrupted citation where the Neo4j node (`United States v. Moses, ca3`)
stores `year=2025` — the same incorrect year the benchmark injected. Layer 4 compares
`2025 == 2025`, finds no mismatch, and passes it. Corpus data quality issue; undetectable
by the current architecture.

###### 10-Fold Cross-Validation (500 entries)

| Layer                  | Precision         | Recall            | F1                |
| ---------------------- | ----------------- | ----------------- | ----------------- |
| Layer 1 — Existence    | 1.000 ± 0.000     | 0.584 ± 0.020     | 0.737 ± 0.016     |
| Layer 2 — Semantic     | 0.000 ± 0.000     | 0.000 ± 0.000     | 0.000 ± 0.000     |
| Layer 3 — Connectivity | 0.000 ± 0.000     | 0.000 ± 0.000     | 0.000 ± 0.000     |
| Layer 4 — Metadata     | 1.000 ± 0.000     | 0.874 ± 0.106     | 0.929 ± 0.062     |
| **Combined**           | **1.000 ± 0.000** | **0.944 ± 0.045** | **0.971 ± 0.024** |

Fold F1 range: 0.936 – 1.000. No anomalous folds detected.

---

##### Interpreting the Results

###### Why the test-set F1 is 0.990

The test-set results are strong but not perfect. While Layers 1 and 3 perform as expected, Layer 4 missed one Type B hallucination (benchmark_id=171) because the underlying Neo4j node contained the same incorrect year (2025) as the injected corruption. This highlights that the detector is only as accurate as the underlying corpus data.

###### Why Layers 2 and 3 show F1=0.0 in isolation

This is not a failure. Every hallucinated citation that reaches Layers 2 and 3 is a Type B case, and Type B cases carry valid semantic scores and strong citation network footprints from the underlying real case. Layers 2 and 3 have no signal to distinguish them from genuine citations. The isolated metrics honestly reflect what each layer contributes on its own. In production, Layers 2 and 3 provide redundancy for edge cases outside this benchmark's scope — for instance, corpus cases that exist in Neo4j but were not indexed in Milvus, or citations where CourtListener resolution is ambiguous.

###### Why Layer 4 recall varies across folds (±0.106)

Some folds contain Type B entries that Layer 4 cannot catch. This happens when a year-corrupted
citation references a Neo4j node that stores the same (incorrect) year as the benchmark
corruption, or when the corrupted court ID happens to match the node's actual court. The
one test-set FN (benchmark_id=171) is a confirmed example of the former — the corpus
stores `year=2025` on that node, making the mismatch undetectable. This variance is the
most honest signal from the CV results and is the primary architectural limitation to
document in the final writeup.

###### The primary undetected hallucination type

The most dangerous real-world hallucination — a real case cited for a legal proposition it does not support — is undetectable by any current layer. Catching this would require the system to read and reason about the full opinion text, not just verify existence and metadata. This is the central limitation and the main direction for future work.

### Week 9 — Visualization & Frontend Integration

**Date:** April 11, 2026

#### Built This Week

- `benchmark/density_histogram.py` — new script that reads `per_entry` from
  `eval_report.json` and plots citation density distributions for REAL vs.
  HALLUCINATED labels; saved to `visualization/density_histogram.png`

#### Configuration Changes

- `config.py` — updated three thresholds:
  - `SIMILARITY_THRESHOLD` 0.75 → 0.60
  - `RRF_THRESHOLD` 0.02 → 0.010
  - `CITATION_DENSITY_THRESHOLD` 3 → 1

#### Outputs Generated

- `visualization/umap_circuit.html` — standalone UMAP colored by circuit; 1,353
  corpus cases; left-cluster grouping consistent with circuit-level semantic similarity
- `visualization/umap_year.html` — standalone UMAP colored by year
- `visualization/density_histogram.png` — density distribution across 50 REAL and
  21 HALLUCINATED entries (29 HALLUCINATED entries have `density_score=None`;
  these were caught before Layer 3 and never received a score)

#### Validated This Week

- Full app smoke test: FastAPI + Streamlit running end-to-end
- UMAP overlay confirmed working: REAL citations plot as stars inside corpus
  clusters, SUSPICIOUS citations plot as diamonds in distinct regions
- Haiku explanation streaming confirmed for REAL verdicts
- UMAP runtime: ~30s on first cache load; fit on 1,353 cases, output shape (1,353, 2)

#### Known Limitations Documented

- 29 of 50 HALLUCINATED benchmark entries have `density_score=None` — hard
  hallucinations (fabricated IDs) are caught before Layer 3 and excluded from
  the density histogram; histogram reflects Type B hallucinations only
- Terry v. Ohio returns `density_score=0` — landmark case is not in the Verit
  corpus, so Layer 3 has no graph footprint to evaluate against

### Week 10 — Citation Graph Tab & Final Visualizations

#### What Was Built

**`visualization/graph_viz.py`** — New module that queries Neo4j for the citation
subgraph of a submitted case and renders it as an interactive PyVis network.

- Pulls 1- or 2-hop `CITES` neighborhoods from Neo4j
- Node color: 🔴 submitted case · 🟡 landmark · 🔵 corpus · ⚫ stub
- Node size proportional to `cite_count`
- Hover tooltips: case name, year, court, citation count
- Returns HTML string for embedding in Streamlit

**`frontend/app.py`** — Added Tab 3: 🔗 Citation Graph

- Dropdown selects from REAL/SUSPICIOUS citations in the last check result
- 1-hop / 2-hop toggle
- PyVis graph rendered via `st.components.v1.html()`
- "Open in Neo4j Browser" button (`st.link_button`) pre-fills a Cypher query
  at `localhost:7474` for deeper exploration

#### Bug Fixes This Week

- **Cypher parameter map syntax error** — Neo4j does not accept `$param` inside
  `MATCH` pattern maps or relationship length syntax (`*1..$hops`). Fixed by
  switching to an f-string with literals baked in directly.
- **Stub filter removing all results** — The `WHERE neighbor.stub = false` clause
  was filtering out the entire 1-hop neighborhood for cases whose citations are
  all stub nodes. Removed the filter; stubs now render as gray nodes, which is
  informative rather than invisible.

#### Visualizations Complete

| Visualization                   | File                                  | Status |
| ------------------------------- | ------------------------------------- | ------ |
| UMAP by circuit                 | `visualization/umap_circuit.html`     | ✅     |
| UMAP by year                    | `visualization/umap_year.html`        | ✅     |
| UMAP with hallucination overlay | Screenshot — Corpus Map tab           | ✅     |
| Citation density histogram      | `visualization/density_histogram.png` | ✅     |
| Citation Graph (PyVis)          | Screenshot — Citation Graph tab       | ✅     |

#### Install

```powershell
pip install pyvis
```

Add to `requirements.txt`:

```
pyvis==0.3.2
```

---

### API Security

#### Current State (Week 6)

- **Input size limit:** requests exceeding 50,000 characters are rejected with HTTP 400.
  Prevents runaway legal-bert inference on oversized payloads.
- **CORS:** restricted to `http://localhost:8501` (Streamlit default port).
- **No authentication:** single-user local deployment, no API keys required.
- **No rate limiting:** not needed for local use.

### Design Decisions Pending Implementation

The following decisions were made during Weeks 1-3 planning and will be implemented
in later weeks. Recorded here for traceability.

**EyeCite scoped to query time and benchmark (Weeks 6-7)** — EyeCite was originally
listed as a Week 2 deliverable but was correctly deferred. Its role is: (1) at query
time in Week 6, parse raw AI-generated text to extract citation strings and resolve
them to CourtListener opinion IDs before the three-layer pipeline runs; (2) in Week 7,
extract real citation strings from corpus `plain_text` for the benchmark's real
citation sample. EyeCite is not used for graph construction — that uses CourtListener's
structured `opinions_cited` API field.

**Hybrid search (Week 5)** — Layer 2 will use Reciprocal Rank Fusion (RRF) to combine
dense HNSW vector search (Milvus) with sparse BM25 keyword search. RRF chosen over
weighted sum because it requires no weight tuning and is robust by default. BM25
improves candidate retrieval quality — it does not detect hallucinations directly.

**Vector pruning (Week 4)** — cases with `plain_text` under 200 characters or null
will be excluded from Milvus indexing. They remain in Neo4j as graph nodes. Short
opinions are typically procedural orders with no substantive Fourth Amendment content
and would produce noisy embeddings that cluster artificially.

**L2 normalization (Week 4)** — all embeddings will be L2-normalized before storage
in Milvus so cosine similarity is equivalent to dot product. Applied after mean-pooling,
before parquet save.

**Text cleaning before embedding (Week 4)** — court headers, attorney lists, and raw
citation strings will be stripped/normalized before passing text to legal-bert. Raw
citations replaced with `[CITATION]` token to reduce token budget waste and embedding
noise.

**UMAP over PCA for visualization (Week 9)** — UMAP preserves local neighborhood
structure in the embedding space; PCA collapses legal text embeddings into a dense
blob. StandardScaler applied before UMAP to prevent high-variance dimensions from
dominating the 2D projection.

UMAP was selected over PCA because Legal-BERT embeddings lie on a non-linear manifold — linear dimensionality reduction would not faithfully represent the semantic clustering structure. UMAP preserves local neighborhood topology (n_neighbors=15) while allowing global structure to compress, making it appropriate for visualizing semantic similarity clusters in the embedding space.

**Balanced benchmark with 3 hallucination subtypes (Week 7)** — 50/50 real vs
hallucinated, hallucinated split into: Type A (fully fabricated), Type B (real case,
corrupted details), Type C (plausible but nonexistent). Three subtypes test different
failure modes of the detector.

**All thresholds tuned on validation set (Week 8)** — cosine similarity threshold,
RRF threshold, and citation density threshold will all be tuned on a held-out validation
split, never on the test set.

## Acknowledgements

- [CourtListener](https://www.courtlistener.com) — Free Law Project
- [EyeCite](https://github.com/freelawproject/eyecite) — Free Law Project
- [CLERC Dataset](https://huggingface.co/datasets/jhu-clsp/CLERC) — Johns Hopkins CLSP
- [CaseHOLD](https://huggingface.co/datasets/casehold/casehold) — AI2

---

## License

MIT License — see `LICENSE` for details.

---

_Built as a solo academic project. Due May 8._
