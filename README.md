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

Verit runs every citation through a three-layer verification pipeline:

```
AI-Generated Legal Text
        │
        ▼
 EyeCite Extraction  →  Parse all citation strings from text
        │
        ▼
 Check 1: Neo4j      →  Does this case exist as a node in the graph?
        │
        ▼
 Check 2: Milvus     →  Is it semantically relevant? (cosine similarity)
        │
        ▼
 Check 3: Neo4j      →  Is it graph-connected to the legal topic?
        │
        ▼
 Verdict + Confidence Score
 { "verdict": "valid" | "real_irrelevant" | "hallucinated", "confidence": 0.0–1.0 }
```

Each check catches what the others miss. A citation can pass the existence check and still be semantically irrelevant. A citation can score high on semantic similarity and still be disconnected from the citing document's legal doctrine in the real citation network. All three checks together produce a meaningfully more accurate verdict than any single check alone.

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
├── docker-compose.yml          # Neo4j container config
├── requirements.txt
├── README.md
├── tests/
│   ├── __init__.py
│   ├── test_data.py        ← replaces sanity_check.py and sanity_check2.py
│   ├── test_db.py          ← will hold Neo4j and Milvus connection tests (Week 3)
│   ├── test_detector.py    ← will hold hallucination detector tests (Week 6)
│   └── conftest.py         ← shared test fixtures
│
├── data/
│   ├── fetch_cases.py              # CourtListener API ingestion (4th Amendment filter)
│   ├── fetch_all_opinions.py       # Fetches full opinion text for each case
│   ├── hf_dataset_loader.py        # Load CLERC + CaseHOLD
│   ├── convert_to_parquet.py       # Converts enriched JSON → Parquet for downstream use
│   ├── raw/                        # Raw JSON from CourtListener (not pushed to GitHub)
│   │   ├── cases_sample.json       # Initial 500 case metadata
│   │   └── cases_enriched.json     # Full text + citation URLs added
│   └── processed/                  # Parquet files for pipeline consumption
│       └── cases_enriched.parquet  # Final cleaned dataset used by all downstream scripts
│
├── preprocessing/
│   ├── text_cleaner.py         # HTML stripping, chunking, dedup
│   └── eyecite_parser.py       # Citation extraction + edge list
│
├── db/
│   ├── neo4j_client.py         # Graph node/edge creation
│   └── milvus_client.py        # Vector collection management
│
├── embeddings/
│   └── embed_pipeline.py       # Batch BERT embedding + Milvus indexing
│
├── detector/
│   ├── existence_check.py      # Neo4j node lookup
│   ├── relevance_check.py      # Milvus cosine similarity
│   └── connectivity_check.py   # Neo4j graph traversal
│
├── api/
│   └── main.py                 # FastAPI /check-citation endpoint
│
└── benchmark/
    ├── generate_hallucinations.py
    ├── evaluate.py             # Precision, recall, F1 per check
    └── results/
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
```

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

The benchmark compares three detection strategies across a labeled dataset of real and hallucinated citations:

| Strategy                          | Precision | Recall | F1  |
| --------------------------------- | --------- | ------ | --- |
| Existence check only (Neo4j)      | TBD       | TBD    | TBD |
| Semantic similarity only (Milvus) | TBD       | TBD    | TBD |
| Graph connectivity only (Neo4j)   | TBD       | TBD    | TBD |
| **All three combined**            | TBD       | TBD    | TBD |

Results will be updated as evaluation runs are completed.

---

## Development Timeline

| Week | Dates           | Milestone                                             | Date Completed |
| ---- | --------------- | ----------------------------------------------------- | -------------- |
| 1    | Feb 24 – Mar 2  | Environment setup, first 500 cases from CourtListener | 3-2-26         |
| 2    | Mar 3 – Mar 9   | Full data ingestion, EyeCite parsing, edge list       | 3-4-26         |
| 3    | Mar 10 – Mar 16 | Neo4j graph build and verification                    |
| 4    | Mar 17 – Mar 23 | BERT embedding pipeline + Milvus indexing             |
| 5    | Mar 24 – Mar 30 | Semantic similarity retrieval layer                   |
| 6    | Mar 31 – Apr 6  | Hallucination detector — all three checks             |
| 7    | Apr 7 – Apr 13  | Benchmark dataset construction                        |
| 8    | Apr 14 – Apr 20 | Evaluation — precision, recall, F1, tradeoff curves   |
| 9    | Apr 21 – Apr 27 | Error analysis + citation graph visualization         |
| 10   | Apr 28 – May 8  | Final writeup and submission                          |

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

---

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
