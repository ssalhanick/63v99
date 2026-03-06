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
2. **Semantic relevance check** — is it relevant to the citing text? (Milvus cosine similarity)
3. **Graph connectivity check** — is it connected to the legal topic? (Neo4j traversal)

---

## Tech Stack

| Component           | Tool                    | Purpose                                        |
| ------------------- | ----------------------- | ---------------------------------------------- |
| Vector Store        | Milvus Lite             | Store and search 768-dim case embeddings       |
| Graph Database      | Neo4j 5.15 (Docker)     | Store citation relationships as directed graph |
| Embedding Model     | legal-bert-base-uncased | Convert legal text to semantic vectors         |
| Citation Extraction | EyeCite                 | Parse citation strings from raw text           |
| API Layer           | FastAPI                 | Expose /check-citation endpoint                |
| Infrastructure      | Docker + Docker Compose | Run Neo4j locally                              |
| Language            | Python 3.10             |                                                |
| IDE                 | VS Code                 |                                                |

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
- **Credentials:** neo4j / Verit2025!! (stored in .env)
- **Browser:** http://localhost:7474
- **Start:** `docker-compose up -d`
- **Status:** ✅ Connection verified

### Landmark Anchor Cases (need real CourtListener opinion IDs)

These need to be verified and updated in `config.py` before running graph_loader.py:

- Terry v. Ohio (1968) — estimated ID: 107729
- Katz v. United States (1967) — estimated ID: 107564
- Mapp v. Ohio (1961) — estimated ID: 106285
- United States v. Leon (1984) — ID: needs verification
- Illinois v. Gates (1983) — ID: needs verification

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
│   └── graph_loader.py           # IN PROGRESS — loads parquet into Neo4j graph
├── embeddings/
│   └── __init__.py
├── preprocessing/
│   └── __init__.py
├── detector/
│   └── __init__.py
├── api/
│   └── __init__.py
├── benchmark/
│   └── __init__.py
└── tests/
    ├── __init__.py
    ├── conftest.py               # Fixtures: raw_cases, merged_cases
    ├── test_data.py              # 13 passing tests ✅
    ├── test_db.py                # Empty — to be built in Week 3
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
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# CourtListener
COURTLISTENER_TOKEN    = os.getenv("COURTLISTENER_TOKEN")
COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"

# Landmark Fourth Amendment anchor cases (CourtListener opinion IDs)
# NOTE: Leon and Gates IDs need verification before running graph_loader.py
LANDMARK_IDS = [
    107729,   # Terry v. Ohio (1968)
    107564,   # Katz v. United States (1967)
    106285,   # Mapp v. Ohio (1961)
    111252,   # United States v. Leon (1984) — NEEDS VERIFICATION
    110930,   # Illinois v. Gates (1983) — NEEDS VERIFICATION
]
```

---

## Timeline Status

| Week | Dates           | Milestone                                     | Status         |
| ---- | --------------- | --------------------------------------------- | -------------- |
| 1    | Feb 24 – Mar 2  | Environment setup, Docker, Neo4j, first cases | ✅ Complete    |
| 2    | Mar 3 – Mar 9   | Full data ingestion, EyeCite parsing, Parquet | ✅ Complete    |
| 3    | Mar 10 – Mar 16 | Neo4j graph build and verification            | 🔄 In Progress |
| 4    | Mar 17 – Mar 23 | BERT embedding pipeline + Milvus indexing     | ⬜ Upcoming    |
| 5    | Mar 24 – Mar 30 | Semantic similarity retrieval layer           | ⬜ Upcoming    |
| 6    | Mar 31 – Apr 6  | Hallucination detector — all three checks     | ⬜ Upcoming    |
| 7    | Apr 7 – Apr 13  | Benchmark dataset construction                | ⬜ Upcoming    |
| 8    | Apr 14 – Apr 20 | Evaluation — precision, recall, F1            | ⬜ Upcoming    |
| 9    | Apr 21 – Apr 27 | Error analysis + graph visualization          | ⬜ Upcoming    |
| 10   | Apr 28 – May 8  | Final writeup and submission                  | ⬜ Upcoming    |

---

## Week 3 — Current State and Next Steps

### Completed

- ✅ Neo4j container running (`verit_neo4j`)
- ✅ Python connection verified (`python -m db.neo4j_client`)
- ✅ `db/graph_loader.py` written but not yet run

### Immediate Next Steps

1. **Verify landmark case IDs** — run `db/verify_landmarks.py` to confirm
   real CourtListener opinion IDs for Leon and Gates before loading graph
2. **Run graph loader** — `python -m db.graph_loader`
3. **Verify graph in Neo4j browser** — run Cypher queries to confirm node/edge counts
4. **Write `tests/test_db.py`** — automated graph validation tests

### Verify Landmarks Script (db/verify_landmarks.py)

```python
import requests
from config import COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL

headers = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}

cases = {
    "United States v. Leon": "united-states-v-leon",
    "Illinois v. Gates":     "illinois-v-gates",
    "Terry v. Ohio":         "terry-v-ohio",
    "Katz v. United States": "katz-v-united-states",
    "Mapp v. Ohio":          "mapp-v-ohio"
}

for name, slug in cases.items():
    url = f"{COURTLISTENER_BASE_URL}/search/?q={slug}&order_by=score+desc&format=json"
    r = requests.get(url, headers=headers)
    data = r.json()
    if data.get("results"):
        result = data["results"][0]
        print(f"{name}:")
        print(f"  cluster_id: {result['cluster_id']}")
        print(f"  opinion_id: {result['opinions'][0]['id']}")
```

### Cypher Queries to Run After Graph Load

```cypher
// Count all nodes
MATCH (c:Case) RETURN count(c) AS total_cases

// Count all edges
MATCH ()-[r:CITES]->() RETURN count(r) AS total_citations

// Check landmark cases
MATCH (c:Case {landmark: true}) RETURN c.name, c.id

// Most cited cases (top 10)
MATCH (c:Case)<-[:CITES]-(other)
RETURN c.name, count(other) AS citations
ORDER BY citations DESC
LIMIT 10

// Check connectivity of Terry v. Ohio
MATCH (terry:Case {id: 107729})<-[:CITES]-(citing)
RETURN count(citing) AS cases_citing_terry
```

---

## Key Design Decisions Made

1. **Multi-circuit corpus** — removed ca9-only filter to get enough cases with plain text
2. **Post-2015 focus** — older cases have poor plain text coverage (~12% vs ~87%)
3. **Two-tier connectivity** — direct citation (1 hop) + shared landmark anchor (2 hops)
4. **Structure-aware chunking** — paragraph boundaries with 1-paragraph overlap, 512 token ceiling
5. **Balanced benchmark** — 50/50 real vs hallucinated, hallucinated split into 3 subtypes
6. **Parquet for processed data** — JSON for raw/debugging, Parquet for all pipeline consumption
7. **Cosine similarity threshold** — tune on validation set (not test set) in Week 8

---

## Known Issues / Things to Watch

- Landmark IDs for Leon and Gates need verification before graph load
- `test_raw_cases_correct_court` checks ca9 cases are present (not that ALL are ca9)
- EyeCite parsing not yet implemented — currently using `opinions_cited` URLs from CourtListener API
- Benchmark generation script not yet written (Week 7)
- Cosine similarity threshold not yet tuned (Week 8)

---

## How to Start Each Session

```powershell
# 1. Navigate to project
cd "C:\Users\ssalh\Grad School\2026\01_Spring\MIS6V99\Verit"

# 2. Activate venv
.venv\Scripts\activate

# 3. Start Docker Desktop (from Start menu, wait for whale icon)

# 4. Start Neo4j
docker-compose up -d

# 5. Verify everything
docker ps
python -m pytest tests/test_data.py -v
```
