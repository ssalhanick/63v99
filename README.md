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
│
├── data/
│   ├── fetch_cases.py          # CourtListener API ingestion (4th Amendment filter)
│   ├── hf_dataset_loader.py    # CLERC + CaseHOLD loader
│   └── raw/                    # Downloaded case JSON
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
| 2    | Mar 3 – Mar 9   | Full data ingestion, EyeCite parsing, edge list       |
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
