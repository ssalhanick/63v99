# Verit — Week 11 Session Context

**Week 11 Goal:** Final writeup and PowerPoint presentation.
All implementation is complete as of Week 10.

---

## Implementation Complete — What Was Built

### Architecture Overview (for writeup)

Verit is a four-layer hallucination detection pipeline for AI-generated legal citations,
scoped to the Fourth Amendment corpus (~1,300 cases).

```
Raw legal text
      ↓
[EyeCite Parser]          — extracts + resolves citation strings to case IDs
      ↓
[Layer 1 — Existence]     — Neo4j lookup: does this case ID exist in the graph?
      ↓ (FAIL → HALLUCINATED, skip Layers 2+3)
[Layer 2 — Semantic]      — Hybrid search: Legal-BERT dense (Milvus HNSW) + BM25 sparse → RRF fusion
      ↓
[Layer 3 — Connectivity]  — Neo4j graph query: citation density vs. corpus cases
      ↓
[Verdict Logic]           — Deterministic threshold combination of L1+L2+L3
      ↓
[Haiku — Explanation]     — RAG: top_matches passed to Claude Haiku for streaming explanation
      ↓
[Streamlit Frontend]      — Three tabs: Citation Checker, Corpus Map, Citation Graph
```

### File Map (complete)

| File | Role |
|---|---|
| `detector/eyecite_parser.py` | Citation extraction + resolution |
| `detector/existence_check.py` | Layer 1 — Neo4j existence lookup |
| `detector/semantic_check.py` | Layer 2 — hybrid dense+sparse search, RRF fusion |
| `detector/connectivity_check.py` | Layer 3 — citation density query |
| `detector/pipeline.py` | Orchestrates all layers, returns CitationVerdict |
| `api/main.py` | FastAPI endpoint wrapping the pipeline |
| `frontend/app.py` | Streamlit UI — three tabs |
| `frontend/llm.py` | Claude Haiku streaming explanation + correction |
| `visualization/umap_viz.py` | UMAP corpus map + hallucination overlay |
| `visualization/graph_viz.py` | Neo4j subgraph → PyVis citation graph |
| `benchmark/density_histogram.py` | Citation density distribution visualization |

---

## Where RAG Lives (be precise in writeup)

RAG in Verit is split across two components:

**Retrieval — Layer 2 (`detector/semantic_check.py`)**
- Input: surrounding paragraph context of the citation (not just the citation string)
- Dense: Legal-BERT (mean-pooled, L2-normalized, 768-dim) → Milvus HNSW cosine ANN search
- Sparse: BM25 over tokenized corpus text
- Fusion: Reciprocal Rank Fusion (RRF) → ranked `top_matches` list
- Output: top-k corpus cases with case name, court, date, cite_count, scores

**Generation — Haiku (`frontend/llm.py`)**
- Input: citation string, verdict, scores, top_matches from Layer 2
- Output: streaming natural language explanation of the verdict
- For HALLUCINATED: additional streaming correction suggestion toward closest real cases

**Critical distinction for writeup:**
> The LLM explains the verdict but does not make it.
> The verdict is computed deterministically by threshold logic in `detector/pipeline.py`.
> Haiku is purely explanatory — it receives the already-computed verdict as input.

---

## Benchmark Results (use these numbers)

```
Thresholds: sim=0.6, rrf=0.01, density=1

Layer 1 (Existence):    P=1.00  R=0.58  F1=0.734
Layer 2 (Semantic):     P=0.00  R=0.00  F1=0.000  ← correct behavior, see below
Layer 3 (Connectivity): P=0.00  R=0.00  F1=0.000  ← correct behavior, see below
Layer 4 (Combined):     P=1.00  R=0.952 F1=0.976

Combined pipeline:      P=1.00  R=0.98  F1=0.990  Accuracy=0.99

Subtype F1:
  REAL: 0.000   ← subtype_f1 for REAL reflects something to interrogate
  C (fully fabricated name): 1.000
  A (fabricated + wrong doctrine): 1.000
  B (real case, wrong proposition): 0.976
```

### How to Narrate the Metrics Honestly

**Layer 1 F1=0.734 is expected.**
Layer 1 catches all Type A and C hallucinations (fully fabricated cases with no Neo4j node).
It misses all 21 Type B hallucinations because those use real case IDs that exist in Neo4j.
Recall=0.58 reflects this scoping, not a bug.

**Layers 2 and 3 F1=0.0 in isolation is expected and is a finding, not a failure.**
Every hallucination that survives Layer 1 is Type B — a real case with a valid embedding
footprint and real citation density. Layers 2 and 3 cannot distinguish Type B hallucinations
from REAL citations in isolation because the signals are identical. This is the correct
architectural behavior: these layers are designed as ensemble signals, not standalone detectors.

**Combined F1=0.990 is legitimate but scoped.**
The one miss (FN=1, benchmark_id=171) is a Type B citation with a year-corrupted Neo4j node.
The metrics reflect performance on the three hallucination types in the benchmark.
They do not reflect performance on the hardest real-world case (see Limitations below).

---

## Known Limitations — Document All of These

These must appear in the writeup. Do not skip any.

### 1. Synthetic Benchmark Bias (most important)
All hallucinated citations in the benchmark were synthetically generated.
Synthetic hallucinations tend to be cleaner and easier to detect than
real LLM hallucinations because:
- They have no Neo4j node at all (Types A/C) — caught trivially by Layer 1
- They are written with obviously fabricated legal context
- Real LLM hallucinations are often more subtle: slightly wrong reporters,
  real cases cited for wrong propositions, plausible-sounding but incorrect doctrine

**Implication:** F1=0.990 likely overstates real-world performance.
The benchmark measures what the system can detect, not what LLMs actually produce.

### 2. Out-of-Scope Hallucination Type (undetectable by design)
A real case cited for a completely wrong legal proposition is undetectable
by any current layer. The case exists in Neo4j (passes L1), its embedding
is semantically similar to Fourth Amendment doctrine (passes L2), and it
has real citation density (passes L3). This is the primary future work item.

### 3. One FN — benchmark_id=171
Type B, year-corrupted citation. Neo4j node has the same incorrect year (2025)
as the hallucinated citation, so the connectivity check cannot distinguish it.
Root cause: corpus data quality, not architecture.

### 4. Layers 2 and 3 F1=0.0 in Isolation
Documented above. Worth a dedicated paragraph in the limitations section
because it looks alarming out of context.

### 5. CourtListener Cluster ID Drift (operational risk)
Live API cluster IDs had drifted from IDs scraped into Neo4j.
Fixed in Week 9 by prioritizing corpus parquet index over live API.
Worth noting as a production deployment risk.

### 6. Fourth Amendment Corpus Only
The system is scoped to ~1,300 Fourth Amendment cases.
Generalization to other legal domains requires re-scraping, re-embedding,
re-indexing, and re-tuning thresholds. Not a limitation per se — a scoping decision.

---

## Visualizations for Writeup (all complete)

| Visualization | File | Status |
|---|---|---|
| UMAP by circuit | `visualization/umap_circuit.html` | ✅ |
| UMAP by year | `visualization/umap_year.html` | ✅ |
| UMAP with hallucination overlay | Screenshot from Corpus Map tab | ✅ |
| Citation density histogram | `visualization/density_histogram.png` | ✅ |
| Citation Graph (PyVis) | Screenshot from Citation Graph tab | ✅ |

### How to Use the Density Histogram in Writeup
The histogram shows REAL citations clustering at high density scores (~35)
while HALLUCINATED citations cluster in the middle range (~22–29).
This is the key visual for Layer 3 — it shows the signal exists but overlaps,
which is why Layer 3 alone has F1=0.0 and must be used as an ensemble signal.

---

## PowerPoint Structure (suggested)

### Slide 1 — Title
Verit: Detecting Hallucinated Legal Citations in AI-Generated Text

### Slide 2 — Motivation
- LLMs hallucinate legal citations with high confidence
- Legal practitioners cannot easily verify citations at scale
- Existing tools focus on general fact-checking, not citation structure

### Slide 3 — Problem Scoping
- Domain: Fourth Amendment, federal appellate corpus
- Hallucination types: A (fabricated + wrong doctrine), B (real case, wrong proposition), C (fully fabricated)
- Out of scope: proposition-level hallucination (real case, wrong legal claim)

### Slide 4 — System Architecture
Use the architecture diagram from above. One slide, clean flow diagram.
Emphasize: deterministic pipeline + RAG explanation layer.

### Slide 5 — Data & Corpus
- ~1,300 Fourth Amendment cases scraped from CourtListener
- Legal-BERT embeddings (768-dim, mean-pooled)
- Milvus HNSW vector index + BM25 sparse index
- Neo4j citation graph

### Slide 6 — Layer 1: Existence Check
- Neo4j lookup by case ID
- P=1.00, R=0.58, F1=0.734
- Catches all Type A and C hallucinations
- Misses Type B (real case IDs)

### Slide 7 — Layer 2: Semantic Check
- Legal-BERT + BM25 + RRF hybrid search
- Context-level check, not citation-string-level
- F1=0.0 in isolation → ensemble signal only
- Show UMAP visualization here

### Slide 8 — Layer 3: Connectivity Check
- Neo4j citation density query
- F1=0.0 in isolation → ensemble signal only
- Show density histogram here

### Slide 9 — Combined Results
- P=1.00, R=0.98, F1=0.990
- Table of layer metrics
- Subtype F1 breakdown

### Slide 10 — RAG Explanation Layer
- Haiku receives top_matches from Layer 2
- Generates streaming verdict explanation
- HALLUCINATED → correction suggestion toward closest real cases
- LLM explains, does not decide

### Slide 11 — Frontend Demo
- Screenshots: Citation Checker, Corpus Map, Citation Graph tabs
- Citation Graph PyVis screenshot

### Slide 12 — Limitations & Future Work
- Synthetic benchmark bias
- Proposition-level hallucination (out of scope)
- Generalization beyond Fourth Amendment
- Production: rate limiting, auth, CourtListener ID drift

### Slide 13 — Conclusion
- Three-layer deterministic pipeline + RAG explanation
- F1=0.990 on benchmark (with caveats)
- Demonstrates viability of structured citation verification
- Foundation for domain-generalizable legal AI tool

---

## Week 11 Session Checklist

- [ ] Write Abstract (~150 words)
- [ ] Write Introduction + Motivation
- [ ] Write Related Work (citation hallucination detection, RAG in legal AI, hybrid search)
- [ ] Write System Architecture section
- [ ] Write Methodology (one subsection per layer)
- [ ] Write RAG section (retrieval + generation, LLM-explains-not-decides)
- [ ] Write Results + honest metrics narration
- [ ] Write Limitations (all 6 above)
- [ ] Write Future Work
- [ ] Write Conclusion
- [ ] Build PowerPoint from slide structure above
- [ ] Insert all 5 visualizations into appropriate sections
- [ ] Screenshot Citation Checker UI for PowerPoint slide 11