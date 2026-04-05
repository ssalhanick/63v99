# Verit — Recommendations for Strengthening Detection & Confidence

> **Context:** These recommendations are written as constructive suggestions to evolve the current prototype into a more robust system. The existing infrastructure is strong — the data pipeline, graph loading, benchmark framework, and API layer are all well-built. These suggestions focus on how to get more signal out of the data already collected and the infrastructure already in place.

---

## 1. Richer Graph Ontology: From Frequency to Relational Strength

### The Current Situation

The graph currently has one node type (`Case`) and one edge type (`CITES`). The only query executed is a one-hop shared-neighbor count: how many cases does the target co-cite alongside corpus cases? That signal is weak at the current corpus scale — the threshold was tuned to 1, meaning almost anything passes.

The same graph data could support significantly more expressive queries if the ontology were extended. The CourtListener data already contains most of what is needed.

### Recommendation 1.1 — Add Doctrine Nodes

Introduce a `Doctrine` node type representing the major Fourth Amendment legal doctrines:

```
(:Doctrine {id: "terry_stop", name: "Terry Stop / Investigatory Stop"})
(:Doctrine {id: "exclusionary_rule", name: "Exclusionary Rule"})
(:Doctrine {id: "plain_view", name: "Plain View Doctrine"})
(:Doctrine {id: "exigent_circumstances", name: "Exigent Circumstances"})
(:Doctrine {id: "automobile_exception", name: "Automobile Exception"})
(:Doctrine {id: "consent_search", name: "Consent to Search"})
(:Doctrine {id: "probable_cause", name: "Probable Cause Standard"})
(:Doctrine {id: "good_faith", name: "Good Faith Exception"})
```

Each corpus case is then linked to the doctrines it primarily applies:

```cypher
(case:Case)-[:APPLIES_DOCTRINE]->(doctrine:Doctrine)
```

This enables a completely new class of query: **do two citations in the same document reference the same doctrine?** If an LLM generates a brief citing two cases, and both cases are linked to `exclusionary_rule` in the graph, their combined use is internally coherent — that's a `REAL` signal. If one case applies to `automobile_exception` and the other to `plain_view` and the context suggests they're being used interchangeably, that's a `SUSPICIOUS` signal.

Doctrine membership can be bootstrapped from the `LEGAL_PRESERVE` token list already in `tokenize_bm25.py` — the domain vocabulary is already annotated. A simple keyword classifier over the case `plain_text` would assign doctrine labels with reasonable accuracy without requiring a fine-tuned model.

### Recommendation 1.2 — Differentiate Citation Relationship Types

The current `CITES` edge treats all citations as identical. In legal practice they are not:

| Relationship | Meaning | Current behavior |
|:---|:---|:---|
| `FOLLOWS` | Case A directly applies the holding of Case B | Collapsed into `CITES` |
| `DISTINGUISHES` | Case A cites Case B but reaches a different outcome | Collapsed into `CITES` |
| `OVERRULES` | Case A explicitly nullifies Case B's holding | Collapsed into `CITES` |
| `EXTENDS` | Case A expands Case B's holding to new facts | Collapsed into `CITES` |

Replacing `CITES` with typed relationships changes what "co-citation" means:

- Two cases cited together that both `FOLLOW` a shared authority → strong coherence signal
- A case cited that is `OVERRULED` by another case in the same brief → hallucination or error signal
- A case cited whose holding was `DISTINGUISHED` in the very cluster being searched → context mismatch

EyeCite already extracts some of this signal (it identifies negative treatment markers like "but see", "distinguished by", etc.). The `treatment` field on CourtListener opinion relationships carries this data as well.

### Recommendation 1.3 — Add Court Hierarchy Nodes

```
(:Court {id: "scotus"})
(:Court {id: "ca9"})-[:BOUND_BY]->(:Court {id: "scotus"})
(:Court {id: "ca4"})-[:BOUND_BY]->(:Court {id: "scotus"})
```

This models the hierarchical binding authority structure. A 9th Circuit brief citing a 5th Circuit case is persuasive but not binding — the graph could model that distinction. More importantly, it enables a precedential weight score: citations to SCOTUS decisions carry more authority than citations to district courts, and the connectivity layer could weight edges by court level rather than treating all `CITES` edges equally.

---

## 2. Cross-Citation Relational Analysis

### The Current Situation

Each citation in a document is verified independently. There is no analysis of whether multiple citations in the same document are internally consistent with each other.

### Recommendation 2.1 — Co-Citation Coherence Score

When a document contains multiple citations, compute a pairwise coherence score for each citation pair:

**Graph-based:** What is the shortest path between Case A and Case B in the citation graph? Cases that are doctrinally related will have short paths through shared authorities. Cases that are unrelated will have long paths or no path at all.

```cypher
MATCH path = shortestPath(
  (a:Case {id: $id_a})-[:CITES*..5]-(b:Case {id: $id_b})
)
RETURN length(path) AS hops
```

A hallucinated citation dropped into a brief alongside real cases will often have no path to its neighbors — it sits in isolation in the graph, disconnected from the doctrinal cluster the brief is about.

**Embedding-based:** Compute cosine similarity between the embedded vectors of Case A and Case B directly. Cases discussing the same Fourth Amendment doctrine will be close in embedding space. A fabricated case in an area the LLM doesn't understand well will generate context that sits far from the embedding cluster of the real cases in the same brief.

**Doctrine overlap:** If the richer ontology from §1.1 is implemented, check whether all cited cases share at least one common `APPLIES_DOCTRINE` label. A coherent brief has doctrinal unity; a hallucinated brief may not.

### Recommendation 2.2 — Citation Neighborhood Jaccard Similarity

For two citations A and B, compute the Jaccard similarity of their citation neighborhoods:

```
J(A, B) = |neighbors(A) ∩ neighbors(B)| / |neighbors(A) ∪ neighbors(B)|
```

Real cases cited together in a Fourth Amendment brief will share many of the same precedents in their citation neighborhoods (Terry, Katz, Leon, etc.). A hallucinated case will have either an empty neighborhood (stub node only) or a neighborhood that overlaps poorly with real Fourth Amendment authorities. This gives a continuous pairwise confidence score across all citation pairs in a document.

---

## 3. LLM-as-Judge for Holding Verification (L4 Replacement / Extension)

### The Current Situation

Layer 4 currently checks only `year` and `court` string matches against the graph node — purely structural metadata. It cannot assess whether the case is being used for the correct legal proposition.

### Recommendation 3.1 — LLM Holding Verification

The most important undetected hallucination type is: **a real case, cited for something it does not say.** None of the current four layers can catch this. LLM-as-judge is the right tool for it.

The approach:

1. For each passing citation, retrieve its full `plain_text` from the corpus (already stored in parquet)
2. Extract a summary of the case's actual holding (either via the LLM or from CourtListener's `headmatter` field)
3. Extract the claimed holding from the citation context in the AI-generated document (the 3-sentence window is already captured)
4. Prompt a judge LLM to evaluate alignment:

```
System: You are a federal appellate law clerk. Your task is to determine
        whether a legal citation is being used accurately.

User:   Case: {case_name} ({year}, {court})
        Actual holding (from opinion text): {holding_summary}
        How it is cited in the document: "{citation_context}"
        
        Does the document accurately represent what this case held?
        Respond: ACCURATE | INACCURATE | UNCERTAIN
        Brief reasoning: [1-2 sentences]
```

The `top_matches` field already exists on every `CitationResult` and flows through the API — the RAG retrieval infrastructure is already in place. Only the holding-extraction prompt and the judge call need to be built.

### Recommendation 3.2 — Multi-Judge Ensemble for Calibration

A single LLM call can be miscalibrated — the model may be too permissive or too strict based on how the prompt is framed. A multi-judge ensemble addresses this:

- Run the holding verification prompt with 2-3 different `system` prompt personas (e.g., "strict textualist clerk", "pragmatic common law clerk", "neutral fact-checker")
- Record each judge's verdict independently
- Produce a confidence score based on judge agreement:
  - All three agree ACCURATE → high confidence REAL
  - All three agree INACCURATE → high confidence HALLUCINATED
  - Mixed verdicts → SUSPICIOUS, flag for human review

This is directly analogous to the existing L2 ⊕ L3 → SUSPICIOUS logic but applied to holding-level accuracy rather than structural signals.

### Recommendation 3.3 — Phrasing Authenticity Check

Federal appellate opinions have highly consistent phrasing patterns. Legal BERT was trained on this vocabulary, and an LLM fine-tuned on legal text could be prompted to assess whether the citation context reads like authentic judicial writing:

- **Authentic signal:** "The Court held that a Terry stop requires reasonable articulable suspicion of criminal activity." — standard legal phrasing
- **Hallucination signal:** "The Court established that officers may use the rainbow protocol under the quantum warrant exception." — plausible-sounding but non-standard

A simpler proxy: compute perplexity of the citation context against a legal domain language model. High perplexity relative to the corpus mean is a hallucination signal. The legal-bert model already in the stack can produce token-level log-probabilities that feed this metric without an additional model.

---

## 4. Retrieval Improvements

### Recommendation 4.1 — Chunk-Level Embeddings Instead of Full-Document Centroids

The most impactful retrieval improvement with least new infrastructure: store paragraph-level embeddings in Milvus rather than per-case centroids.

**Current:** 1 vector per case → 1,358 vectors in Milvus  
**Proposed:** ~10-50 vectors per case → ~15,000–70,000 vectors in Milvus, each tagged with `case_id` and `chunk_index`

Querying a 3-sentence context window against a paragraph embedding is a much better match geometry than querying it against a centroid over an entire 50,000-character opinion. The highest-scoring chunk across all paragraphs of a case defines that case's relevance score. This change alone would substantially improve recall for relevant but topically narrow cases.

The embedding pipeline (`embed_cases.py`) already performs chunking — the chunk vectors just aren't stored individually. The parquet output could trivially be changed to one row per chunk.

### Recommendation 4.2 — Fix BM25 Query Tokenization

As noted in the evaluation, corpus tokenization uses spaCy lemmatization but query tokenization uses a bare regex. The fix is simple and high-value:

```python
# Cache a lightweight spaCy pipe at startup alongside the BM25 index
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def _tokenize_query(text: str) -> list[str]:
    doc = _nlp(text.lower()[:5000])
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop or token.lemma_ in LEGAL_PRESERVE
        if token.is_alpha and len(token) > 1
    ]
```

This brings query tokenization into alignment with corpus tokenization and restores BM25's full recall potential. The spaCy model is already a dependency in `requirements.txt`.

### Recommendation 4.3 — Circuit-Aware Retrieval Boosting

The citing document often comes from a specific jurisdiction. If that information is available (e.g., extracted from the document header or inferred from citation patterns), Milvus supports metadata filtering:

```python
results = milvus.search(
    data=[query_vector],
    filter=f"court_id in ['ca9', 'scotus']",  # restrict to binding circuits
    ...
)
```

A 9th Circuit document should weight 9th Circuit and SCOTUS cases higher than 5th Circuit cases, even if the 5th Circuit case is semantically closer in embedding space. This is a well-understood IR technique (jurisdiction-aware retrieval) that maps cleanly onto the existing Milvus schema since `court_id` is already stored in the case metadata parquet.

---

## 5. Calibrated Confidence Scoring

### Recommendation 5.1 — Replace Boolean Fusion with a Learned Score

The current verdict fusion collapses all continuous signals to booleans and combines them with AND/OR logic. A learned scoring model over the raw signals would be more expressive and produce a calibrated probability:

**Input features per citation:**
- `l1_exists`: 0 or 1
- `l2_rrf_score`: float (0 to ~0.05)
- `l2_dense_score`: float (0 to 1)
- `l3_density_score`: integer
- `l4_valid`: 0, 1, or null
- `name_match_score`: fuzzy match between citation party names and node name
- `cross_citation_jaccard`: from §2.2 (if multiple citations present)

**Output:** `p_hallucinated` float (0 to 1)

A logistic regression trained on the 200-entry benchmark dataset would give a calibrated probability without requiring a large dataset. The benchmark already has ground-truth labels, and `evaluate.py` already computes all raw scores. The training data is right there.

The resulting score can be bucketized into `REAL / SUSPICIOUS / HALLUCINATED` for backward compatibility, but also surfaced as a raw float in the API response for consumers who want to set their own threshold.

### Recommendation 5.2 — Expose Scores in the API and UI

Even before a learned model is built, the raw layer scores should be exposed in the API response and the UI:

```json
{
  "verdict": "SUSPICIOUS",
  "confidence_signals": {
    "rrf_score": 0.023,
    "dense_score": 0.61,
    "density_score": 2,
    "metadata_valid": true
  }
}
```

The `CitationVerdict` dataclass already stores all of these. They are just not serialized through the API response schema. This is a small schema change with high value for any downstream consumer.

---

## 6. Additional High-Value Recommendations

### Recommendation 6.1 — Party Name Consistency Check

The citation string contains party names (e.g., `"United States v. Jones"`). The graph node stores the case name. A simple fuzzy string match between the cited party names and the node `name` property catches a class of hallucination the current pipeline misses entirely: correct case ID, wrong party names.

```python
from rapidfuzz import fuzz
name_score = fuzz.token_sort_ratio(cited_name, node_name) / 100.0
```

A score below ~0.70 on a tokenized comparison of party names is a strong hallucination signal. This requires no new infrastructure — just an additional property fetch in the Layer 1 query and a string comparison.

### Recommendation 6.2 — PageRank as an Authority Weight

The graph's 30,806 CITES edges support PageRank computation, which measures the importance of a case in the citation network based not just on how often it is cited but on who cites it:

```cypher
CALL gds.pageRank.stream('case-graph') 
YIELD nodeId, score
```

PageRank could replace or augment the raw citation density in Layer 3. A case with high PageRank in the Fourth Amendment corpus is a genuine authority. A hallucinated case that happens to share a single co-cited precedent (density = 1, passes current L3) but has zero PageRank is still suspicious.

Neo4j Graph Data Science library includes PageRank natively — it's available without any additional dependencies.

### Recommendation 6.3 — Temporal Precedential Logic

Legal citation has a temporal constraint: a case can only be cited for a holding it existed to make at the time of the citing document. Two checks are currently missing:

1. **Citation predates cited case:** If the AI claims a 2015 case cites a 2019 decision, the citation is impossible. The graph has `year` on all nodes — this is a trivially cheap check.
2. **Overruled case status:** If Case B was overruled by Case C, citing Case B for the overruled proposition is a legal error (not necessarily a hallucination, but a confidence signal). CourtListener's `negative_treatment` field carries this data.

### Recommendation 6.4 — Wire Layer 4 Into Production (One-File Fix)

This is the single highest-ROI change relative to effort. Layer 4 (`metadata_check.py`) achieves F1=1.0 on Type B hallucinations in the benchmark and requires three lines added to `pipeline.py`:

```python
from detector.metadata_check import check_metadata

# After Layer 1 passes:
meta = check_metadata(citation.case_id, citation.citation_string, driver=driver)
if not meta.is_valid:
    # verdict = HALLUCINATED, short-circuit
```

Everything else — the dataclass, the court alias table, the Neo4j query — is already built and tested.

---

## Priority Summary

| Recommendation | Effort | Signal Gain | Dependencies |
|:---|:---|:---|:---|
| Wire Layer 4 into pipeline | ⬛ Low (~3 lines) | 🔴 High — catches Type B in production | None |
| Fix BM25 query tokenization | ⬛ Low (~10 lines) | 🟠 Medium — restores BM25 recall | None |
| Party name consistency check | ⬛ Low | 🟠 Medium — new hallucination class | `rapidfuzz` |
| Expose raw scores in API/UI | ⬛ Low | 🟢 UX — actionable for consumers | None |
| Chunk-level retrieval | 🟨 Medium | 🔴 High — better semantic match geometry | Milvus reindex |
| LLM holding verification | 🟨 Medium | 🔴 High — only layer that catches misuse | Claude API |
| Cross-citation coherence (Jaccard) | 🟨 Medium | 🟠 Medium — multi-citation signal | Graph query |
| Doctrine node ontology | 🟥 High | 🔴 High — enables relational reasoning | Schema + annotation |
| PageRank authority weighting | 🟨 Medium | 🟠 Medium — weighted connectivity | Neo4j GDS |
| Temporal precedential logic | 🟨 Medium | 🟠 Medium — catches impossible citations | Graph property check |
| Learned calibrated scorer | 🟨 Medium | 🔴 High — calibrated P(hallucinated) | Benchmark data |
| Typed citation relationships | 🟥 High | 🔴 High — overruled/distinguished signals | CourtListener + schema |
| Circuit-aware retrieval boosting | ⬛ Low | 🟢 Low-medium — jurisdiction signal | Milvus filter |
| Multi-judge LLM ensemble | 🟥 High | 🟠 Medium — calibration, not raw signal | Claude API |

---

*These recommendations are intended constructively. The existing pipeline is a solid foundation — most of the data needed to implement these enhancements is already collected and stored. The suggestions aim to close the gap between the infrastructure complexity and the signal actually extracted from it.*
