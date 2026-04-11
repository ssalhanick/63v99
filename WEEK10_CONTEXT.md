# Verit — Week 10 Session Context

**Week 10 Goal:** Add the Citation Graph tab to the Streamlit frontend and complete
the final writeup visualizations. This is the last implementation week before the
final report.

---

## Carrying Over from Week 9

All Week 9 deliverables are complete:

| Item | Status |
|---|---|
| `visualization/umap_viz.py` — UMAP embedding map | ✅ Done |
| `frontend/llm.py` — Claude Haiku streaming explanations | ✅ Done |
| `frontend/app.py` — Two-tab Streamlit UI | ✅ Done |
| Corpus citation index fallback (parquet → Neo4j ID sync fix) | ✅ Done |
| UMAP focus mode (corpus dim + centroid fallback for HALLUCINATED) | ✅ Done |

### Week 9 Bug Fixes Applied (document in writeup)
- **CourtListener cluster ID drift** — Live API cluster IDs had drifted from the IDs
  scraped into Neo4j. Fixed by flipping lookup priority: corpus parquet index is now
  checked first (guaranteed Neo4j-compatible IDs), CourtListener API is the fallback.
- **`exists` field name mismatch** — API sent `exists`, frontend read `existence`.
  Fixed in `frontend/app.py` line ~117.
- **HALLUCINATED citations invisible on UMAP** — Fixed with centroid fallback:
  HALLUCINATED markers are now placed at the mean coordinate of their top semantic
  matches, making them visible on the corpus map.

---

## Week 10 Primary Task: Citation Graph Tab

### What to Build

A third Streamlit tab — **🔗 Citation Graph** — that renders the Neo4j citation
subgraph for REAL/SUSPICIOUS citations from the last check result.

### Design

```
Tab 3 — 🔗 Citation Graph

[ Select citation to explore ] ← dropdown of REAL/SUSPICIOUS from session_state

[ Network graph — Plotly or PyVis ]
  - Center node: submitted citation (highlighted)
  - 1-hop neighbors: cases it CITES
  - 2-hop neighbors (optional, toggleable): cases those cite
  - Node color: landmark (gold) / corpus (blue) / stub (gray)
  - Node size: proportional to cite_count
  - Edge: CITES relationship
  - Hover: case name, year, court, cite_count

[ Open in Neo4j Browser ] ← button that opens localhost:7474 with Cypher pre-filled

### Neo4j Browser Button — Implementation Detail

The "Open in Neo4j Browser" button appears below the PyVis graph whenever a REAL
or SUSPICIOUS citation is selected. It opens the Neo4j Browser at localhost:7474
with the subgraph Cypher pre-filled and ready to run.

**Streamlit implementation:**
Use `st.link_button()` — not `st.button()` — because it needs to open an external URL.

**URL format:**
```python
import urllib.parse

cypher = f"""MATCH path = (c:Case {{id: {case_id}}})-[:CITES*1..2]->(neighbor:Case)
RETURN path
LIMIT 100"""

encoded = urllib.parse.quote(cypher)
neo4j_url = f"http://localhost:7474/browser/?cmd=edit&arg={encoded}"

st.link_button("Open in Neo4j Browser", neo4j_url)
```

**What the query shows:**
- The selected case as the center node
- All cases it directly cites (1-hop)
- All cases those cite in turn (2-hop) — includes both stubs and full corpus cases
- No `stub = false` filter here — you want to see everything Neo4j has,
  including stub nodes, because the full citation neighborhood is the point
- LIMIT 100 keeps the browser from hanging on highly-connected landmark cases

**Behavior:**
- Clicking the button opens a new browser tab at localhost:7474
- The Cypher is pre-loaded in the editor but NOT auto-run — user hits play to execute
- This is intentional: gives the user a chance to modify the query before running
- Only appears for REAL and SUSPICIOUS verdicts — HALLUCINATED citations are not
  in Neo4j so the button would return empty results
```

### Cypher query to pull subgraph

```cypher
MATCH path = (c:Case {id: $case_id})-[:CITES*1..2]->(neighbor:Case)
WHERE neighbor.stub = false OR neighbor.stub IS NULL
RETURN path
LIMIT 100
```

### Implementation path

**Option A — Plotly + NetworkX** (no new installs needed)
- Pull nodes/edges from Neo4j
- Use `networkx.spring_layout()` for coordinates
- Render with `plotly.graph_objects.Scatter` (nodes) + `Scatter` mode='lines' (edges)
- Pros: already installed, consistent with UMAP tab styling
- Cons: spring layout is non-deterministic; less interactive than PyVis

**Option B — PyVis** (requires `pip install pyvis`)
- Generate HTML with PyVis, embed with `st.components.v1.html()`
- Physics-based layout, draggable nodes, built-in zoom
- Already planned for Week 10 in PROJECTCONTEXT.md
- Pros: much more interactive, better for demos
- Cons: one new dependency

**Recommendation: Option B (PyVis).** It's planned, produces better demo output,
and the install is trivial.

### Install before Week 10 session

```powershell
cd "C:\Users\ssalh\Grad School\2026\01_Spring\MIS6V99\Verit"
.venv\Scripts\activate
pip install pyvis
```

Then add to `requirements.txt`:
```
pyvis==0.3.2
```

### Files to create / modify

| File | Change |
|---|---|
| `visualization/graph_viz.py` | [NEW] Pull subgraph from Neo4j, return PyVis Network object |
| `frontend/app.py` | Add Tab 3 — import graph_viz, render with st.components.v1.html() |

---

## Week 10 Secondary Task: Density Histogram

This was listed as Task #5 in WEEK9_CONTEXT but was not yet built.
The script is fully specced — just needs to be run.

```python
# benchmark/density_histogram.py  (already written in WEEK9_CONTEXT.md)
# Run:
python benchmark/density_histogram.py
# Output: visualization/density_histogram.png
```

> **Blocker:** benchmark.json entries need a `density_score` field populated.
> Check eval_report.json for Layer 3 scores, or re-run entries through
> connectivity_check.py and write scores back to the JSON.

---

## Week 10 Writeup Tasks

Complete the final visualizations for the report:

| Visualization | Source | Status |
|---|---|---|
| UMAP corpus by circuit | `visualization/umap_circuit.html` | ✅ Generated Week 9 |
| UMAP corpus by year | `visualization/umap_year.html` | ✅ Generated Week 9 |
| UMAP with hallucination overlay | Screenshot of Corpus Map tab | ✅ Take during Week 10 session |
| Citation density histogram | `visualization/density_histogram.png` | ⬜ Not yet built |
| Citation graph (PyVis) | Screenshot of Citation Graph tab | ⬜ Not yet built |

---

## Known Limitations to Document (from Week 9)

These belong in the final writeup — do not skip them:

1. **One FN (benchmark_id=171)** — Type B year-corrupted citation; Neo4j node has
   the same incorrect year (`2025`). Root cause: corpus data quality, not architecture.

2. **Layers 2 and 3 F1=0.0 in isolation** — Every hallucinated citation reaching
   those layers is Type B (valid semantic + graph footprint). Not a malfunction.

3. **Layer 4 recall variance ±0.106** — Year-corrupted Type B entries.

4. **Out-of-scope hallucination type** — Real case cited for a wrong proposition;
   undetectable by any current layer. Primary future work item.

5. **CourtListener cluster ID drift** — Documented and fixed Week 9. Worth noting
   as an operational risk for production deployments that skip the corpus index.

---

## How to Start Week 10 Session

```powershell
cd "C:\Users\ssalh\Grad School\2026\01_Spring\MIS6V99\Verit"
.venv\Scripts\activate

# Install PyVis first if not done
pip install pyvis

# Start Docker / Neo4j
docker-compose up -d

# Terminal 1 — FastAPI
uvicorn api.main:app --reload

# Terminal 2 — Streamlit
streamlit run frontend/app.py
```
