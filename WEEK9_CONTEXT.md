# Verit — Week 9 Session Context

**Week 9 Goal:** Run and validate the three things built for this week:
`visualization/umap_viz.py`, `frontend/llm.py`, and the updated `frontend/app.py`.
All three scripts already exist. This week is about launching them, fixing issues as they
surface, and producing the visualization outputs needed for the final writeup.

---

## What Is Already Built

| File | Status | Purpose |
|---|---|---|
| `visualization/umap_viz.py` | ✅ Built | StandardScaler → UMAP → Plotly figure; exports HTML standalone |
| `frontend/llm.py` | ✅ Built | Claude Haiku streaming explanation + correction via RAG |
| `frontend/app.py` | ✅ Built | Streamlit app with two tabs: Citation Checker + Corpus Map |

### What `umap_viz.py` Does

- Loads `data/processed/embeddings.parquet` (768-dim vectors) and
  `data/processed/cases_cleaned.parquet` (metadata)
- Applies `StandardScaler` (zero mean, unit variance per dimension)
- Runs UMAP: `n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42`
- Returns a Plotly scatter figure colored by `circuit` or `year`
- `overlay_submitted_citations()` — adds verdict-colored markers on top of corpus points
- Standalone run: `python -m visualization.umap_viz` → saves
  `visualization/umap_circuit.html` and `visualization/umap_year.html`

### What `llm.py` Does

- Two streaming generators: `stream_explanation()` and `stream_correction()`
- Both passed directly to `st.write_stream()`
- RAG: `top_matches` from Layer 2 are formatted into the system prompt so Haiku's
  explanations reference actual corpus cases
- Model: `claude-haiku-4-5-20251001` (set in config.py as `ANTHROPIC_MODEL`)
- `stream_correction()` only called when verdict = `HALLUCINATED` and `top_matches` is non-empty

### What `app.py` Does (Week 9 version)

Two-tab Streamlit layout:

**Tab 1 — ⚖️ Citation Checker**
- Text area → "Check Citations" button → POST to `http://localhost:8000/check-citation`
- Summary metrics (total / real / suspicious / hallucinated)
- Per-citation card: verdict badge, semantic score, density score, top corpus matches expander
- Haiku toggle (default ON) → streams explanation below each card; streams correction for HALLUCINATED
- Stores last verdict list in `st.session_state["last_citations"]` for UMAP overlay

**Tab 2 — 🗺️ Corpus Map**
- UMAP figure cached with `@st.cache_resource` (~30s load on first open)
- Color-by toggle: `circuit` | `year`
- Overlays submitted citations from `session_state["last_citations"]` if present
  (star=REAL, diamond=SUSPICIOUS, X=HALLUCINATED)

---

## Known Issue: config.py Thresholds Not Updated

`config.py` still has the pre-Week-8 defaults. Update before running anything:

```python
# config.py — required before Week 9 session
SIMILARITY_THRESHOLD       = 0.60   # was 0.75
RRF_THRESHOLD              = 0.010  # was 0.02
CITATION_DENSITY_THRESHOLD = 1      # was 3
```

---

## Dependencies — All Installed

| Package | In requirements.txt |
|---|---|
| `umap-learn==0.5.12` | ✅ |
| `plotly==6.7.0` | ✅ |
| `anthropic==0.86.0` | ✅ |
| `streamlit==1.55.0` | ✅ |
| `scikit-learn==1.7.2` | ✅ (StandardScaler) |

No `pip install` needed. `pyvis` is not yet installed — that's Week 10.

---

## How to Start This Session

```powershell
cd "C:\Users\ssalh\Grad School\2026\01_Spring\MIS6V99\Verit"
.venv\Scripts\activate

# Start Docker Desktop (whale icon in taskbar)
docker-compose up -d
docker logs verit_neo4j --tail 3   # wait for "Started."

# Terminal 1 — FastAPI backend
uvicorn api.main:app --reload

# Terminal 2 — Streamlit frontend
streamlit run frontend/app.py

# Optional: generate standalone UMAP HTML files for writeup
python -m visualization.umap_viz
```

---

## Week 9 Task List

### 1. Fix `config.py` thresholds (1 min)
Update the three threshold values as shown above.

### 2. Generate standalone UMAP HTML exports (for writeup)
```powershell
python -m visualization.umap_viz
```
Expected output:
- `visualization/umap_circuit.html`
- `visualization/umap_year.html`

Open both in browser. Look for:
- Circuit clustering — cases from the same circuit should cluster together
- Any obvious outlier regions (these are likely the cases with unusual plain text)

### 3. Launch and smoke-test the full Streamlit app
1. Start FastAPI in one terminal, Streamlit in another
2. Paste a paragraph containing a known real citation (e.g. _Terry v. Ohio, 392 U.S. 1_)
3. Verify: verdict = REAL, Haiku explanation streams, tab 2 shows corpus map
4. Paste a fabricated citation → verify: HALLUCINATED, correction streams
5. Switch to Corpus Map tab → verify the UMAP renders and citations overlay correctly

### 4. Produce the four writeup visualizations

| Visualization | How to Get It |
|---|---|
| UMAP corpus by circuit | `umap_circuit.html` from standalone run |
| UMAP corpus by year | `umap_year.html` from standalone run |
| UMAP with hallucination overlay | Screenshot of Corpus Map tab after submitting test cases |
| Citation density histogram | Not yet built — see below |

### 5. Citation Density Histogram (not yet built)
This is the one visualization from the plan that isn't implemented yet.
Build a simple script:

```python
# benchmark/density_histogram.py
import json
import matplotlib.pyplot as plt

data = json.load(open("benchmark/benchmark.json", encoding="utf-8"))

real_density    = [e["density_score"] for e in data if e["label"] == "REAL"
                   and "density_score" in e]
halluc_density  = [e["density_score"] for e in data if e["label"] == "HALLUCINATED"
                   and "density_score" in e]

plt.figure(figsize=(8, 4))
plt.hist(real_density,   bins=20, alpha=0.6, label="REAL",         color="#2e7d32")
plt.hist(halluc_density, bins=20, alpha=0.6, label="HALLUCINATED",  color="#c62828")
plt.xlabel("Citation Density Score (Layer 3)")
plt.ylabel("Count")
plt.title("Citation Density Distribution — Real vs. Hallucinated")
plt.legend()
plt.tight_layout()
plt.savefig("visualization/density_histogram.png", dpi=150)
print("Saved → visualization/density_histogram.png")
```

> Note: benchmark.json entries need `density_score` populated. If they don't have it,
> run the entries through `connectivity_check.py` and save the scores back to the JSON,
> OR pull the scores from `eval_report.json` false negatives / test set breakdown.

---

## Known Limitations to Document in Writeup

From Week 8 evaluation — include these honestly in the Week 9 / Week 10 writeup:

1. **One FN (benchmark_id=171)** — Type B year-corrupted citation; Neo4j node has the
   same incorrect year (`2025`) the benchmark injected. Layer 4 undetectable.
   Root cause: corpus data quality, not architecture.

2. **Layers 2 and 3 F1=0.0 in isolation** — not a malfunction. Every hallucinated
   citation reaching those layers is Type B (valid semantic + graph footprint from the
   real underlying case). Layers 2 and 3 provide redundancy for edge cases outside this
   benchmark's scope.

3. **Layer 4 recall variance ±0.106** — year-corrupted Type B entries where the Neo4j
   node carries the same wrong year, or where the corrupted court matches the real court.
   Confirmed FN (id=171) is an example of the former.

4. **Out-of-scope hallucination type** — a real case cited for a proposition it does not
   support is undetectable by any current layer. This is the primary future work item.

---

## Files to Add to `.gitignore` (if not already)

```
visualization/umap_circuit.html
visualization/umap_year.html
visualization/density_histogram.png
```

---

## What PROJECTCONTEXT.md Already Says About Week 9

The PROJECTCONTEXT.md Week 9 section documents:
- UMAP: StandardScaler → UMAP parameters → overlay pattern (all implemented in umap_viz.py ✅)
- LLM: RAG via top_matches, streaming via st.write_stream (implemented in llm.py ✅)
- Citation density histogram: not yet implemented — task #5 above
- Precision-recall curve for Layers 2/3: low priority given both show F1=0 in isolation;
  skip or note as out of scope

The PROJECTCONTEXT already documents Week 9 scripts as built. Update it with actual
results (UMAP runtime, any visual observations) after this session.
