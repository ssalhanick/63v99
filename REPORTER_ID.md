# Verit — Reporter String Enrichment Context

## Why We're Doing This

EyeCite extracts reporter strings from text (e.g. `969 F.3d 285`).
The pipeline resolves these to Neo4j case IDs via `_CORPUS_INDEX` in
`detector/eyecite_parser.py`. `_CORPUS_INDEX` is built from the `citations`
column in `cases_enriched.parquet`.

Some corpus cases were scraped into Neo4j without their reporter citation
strings populated. These cases exist in Neo4j by ID but are invisible to
the pipeline — EyeCite can't resolve them, so they short-circuit at Layer 1
and return HALLUCINATED regardless of their actual graph presence or
semantic footprint.

**Consequence:** Type B hallucinations (real case, wrong proposition) cannot
be demonstrated in the live demo if the target case is missing its reporter
string. Layers 2 and 3 are never reached for these cases.

**Fix:** Enrich `cases_enriched.parquet` with reporter strings pulled from
CourtListener `/clusters/{id}/` for cases where `citations` is null/empty.
No Neo4j changes, no embedding changes, no re-scraping.

---

## What Not To Touch

- Neo4j — no changes, IDs are already correct
- Milvus — no changes, embeddings are already indexed
- Raw JSON scrape files — bypass completely, parquet is source of truth
- `_CORPUS_INDEX` — rebuilds automatically from parquet on server restart

---

## Steps

### Step 1 — Audit the gap

Run this to see how many cases are missing citation strings:

```python
import pandas as pd
import ast

df = pd.read_parquet("data/processed/cases_enriched.parquet",
                     columns=["case_id", "case_name", "citations"])

def is_empty(val):
    if val is None:
        return True
    if isinstance(val, float):  # NaN
        return True
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return len(parsed) == 0
        except:
            return val.strip() == ""
    if isinstance(val, list):
        return len(val) == 0
    return True

missing = df[df["citations"].apply(is_empty)]
print(f"{len(missing)} / {len(df)} cases missing citation strings")
print(missing[["case_id", "case_name"]].head(20))
```

### Step 2 — Enrich from CourtListener

Hit `/clusters/{case_id}/` for each missing case and pull the `citations` field.
We use the existing case_id directly — no cluster ID drift risk because we're
not doing a citation lookup, just fetching metadata for an ID we already have.

```python
import requests
import time
import pandas as pd
import ast
from config import COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL

HEADERS = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}

def is_empty(val):
    if val is None:
        return True
    if isinstance(val, float):
        return True
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return len(parsed) == 0
        except:
            return val.strip() == ""
    if isinstance(val, list):
        return len(val) == 0
    return True

def fetch_citations(case_id: int) -> list[str]:
    url = f"{COURTLISTENER_BASE_URL}/clusters/{case_id}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # citations is a list of reporter strings e.g. ["969 F.3d 285"]
        return data.get("citations", [])
    except Exception as e:
        print(f"  Failed for {case_id}: {e}")
        return []

# Load full parquet
df = pd.read_parquet("data/processed/cases_enriched.parquet")

missing_mask = df["citations"].apply(is_empty)
missing_ids  = df[missing_mask]["case_id"].tolist()
print(f"Fetching citations for {len(missing_ids)} cases...")

updates = {}
for i, case_id in enumerate(missing_ids):
    citations = fetch_citations(int(case_id))
    updates[case_id] = citations
    if i % 50 == 0:
        print(f"  {i}/{len(missing_ids)} done")
    time.sleep(0.5)  # polite delay — respect CourtListener rate limits

# Write back to parquet in place
df.loc[missing_mask, "citations"] = df[missing_mask]["case_id"].map(updates)
df.to_parquet("data/processed/cases_enriched.parquet", index=False)
print("Parquet updated.")
```

### Step 3 — Restart servers

`_CORPUS_INDEX` is built at module import time — it will automatically
pick up the new citation strings on the next server start.

```powershell
# Stop FastAPI and Streamlit (Ctrl+C in each terminal), then:

# Terminal 1
uvicorn api.main:app --reload

# Terminal 2
streamlit run frontend/app.py
```

### Step 4 — Verify

Confirm the index now contains the reporter string for Novak:

```python
from detector.eyecite_parser import _CORPUS_INDEX
print(_CORPUS_INDEX.get("969 F.3d 285"))
# Expected: (6336455, "Anthony Novak v. City of Parma, Ohio")
```

Then paste this into the Citation Checker to confirm Type B demo works end-to-end:

> In _Novak v. City of Parma_, 969 F.3d 285 (6th Cir. 2022), the court held
> that officers conducting a warrantless search of a suspect's vehicle incident
> to arrest may also search any closed containers found within the passenger
> compartment, regardless of whether the arrestee had been secured and posed
> no further threat.

**Expected result:** REAL or SUSPICIOUS — NOT HALLUCINATED.
If it returns REAL or SUSPICIOUS, Layers 2 and 3 are being reached and the
Type B demonstration is working correctly.

---

## Risk Assessment

- **Low risk** — parquet update only, no downstream index changes
- **Worst case** — CourtListener returns empty for some cases, they stay as-is
- **No rollback needed** — if something goes wrong, the only change is the
  `citations` column in the parquet. All other data is untouched.
- **Time estimate** — ~10-15 minutes of API calls at 0.5s delay per case
