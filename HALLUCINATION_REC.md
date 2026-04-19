# Verit — Next Steps Context

## Current State

The pipeline is fully functional end-to-end with all three verdict types working
correctly. The landmark cases (Terry, Katz, Mapp, Leon, Gates) are integrated
across Neo4j, Milvus, and the parquet. The frontend surfaces LLM reasons for
REAL and SUSPICIOUS verdicts and CourtListener search links for top matches.

---

## Outstanding Issue — HALLUCINATED Top Matches Empty

### Root Cause

When Layer 1 fails (`exists=False`), `pipeline.py` short-circuits immediately
and skips semantic search. This means `top_matches=[]` for all HALLUCINATED
citations, so the frontend has nothing to show for "📚 Closest real cases"
or "💡 Suggested Correction".

### Fix — `detector/pipeline.py`

In the Layer 1 FAIL branch, run `semantic_check` to populate `top_matches`
before appending the HALLUCINATED verdict:

```python
if not exists:
    logger.info("Layer 1 FAIL → HALLUCINATED: %s", citation.citation_string)

    # Still run semantic search to get corpus matches for frontend suggestions
    semantic = semantic_check(citation.context_text)

    verdicts.append(CitationVerdict(
        citation_string = citation.citation_string,
        case_name       = citation.case_name,
        case_id         = citation.case_id,
        exists          = False,
        semantic        = semantic,        # ← was None
        llm_result      = None,
        connectivity    = None,
        verdict         = HALLUCINATED,
        context_text    = citation.context_text,
        top_matches     = semantic.top_matches,  # ← was []
    ))
    continue
```

### Verdict Logic Impact

None — `_compute_verdict` still returns HALLUCINATED because `exists=False`
regardless of semantic results. The semantic search result is only used to
populate `top_matches` for the frontend.

### API Impact

`semantic_score` and `dense_score` will now be populated for HALLUCINATED
citations in the API response. Update `api/main.py` if these should be
suppressed for HALLUCINATED verdicts (optional — they provide useful signal).

### Frontend

No changes needed — `render_citation_result` already handles the
"📚 Closest real cases" expander and "💡 Suggested Correction" stream
for HALLUCINATED citations. They will appear automatically once
`top_matches` is populated.

### Verification

After the fix, re-run:

```python
import requests, json
text = """
In United States v. Garrett, 887 F.3d 452 (9th Cir. 2021), the court held that
the plain view doctrine permits officers to seize any item in a vehicle during a
routine traffic stop without a warrant.
"""
resp = requests.post("http://localhost:8000/check-citation", json={"text": text})
data = resp.json()
print(json.dumps(data, indent=2))
# Expected: top_matches populated with 3-5 real Fourth Amendment cases
```

Then paste the HALLUCINATED test case into the UI and confirm:

- "📚 Closest real cases" expander is open by default
- "💡 Suggested Correction" streams a real case recommendation

---

## Remaining Polish Items

### 1. Semantic Score Display

The `semantic_score` column always shows ~0.024 because RRF scores are
mathematically capped by the formula. Consider replacing it in the UI with
either:

- The `llm_check.is_accurate` bool (most meaningful signal)
- The raw `dense_score` (ranges 0.0–1.0, more intuitive)
- Removing it entirely for HALLUCINATED verdicts where it's null

### 2. Terry v. Ohio Density=0 Warning

Terry and other SCOTUS landmarks will still show density=0 in Layer 3
because your corpus is state/lower federal court cases with no direct
`CITES` edges to SCOTUS nodes. This is a known corpus coverage limitation.
Document it in the demo as expected behavior, not a bug.

### 3. 369 Cases Still Missing Citation Strings

These use non-standard reporters (NY Slip Op, LEXIS, WL) that EyeCite
cannot resolve. They are reachable via Layers 2 and 3 but not Layer 1.
No fix available without a different data source. Note as known limitation.

---

## Demo Test Cases

### ✅ REAL

```
In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held that
a trooper had reasonable suspicion to conduct a traffic stop where a vehicle's
license plate lettering was partially obstructed, and that Utah's license plate
maintenance law applied to out-of-state plates.
```

### ⚠️ SUSPICIOUS (Type B)

```
In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held that
officers may conduct a warrantless search of a vehicle without reasonable
suspicion whenever a narcotics dog is present at the scene, regardless of
whether the initial stop was lawful.
```

### ❌ HALLUCINATED (Type A)

```
In United States v. Garrett, 887 F.3d 452 (9th Cir. 2021), the court held
that the plain view doctrine permits officers to seize any item in a vehicle
during a routine traffic stop without a warrant.
```

### 🏛️ Landmark — Terry v. Ohio

```
In Terry v. Ohio, 392 U.S. 1 (1968), the Court held that a brief investigatory
stop is permissible when an officer has reasonable articulable suspicion that
criminal activity is afoot, even without probable cause for a full arrest.
```

### 🏛️ Landmark — Novak (Type B demo)

```
In Novak v. City of Parma, 33 F.4th 296 (6th Cir. 2022), the court held that
officers conducting a warrantless search of a suspect's vehicle incident to
arrest may also search any closed containers found within the passenger
compartment, regardless of whether the arrestee had been secured and posed
no further threat.
```
