# Verit — LLM Suspicious Check

## Where We Are

The core pipeline is built and tested end-to-end:

- ✅ Layer 1 (existence_check) — working
- ✅ Layer 2a (semantic_check) — working, reverted to RRF mode
- ✅ Layer 2b (llm_check) — written, not yet tested
- ✅ Layer 3 (connectivity_check) — working
- ✅ pipeline.py — updated with LLM integration
- ✅ api/main.py — updated with LLMCheckResult model
- ✅ Type A demo (HALLUCINATED) — working
- ✅ Type B demo (SUSPICIOUS) — working after Novak patch
- ⬜ llm_check standalone test — not yet run
- ⬜ Full end-to-end test with new pipeline — not yet run
- ⬜ Frontend updated to show LLM reason — not yet done

---

## Step 1 — Confirm ANTHROPIC_API_KEY is in config

```python
from config import ANTHROPIC_API_KEY
print(ANTHROPIC_API_KEY[:10] if ANTHROPIC_API_KEY else "NOT SET")
```

If not set, add it to `config.py`:

```python
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Step 2 — Run llm_check standalone test

```powershell
py -m detector.llm_check
```

**Expected output:**

```
--- Correct proposition ---
  accurate=True  | The case held that reasonable suspicion justified the stop... | tokens=~1200

--- Wrong proposition (Type B) ---
  accurate=False | The case did not hold that narcotics dog presence alone permits warrantless search... | tokens=~1200
```

If both return `accurate=True`, the prompt needs tightening — see troubleshooting below.
If you get an auth error, check `ANTHROPIC_API_KEY` in config.

---

## Step 3 — Revert semantic_check.py case-specific changes

During the Option B experiment, `semantic_check.py` was modified to add
`_case_specific_similarity` and a `case_id` parameter. These should be
reverted since LLM check replaces that approach.

Confirm the current `semantic_check` signature is back to:

```python
def semantic_check(context_text: str, top_k: int = TOP_K) -> SemanticResult:
```

And the `pipeline.py` call is:

```python
semantic = semantic_check(citation.context_text)
```

If the `case_id` parameter is still present, remove it — it is no longer needed.

---

## Step 4 — Run full pipeline smoke test

```powershell
py -m detector.pipeline
```

Check the output table includes the new `L2b` column and verdicts are correct:

| Citation      | Verdict      | L1    | L2a  | L2b  | L3    | Density |
| ------------- | ------------ | ----- | ---- | ---- | ----- | ------- |
| 392 U.S. 1    | SUSPICIOUS   | True  | True | skip | False | 0       |
| 490 U.S. 1    | HALLUCINATED | False | -    | -    | -     | -       |
| 923 F.3d 1027 | HALLUCINATED | False | -    | -    | -     | -       |

Terry (392 U.S. 1) will still show SUSPICIOUS because it has no graph
connections (density=0) — that is expected and correct for your corpus.

---

## Step 5 — Run the three demo test cases through the full pipeline

```python
from detector.pipeline import run_pipeline

# REAL — correct case, correct proposition
real_text = """
In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held that
a trooper had reasonable suspicion to conduct a traffic stop where a vehicle's
license plate lettering was partially obstructed, and that Utah's license plate
maintenance law applied to out-of-state plates.
"""

# SUSPICIOUS — correct case, wrong proposition (Type B)
suspicious_text = """
In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held that
officers may conduct a warrantless search of a vehicle without reasonable
suspicion whenever a narcotics dog is present at the scene, regardless of
whether the initial stop was lawful.
"""

# HALLUCINATED — fake citation (Type A)
hallucinated_text = """
In United States v. Garrett, 887 F.3d 452 (9th Cir. 2021), the court held
that the plain view doctrine permits officers to seize any item in a vehicle
during a routine traffic stop without a warrant.
"""

for label, text in [("REAL", real_text), ("SUSPICIOUS", suspicious_text), ("HALLUCINATED", hallucinated_text)]:
    results = run_pipeline(text)
    for v in results:
        llm = v.llm_result
        print(f"\n[{label}]")
        print(f"  Verdict:  {v.verdict}")
        print(f"  L2b:      {llm.is_accurate if llm and not llm.skipped else 'skipped'}")
        print(f"  Reason:   {llm.reason if llm else 'n/a'}")
```

**Expected:**

```
[REAL]        Verdict: REAL
[SUSPICIOUS]  Verdict: SUSPICIOUS  |  L2b: False
[HALLUCINATED] Verdict: HALLUCINATED
```

---

## Step 6 — Update the Streamlit frontend to show LLM reason

The API now returns `llm_check.reason` in the response. The frontend should
surface this so the demo shows _why_ a citation is SUSPICIOUS, not just that
it is.

Find where the frontend renders citation results (likely `frontend/app.py`)
and add the reason string beneath the SUSPICIOUS verdict badge:

```python
# Pseudocode — adapt to your actual Streamlit layout
if citation["verdict"] == "SUSPICIOUS":
    st.warning(f"⚠️ SUSPICIOUS")
    if citation.get("llm_check") and not citation["llm_check"]["skipped"]:
        st.caption(f'"{citation["llm_check"]["reason"]}"')
elif citation["verdict"] == "HALLUCINATED":
    st.error("❌ HALLUCINATED")
elif citation["verdict"] == "REAL":
    st.success("✅ REAL")
```

This is the highest-value frontend change for the demo — a one-line explanation
from Haiku is far more compelling than a bare label.

---

## Step 7 — Restart servers and run end-to-end demo

```powershell
# Terminal 1
uvicorn api.main:app --reload

# Terminal 2
streamlit run frontend/app.py
```

Paste all three test cases into the Citation Checker UI and confirm:

- REAL test → green ✅ REAL, no reason shown
- SUSPICIOUS test → yellow ⚠️ SUSPICIOUS + Haiku's reason explaining the mismatch
- HALLUCINATED test → red ❌ HALLUCINATED

---

## Troubleshooting

| Problem                                                   | Fix                                                                                                      |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `llm_check` returns `accurate=True` for wrong proposition | Tighten system prompt — add "If in doubt, mark as inaccurate"                                            |
| `ANTHROPIC_API_KEY` not found in config                   | Add key to `config.py` or `.env` and reload                                                              |
| API 500 error after pipeline update                       | Check `CitationVerdict` dataclass has `llm_result` field — pipeline and api must match                   |
| Streamlit shows old results                               | Hard refresh browser (Ctrl+Shift+R) or restart Streamlit                                                 |
| LLM check always skipped                                  | Check `plain_text` column exists in `cases_enriched.parquet` and is populated for the test case          |
| Haiku returns non-JSON                                    | The regex strip handles markdown fences — if still failing, print `raw_text` in `_call_haiku` to inspect |

---

## What's Not Worth Fixing Before the Demo

- **369 cases still missing citation strings** — these are non-standard reporters
  (NY Slip Op, LEXIS, WL). They're reachable via Layers 2 and 3 but not Layer 1.
  Not fixable without a different data source. Document as a known limitation.

- **Terry v. Ohio returning SUSPICIOUS** — it's in the corpus but has density=0
  because it's a SCOTUS case with no Neo4j neighbors in your state court corpus.
  Expected behavior, not a bug. Note it in the demo.

- **CourtListener ID drift (72% mismatch)** — confirmed to be historical, not a
  data integrity issue. Parquet and Neo4j are fully in sync. No action needed.

- **RRF threshold tuning** — L2a is now just a domain pre-filter before the LLM
  call. The 0.02 threshold is intentionally permissive. No need to tune it.
