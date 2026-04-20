# Verit — Benchmarking Changes Context

## Why the Benchmark Needs Updating

The original benchmark was designed around a four-layer pipeline where Layer 4
(metadata check — year/court validation) was the primary Type B detector. Layer 4
is no longer active. It has been replaced by Layer 2b (LLM proposition accuracy
check), which catches a fundamentally different and more realistic Type B failure
mode: **proposition hallucination** — a real citation string paired with a
fabricated or inverted holding.

The current test set's Type B cases are year/court corruptions, which Layer 2b
cannot detect (it checks the proposition text, not the metadata). Running the
benchmark as-is will show Layer 2b contributing nothing, which is misleading —
it does catch proposition hallucinations, just not the ones in the current test set.

---

## What Changes

### Test Set

|        | Old                                        | New                               |
| ------ | ------------------------------------------ | --------------------------------- |
| Type A | Fabricated citation string                 | No change                         |
| Type B | Real citation + wrong year/court           | Real citation + wrong proposition |
| Type C | Real citation + correct proposition (REAL) | No change                         |
| Labels | HALLUCINATED / SUSPICIOUS / REAL           | No change                         |

**Type B cases to build:**

- Take 30-40 real citation strings that resolve in `_CORPUS_INDEX`
- Keep the citation string exactly as-is
- Write a plausible-sounding but factually wrong proposition for each
- The wrong proposition should use correct legal vocabulary but misstate
  the holding (reverse the ruling, attribute a different standard, describe
  facts not in the opinion)
- Label these as SUSPICIOUS (expected verdict)

**Good source cases for Type B propositions:**
Use cases already in your corpus with high connectivity (well-known, easy
to verify the real holding against):

```
92 F.4th 1279   — Hoskins v. Withers (traffic stop, license plate)
33 F.4th 296    — Novak v. City of Parma (retaliatory arrest)
157 N.E.3d 406  — Morrison v. Horseshoe Casino
232 N.E.3d 419  — State v. Camper
92 F.4th 1279   — Hoskins v. Withers
392 U.S. 1      — Terry v. Ohio (now in corpus)
389 U.S. 347    — Katz v. United States (now in corpus)
```

### Benchmark Script

| Component                    | Change                                                            |
| ---------------------------- | ----------------------------------------------------------------- |
| Layer 4 row in results table | Replace with Layer 2b                                             |
| Layer 2b metric calculation  | Add — measure per-verdict contribution of `llm_check.is_accurate` |
| Type B subtype label         | Change from "metadata" to "proposition"                           |
| CV folds                     | No change — same 10-fold structure                                |
| Metrics                      | No change — precision/recall/F1 per verdict type                  |

### Results Table Format (new)

| Layer                      | Precision | Recall | F1  |
| -------------------------- | --------- | ------ | --- |
| Layer 1 — Existence        |           |        |     |
| Layer 2a — Semantic        |           |        |     |
| Layer 2b — LLM Proposition |           |        |     |
| Layer 3 — Connectivity     |           |        |     |
| **Combined**               |           |        |     |

Subtype F1: A=?, B=?, C=?

---

## What Stays the Same

- Overall benchmark structure (held-out 100 + 10-fold CV on 500)
- Type A test cases (fabricated citations → HALLUCINATED)
- Type C / REAL test cases
- Precision/recall/F1 metrics
- Zero false positive requirement
- CV methodology and fold assignment

---

## Implementation Order

### Step 1 — Read current benchmark files

```powershell
python -c "
with open('tests/benchmark.py') as f:
    print(f.read())
"
```

```powershell
python -c "
import pandas as pd
df = pd.read_csv('data/benchmark/benchmark_test_set.csv')
print(df.columns.tolist())
print(df.head(10).to_string())
print(df['type'].value_counts())
"
```

### Step 2 — Build new Type B test cases

For each Type B case:

1. Pick a real citation string that resolves in `_CORPUS_INDEX`
2. Pull the first 2,000 words of `plain_text` from `cases_enriched.parquet`
3. Write a wrong proposition (plausible vocabulary, wrong holding)
4. Expected verdict: SUSPICIOUS
5. Add to test set CSV with `type=B`, `subtype=proposition`

### Step 3 — Rewrite benchmark script

- Replace Layer 4 measurement with Layer 2b
- Layer 2b contribution = cases where `llm_check.is_accurate=False`
  correctly identifies a Type B proposition hallucination
- Keep everything else identical

### Step 4 — Run held-out test (100 entries)

```powershell
py -m tests.benchmark --split held-out
```

### Step 5 — Run 10-fold CV (500 entries)

```powershell
py -m tests.benchmark --split cv
```

### Step 6 — Compare against baseline

Key comparisons to make:

- Type B F1: new vs old (expect improvement)
- Combined F1: new vs old (should stay high)
- False positive rate: must remain 0
- Layer 2b F1: new metric, no baseline

---

## Expected Results

Layer 2b should now show non-zero precision/recall/F1 for Type B cases,
replacing what Layer 4 was previously doing. Combined F1 may dip slightly
from 0.990 since Layer 4's near-perfect metadata check is gone, but
Layer 2b should compensate by catching proposition hallucinations that
Layer 4 never tested.

If Type B F1 drops significantly, the LLM prompt in `llm_check.py` may
need tightening — see the troubleshooting section in `verit_next_steps.md`.

---

## Notes on Layer 2a Scores

Layer 2a (semantic/RRF) scored 0.000 in the original benchmark because the
RRF threshold of 0.02 was so permissive that everything passed — it never
contributed a SUSPICIOUS verdict independently. This will likely remain true
in the new benchmark. Layer 2a's role is now a cheap domain pre-filter before
the LLM call, not an independent verdict signal. Document this in the results
write-up rather than treating it as a failure.
