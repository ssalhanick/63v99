"""
benchmark/generate_benchmark.py

Build a balanced 200-citation benchmark dataset for evaluating the Verit
hallucination detector in Week 8.

Split:
    100 real citations  — sampled from corpus, stratified by court_id and year
    100 hallucinated    — 3 subtypes:
        Type A (~33) — fully fabricated via Claude API
        Type B (~34) — real case with corrupted year or court_id
        Type C (~33) — plausible but nonexistent, via Claude API

Output:
    benchmark/benchmark.json

Usage:
    python -m benchmark.generate_benchmark

Notes:
    - Requires Neo4j running (docker-compose up -d)
    - Requires ANTHROPIC_API_KEY in .env
    - EyeCite used to extract citation strings from plain_text
    - 2026 cases excluded from sampling (too few)
    - State courts capped at MAX_STATE_COURT_SHARE of real sample
"""

import json
import os
import random
import re
import sys
import time

import anthropic
import pandas as pd
from eyecite import get_citations
from eyecite.models import FullCaseCitation

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    BENCHMARK_DIR,
    PROCESSED_DIR,
)
from db.neo4j_client import driver

# ── Sampling constants ────────────────────────────────────────────────────────
TOTAL_REAL          = 100
TOTAL_HALLUCINATED  = 100
TYPE_A_COUNT        = 33   # fabricated entirely
TYPE_B_COUNT        = 34   # real case, corrupted details
TYPE_C_COUNT        = 33   # plausible but nonexistent

# Federal circuit court_id prefixes — used for stratification and Type B swaps
FEDERAL_CIRCUITS = ["ca1", "ca2", "ca3", "ca4", "ca5", "ca6",
                    "ca7", "ca8", "ca9", "ca10", "ca11", "cadc"]

# Cap state court representation in real sample to avoid skewing the benchmark
MAX_STATE_COURT_SHARE = 0.4   # at most 40% of real citations from state courts

# Exclude 2026 — too few cases for reliable sampling
EXCLUDE_YEARS = {"2026"}

# Context window around a citation — characters before/after in plain_text
CONTEXT_WINDOW = 600


# Regex for well-formed reporter citations — used to filter Type B candidates
# Matches: 923 F.3d 1027 | 392 U.S. 1 | 284 N.E.2d 612
REPORTER_PATTERN = re.compile(
    r"^\d{1,4}\s+[A-Z][A-Za-z0-9.]*(?:\s?[0-9a-z]+)?\s+\d{1,4}$"
)

# Claude API settings
MAX_TOKENS      = 4096
RETRY_DELAY     = 2   # seconds between retries on API error
MAX_RETRIES     = 3

# Year corruption range for Type B
YEAR_SHIFT_RANGE = [-3, -2, -1, 1, 2, 3]   # never 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_corpus() -> pd.DataFrame:
    """Load cases_enriched.parquet and filter out 2026 cases."""
    path = os.path.join(PROCESSED_DIR, "cases_enriched.parquet")
    df = pd.read_parquet(path)
    df["year"] = df["date_filed"].str[:4]
    df = df[~df["year"].isin(EXCLUDE_YEARS)]
    df = df[df["plain_text"].notna() & (df["plain_text"].str.len() > 200)]
    return df.reset_index(drop=True)


def case_exists_in_neo4j(driver, case_id: int) -> bool:
    """Layer 1 existence check — confirm case node is in Neo4j."""
    with driver.session() as session:
        result = session.run(
            "MATCH (c:Case {id: $id}) RETURN c LIMIT 1",
            id=case_id
        )
        return result.single() is not None


def extract_citations_from_text(plain_text: str) -> list[dict]:
    """
    Run EyeCite over plain_text and return structured citation objects.
    Only returns FullCaseCitation instances (not statutory or unknown).
    Each result: {citation_string, reporter, volume, page}
    """
    found = get_citations(plain_text)
    results = []
    for c in found:
        if not isinstance(c, FullCaseCitation):
            continue
        try:
            citation_string = c.corrected_citation()
        except Exception:
            continue
        results.append({
            "citation_string": citation_string,
            "reporter": getattr(c.token, "reporter", None),
            "volume":   getattr(c.token, "volume", None),
            "page":     getattr(c.token, "page", None),
        })
    return results


def get_context_around_citation(plain_text: str, citation_string: str) -> str | None:
    """
    Find the first occurrence of citation_string in plain_text and return
    a CONTEXT_WINDOW character window centered on it.
    Returns None if citation not found.
    """
    idx = plain_text.find(citation_string)
    if idx == -1:
        return None
    start = max(0, idx - CONTEXT_WINDOW // 2)
    end   = min(len(plain_text), idx + len(citation_string) + CONTEXT_WINDOW // 2)
    return plain_text[start:end].strip()

def is_clean_reporter_citation(citation_string: str) -> bool:
    """Return True only for well-formed volume-reporter-page citations."""
    return bool(REPORTER_PATTERN.match(citation_string.strip()))

def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Stratified sample across court_id and year.
    Caps state court representation at MAX_STATE_COURT_SHARE.
    """
    df = df.copy()
    df["is_federal"] = df["court_id"].apply(
        lambda c: any(c.startswith(f) for f in FEDERAL_CIRCUITS)
    )

    max_state = int(n * MAX_STATE_COURT_SHARE)
    max_fed   = n - max_state

    federal_df = df[df["is_federal"]]
    state_df   = df[~df["is_federal"]]

    def sample_stratum(pool: pd.DataFrame, target: int) -> pd.DataFrame:
        if len(pool) <= target:
            return pool
        # Proportional stratification by year
        pool = pool.copy()
        pool["stratum"] = pool["year"]
        counts = pool["stratum"].value_counts(normalize=True) * target
        counts = counts.round().astype(int)
        # Adjust rounding error
        diff = target - counts.sum()
        if diff != 0:
            counts.iloc[0] += diff
        sampled = []
        for stratum, count in counts.items():
            stratum_pool = pool[pool["stratum"] == stratum]
            k = min(count, len(stratum_pool))
            sampled.append(stratum_pool.sample(k, random_state=42))
        return pd.concat(sampled)

    fed_sample   = sample_stratum(federal_df, max_fed)
    state_sample = sample_stratum(state_df,   max_state)
    combined     = pd.concat([fed_sample, state_sample])

    # Top up if we came up short
    if len(combined) < n:
        remaining = df[~df.index.isin(combined.index)]
        shortfall = n - len(combined)
        top_up    = remaining.sample(min(shortfall, len(remaining)), random_state=42)
        combined  = pd.concat([combined, top_up])

    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


# ── Real citation sampling ────────────────────────────────────────────────────

def build_real_citations(df: pd.DataFrame, driver, target: int) -> list[dict]:
    """
    Extract real citations from corpus plain_text via EyeCite.
    Confirm each cited case exists in Neo4j before including.
    Returns list of benchmark records with label=REAL.
    """
    print(f"\n[Real] Targeting {target} real citations...")
    pool    = stratified_sample(df, min(target * 6, len(df)))  # oversample, then filter
    records = []
    seen_citations = set()

    for _, row in pool.iterrows():
        if len(records) >= target:
            break

        plain_text = row["plain_text"]
        citations  = extract_citations_from_text(plain_text)

        for cit in citations:
            if len(records) >= target:
                break

            cs = cit["citation_string"]
            
            if not is_clean_reporter_citation(cs):
                continue

            if cs in seen_citations:
                continue

            context = get_context_around_citation(plain_text, cs)
            if not context:
                continue

            # We don't always have the cited case_id from EyeCite alone —
            # record the source case_id and note the citation is from its text.
            # Layer 1 will re-verify at benchmark eval time.
            seen_citations.add(cs)
            records.append({
                "citation":  cs,
                "context":   context,
                "label":     "REAL",
                "subtype":   None,
                "case_id":   int(row["case_id"]),   # source case, not cited case
                "court_id":  row["court_id"],
                "year":      row["year"],
            })
            print(f"  [Real] {len(records):>3}/{target}  {cs}")

    print(f"[Real] Collected {len(records)} real citations.")
    return records


# ── Claude API helpers ────────────────────────────────────────────────────────

def call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    """Call Claude API with retry logic. Returns response text."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"  [API] Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    raise RuntimeError("Claude API failed after max retries.")


def parse_json_from_response(text: str) -> list[dict]:
    """Strip markdown fences and parse JSON array from Claude response."""
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


# ── Type A — Fully fabricated ─────────────────────────────────────────────────

TYPE_A_PROMPT = """You are helping build a legal hallucination detection benchmark.

Generate {n} completely fabricated Fourth Amendment case citations that do NOT exist.
They should look realistic — plausible case names, realistic reporters, years between
2010 and 2025, and realistic courts (federal circuits or state appellate courts).

Each citation must be paired with a short context paragraph (3-5 sentences) of
AI-generated legal text that cites it, as if an AI wrote a legal brief using this
fake case. The context should discuss a Fourth Amendment issue (search, seizure,
warrant, probable cause, stop and frisk, etc.).

Return ONLY a JSON array with no preamble or markdown. Each object:
{{
  "citation": "United States v. Fake, 923 F.3d 100 (9th Cir. 2019)",
  "context": "...paragraph citing this case..."
}}

Important:
- Do NOT use real case names (Terry, Katz, Mapp, Leon, Gates, Carpenter, Riley, etc.)
- Invent both the case name AND the citation string
- Make the reporter and page numbers look realistic
- Vary circuits and years across the {n} citations
- Each citation must be unique
"""

def build_type_a(client: anthropic.Anthropic, target: int) -> list[dict]:
    """Generate fully fabricated citations via Claude API in batches."""
    print(f"\n[Type A] Generating {target} fabricated citations...")
    batch_size = 10
    records = []
    while len(records) < target:
        remaining = target - len(records)
        n = min(batch_size, remaining)
        prompt = TYPE_A_PROMPT.format(n=n)
        raw = call_claude(client, prompt)
        try:
            parsed = parse_json_from_response(raw)
        except json.JSONDecodeError:
            print(f"  [Type A] JSON parse failed on batch, skipping...")
            continue
        for item in parsed[:n]:
            records.append({
                "citation": item["citation"],
                "context":  item["context"],
                "label":    "HALLUCINATED",
                "subtype":  "A",
                "case_id":  None,
                "court_id": None,
                "year":     None,
            })
            print(f"  [Type A] {len(records):>3}/{target}  {item['citation']}")
    print(f"[Type A] Generated {len(records)} fabricated citations.")
    return records


# ── Type B — Real case, corrupted details ────────────────────────────────────

def corrupt_year(year_str: str) -> str:
    year  = int(year_str)
    shift = random.choice(YEAR_SHIFT_RANGE)
    return str(max(2005, min(2025, year + shift)))

def corrupt_court(court_id: str, all_court_ids: list[str]) -> str:
    """
    Swap court_id. If federal, swap to another federal circuit.
    If state, swap to a different state court from the corpus.
    """
    is_fed = any(court_id.startswith(f) for f in FEDERAL_CIRCUITS)
    if is_fed:
        options = [c for c in FEDERAL_CIRCUITS if c != court_id]
        return random.choice(options)
    else:
        state_options = [c for c in all_court_ids
                         if c != court_id
                         and not any(c.startswith(f) for f in FEDERAL_CIRCUITS)]
        if state_options:
            return random.choice(state_options)
        return random.choice(FEDERAL_CIRCUITS)   # fallback to federal

def build_type_b(df: pd.DataFrame, target: int) -> list[dict]:
    """
    Corrupt real corpus cases — shift year or swap court_id.
    Pull citation strings and context from plain_text via EyeCite.
    """
    print(f"\n[Type B] Generating {target} corrupted real citations...")
    all_court_ids = df["court_id"].unique().tolist()
    pool    = df.sample(min(len(df), TYPE_B_COUNT * 20), random_state=99).reset_index(drop=True)
    records = []
    seen    = set()

    for _, row in pool.iterrows():
        if len(records) >= target:
            break

        plain_text = row["plain_text"]
        citations  = extract_citations_from_text(plain_text)

        for cit in citations:
            if len(records) >= target:
                break

            cs = cit["citation_string"]

            if not is_clean_reporter_citation(cs):
                continue
            if cs in seen:
                continue

            context = get_context_around_citation(plain_text, cs)
            if not context:
                continue

            # Corrupt either year or court — alternate to get variety
            if len(records) % 2 == 0:
                corruption      = "year"
                corrupted_year  = corrupt_year(row["year"])
                corrupted_court = row["court_id"]
            else:
                corruption      = "court"
                corrupted_year  = row["year"]
                corrupted_court = corrupt_court(row["court_id"], all_court_ids)

            seen.add(cs)
            records.append({
                "citation":          f"{cs} ({corrupted_court})" if corruption == "court" else f"{cs} [{corrupted_year}]",
                "context":           context,
                "label":             "HALLUCINATED",
                "subtype":           "B",
                "case_id":           int(row["case_id"]),
                "court_id":          row["court_id"],
                "year":              row["year"],
                "original_citation": cs,
                "corrupted_year":    corrupted_year  if corruption == "year"  else None,
                "corrupted_court":   corrupted_court if corruption == "court" else None,
                "corruption_type":   corruption,
            })
            print(f"  [Type B] {len(records):>3}/{target}  {records[-1]['citation']}  [{corruption} corrupted]")

    print(f"[Type B] Generated {len(records)} corrupted citations.")
    return records


# ── Type C — Plausible but nonexistent ───────────────────────────────────────

TYPE_C_PROMPT = """You are helping build a legal hallucination detection benchmark.

Generate {n} plausible-sounding but completely nonexistent Fourth Amendment case
citations. These should look like real cases an AI might confabulate — realistic
names, correct citation format, plausible courts and years (2010–2025) — but they
must NOT actually exist.

Each citation must be paired with a short context paragraph (3-5 sentences) of
AI-generated legal text that cites it, discussing a Fourth Amendment issue.

Return ONLY a JSON array with no preamble or markdown. Each object:
{{
  "citation": "United States v. Torres, 847 F.3d 214 (9th Cir. 2017)",
  "context": "...paragraph citing this case..."
}}

Important:
- Use common surnames (Torres, Rivera, Johnson, Williams, Martinez, etc.)
- Format must look exactly like a real federal or state appellate citation
- Do NOT use real landmark case names
- Vary circuits, reporters, and years
- Each citation must be unique
- These should be the hardest subtype to detect — maximally plausible
"""

def build_type_c(client: anthropic.Anthropic, target: int) -> list[dict]:
    """Generate plausible-but-nonexistent citations via Claude API in batches."""
    print(f"\n[Type C] Generating {target} plausible nonexistent citations...")
    batch_size = 10
    records = []
    while len(records) < target:
        remaining = target - len(records)
        n = min(batch_size, remaining)
        prompt = TYPE_C_PROMPT.format(n=n)
        raw = call_claude(client, prompt)
        try:
            parsed = parse_json_from_response(raw)
        except json.JSONDecodeError:
            print(f"  [Type C] JSON parse failed on batch, skipping...")
            continue
        for item in parsed[:n]:
            records.append({
                "citation": item["citation"],
                "context":  item["context"],
                "label":    "HALLUCINATED",
                "subtype":  "C",
                "case_id":  None,
                "court_id": None,
                "year":     None,
            })
            print(f"  [Type C] {len(records):>3}/{target}  {item['citation']}")
    print(f"[Type C] Generated {len(records)} plausible nonexistent citations.")
    return records


# ── Assembly + output ─────────────────────────────────────────────────────────

def save_benchmark(records: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved {len(records)} records to {output_path}")


def print_summary(records: list[dict]) -> None:
    total       = len(records)
    real        = sum(1 for r in records if r["label"] == "REAL")
    hall        = sum(1 for r in records if r["label"] == "HALLUCINATED")
    type_a      = sum(1 for r in records if r.get("subtype") == "A")
    type_b      = sum(1 for r in records if r.get("subtype") == "B")
    type_c      = sum(1 for r in records if r.get("subtype") == "C")
    print("\n─── Benchmark Summary ───────────────────────────────")
    print(f"  Total:          {total}")
    print(f"  Real:           {real}")
    print(f"  Hallucinated:   {hall}")
    print(f"    Type A (fabricated):    {type_a}")
    print(f"    Type B (corrupted):     {type_b}")
    print(f"    Type C (plausible):     {type_c}")
    print("─────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)

    output_path = os.path.join(BENCHMARK_DIR, "benchmark.json")

    print("Loading corpus...")
    df = load_corpus()
    print(f"  {len(df)} cases loaded after filtering.")

    print("Connecting to Neo4j...")
    # driver imported directly from db.neo4j_client

    print("Initializing Anthropic client...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Real citations ────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(BENCHMARK_DIR, "checkpoint_real.json")
    if os.path.exists(checkpoint_path):
        print("Loading real citations from checkpoint...")
        with open(checkpoint_path, "r") as f:
            real_records = json.load(f)
        print(f"  {len(real_records)} real citations loaded from checkpoint.")
    else:
        real_records = build_real_citations(df, driver, TOTAL_REAL)
        os.makedirs(BENCHMARK_DIR, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(real_records, f, indent=2)
        print(f"  Checkpoint saved to {checkpoint_path}")

    # ── Hallucinated citations ────────────────────────────────────────────────
    # ── Hallucinated citations ────────────────────────────────────────────────────
    checkpoint_a = os.path.join(BENCHMARK_DIR, "checkpoint_type_a.json")
    if os.path.exists(checkpoint_a):
        print("Loading Type A from checkpoint...")
        with open(checkpoint_a) as f:
            type_a_records = json.load(f)
        print(f"  {len(type_a_records)} Type A records loaded.")
    else:
        type_a_records = build_type_a(client, TYPE_A_COUNT)
        with open(checkpoint_a, "w") as f:
            json.dump(type_a_records, f, indent=2)
        print(f"  Type A checkpoint saved.")

    type_b_records = build_type_b(df, TYPE_B_COUNT)

    checkpoint_c = os.path.join(BENCHMARK_DIR, "checkpoint_type_c.json")
    if os.path.exists(checkpoint_c):
        print("Loading Type C from checkpoint...")
        with open(checkpoint_c) as f:
            type_c_records = json.load(f)
        print(f"  {len(type_c_records)} Type C records loaded.")
    else:
        type_c_records = build_type_c(client, TYPE_C_COUNT)
        with open(checkpoint_c, "w") as f:
            json.dump(type_c_records, f, indent=2)
        print(f"  Type C checkpoint saved.")

    # ── Assemble + shuffle ────────────────────────────────────────────────────
    all_records = real_records + type_a_records + type_b_records + type_c_records
    random.shuffle(all_records)

    for i, record in enumerate(all_records):
        record["benchmark_id"] = i

    print_summary(all_records)
    save_benchmark(all_records, output_path)


if __name__ == "__main__":
    main()