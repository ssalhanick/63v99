"""
benchmark/expand_benchmark.py

Expand benchmark.json from 200 → 500 entries by generating 300 new citations.

New split (300 additions):
    150 real citations  — same stratified EyeCite sampling as generate_benchmark.py
    150 hallucinated:
        Type A (~40) — fully fabricated via Claude API
        Type B (~70) — real case, corrupted details  ← weighted heavier
        Type C (~40) — plausible but nonexistent via Claude API

Appends to existing benchmark.json (does not regenerate existing entries).
Resets benchmark/split_indices.json so cross_validate.py uses the full 500.

Usage:
    python -m benchmark.expand_benchmark

Notes:
    - Requires Neo4j running (docker-compose up -d)
    - Requires ANTHROPIC_API_KEY in .env
    - Skips citations already present in benchmark.json (dedup on citation string)
    - Checkpoints each subtype to avoid re-calling the API on crash
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import anthropic
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    BENCHMARK_DIR,
    PROCESSED_DIR,
)
from db.neo4j_client import driver

# Import generation helpers from original script
from benchmark.generate_benchmark import (
    load_corpus,
    stratified_sample,
    extract_citations_from_text,
    get_context_around_citation,
    is_clean_reporter_citation,
    call_claude,
    parse_json_from_response,
    corrupt_year,
    corrupt_court,
    FEDERAL_CIRCUITS,
    CONTEXT_WINDOW,
    YEAR_SHIFT_RANGE,
)

# ── Expansion targets ─────────────────────────────────────────────────────────
NEW_REAL    = 150
NEW_TYPE_A  = 40    # fabricated
NEW_TYPE_B  = 70    # corrupted — weighted heavier, hardest subtype
NEW_TYPE_C  = 40    # plausible nonexistent

BENCHMARK_PATH   = Path(BENCHMARK_DIR) / "benchmark.json"
SPLIT_CACHE_PATH = Path(BENCHMARK_DIR) / "split_indices.json"

# Checkpoint paths for expansion run (separate from original checkpoints)
CKPT_REAL_EXP = Path(BENCHMARK_DIR) / "expand_checkpoint_real.json"
CKPT_A_EXP    = Path(BENCHMARK_DIR) / "expand_checkpoint_type_a.json"
CKPT_B_EXP    = Path(BENCHMARK_DIR) / "expand_checkpoint_type_b.json"
CKPT_C_EXP    = Path(BENCHMARK_DIR) / "expand_checkpoint_type_c.json"

# ── Claude prompts (same structure as generate_benchmark.py) ──────────────────

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
- These should be maximally plausible — hardest subtype to detect
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_existing_benchmark() -> tuple[list[dict], set[str]]:
    """Load existing benchmark.json and return records + set of seen citation strings."""
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    seen = {r["citation"] for r in records}
    print(f"Loaded {len(records)} existing benchmark entries ({len(seen)} unique citations)")
    return records, seen


def save_checkpoint(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path: Path) -> list[dict] | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"  Loaded {len(records)} records from checkpoint {path.name}")
        return records
    return None


# ── Real citation expansion ───────────────────────────────────────────────────

def build_real_expansion(
    df: pd.DataFrame,
    target: int,
    seen_citations: set[str],
) -> list[dict]:
    """Same logic as generate_benchmark.build_real_citations but skips seen citations."""
    ckpt = load_checkpoint(CKPT_REAL_EXP)
    if ckpt is not None:
        return ckpt

    print(f"\n[Real] Targeting {target} new real citations...")
    pool    = stratified_sample(df, min(target * 8, len(df)))
    records = []
    local_seen = set(seen_citations)

    for _, row in pool.iterrows():
        if len(records) >= target:
            break

        citations = extract_citations_from_text(row["plain_text"])

        for cit in citations:
            if len(records) >= target:
                break

            cs = cit["citation_string"]

            if not is_clean_reporter_citation(cs):
                continue
            if cs in local_seen:
                continue

            context = get_context_around_citation(row["plain_text"], cs)
            if not context:
                continue

            local_seen.add(cs)
            records.append({
                "citation": cs,
                "context":  context,
                "label":    "REAL",
                "subtype":  None,
                "case_id":  int(row["case_id"]),
                "court_id": row["court_id"],
                "year":     row["year"],
            })
            print(f"  [Real] {len(records):>3}/{target}  {cs}")

    print(f"[Real] Collected {len(records)} new real citations.")
    save_checkpoint(records, CKPT_REAL_EXP)
    return records


# ── Type A expansion ──────────────────────────────────────────────────────────

def build_type_a_expansion(
    client: anthropic.Anthropic,
    target: int,
    seen_citations: set[str],
) -> list[dict]:
    ckpt = load_checkpoint(CKPT_A_EXP)
    if ckpt is not None:
        return ckpt

    print(f"\n[Type A] Generating {target} new fabricated citations...")
    batch_size = 10
    records    = []
    local_seen = set(seen_citations)

    while len(records) < target:
        remaining = target - len(records)
        n = min(batch_size, remaining)
        prompt = TYPE_A_PROMPT.format(n=n)
        raw = call_claude(client, prompt)
        try:
            parsed = parse_json_from_response(raw)
        except json.JSONDecodeError:
            print("  [Type A] JSON parse failed, retrying...")
            continue

        for item in parsed[:n]:
            cs = item.get("citation", "").strip()
            if not cs or cs in local_seen:
                continue
            local_seen.add(cs)
            records.append({
                "citation": cs,
                "context":  item.get("context", ""),
                "label":    "HALLUCINATED",
                "subtype":  "A",
                "case_id":  None,
                "court_id": None,
                "year":     None,
            })
            print(f"  [Type A] {len(records):>3}/{target}  {cs}")
            if len(records) >= target:
                break

    print(f"[Type A] Generated {len(records)} new fabricated citations.")
    save_checkpoint(records, CKPT_A_EXP)
    return records


# ── Type B expansion ──────────────────────────────────────────────────────────

def build_type_b_expansion(
    df: pd.DataFrame,
    target: int,
    seen_citations: set[str],
) -> list[dict]:
    ckpt = load_checkpoint(CKPT_B_EXP)
    if ckpt is not None:
        return ckpt

    print(f"\n[Type B] Generating {target} new corrupted real citations...")
    all_court_ids = df["court_id"].unique().tolist()
    pool    = df.sample(min(len(df), target * 20), random_state=7).reset_index(drop=True)
    records = []
    local_seen = set(seen_citations)

    # Weight distribution: 60% court corruption, 40% year corruption
    # Court corruption is harder for Layer 4 to catch when court_id is ambiguous
    for idx, (_, row) in enumerate(pool.iterrows()):
        if len(records) >= target:
            break

        citations = extract_citations_from_text(row["plain_text"])

        for cit in citations:
            if len(records) >= target:
                break

            cs = cit["citation_string"]

            if not is_clean_reporter_citation(cs):
                continue
            if cs in local_seen:
                continue

            context = get_context_around_citation(row["plain_text"], cs)
            if not context:
                continue

            # 60% court, 40% year — more court corruptions in expansion
            if len(records) % 5 in (0, 1, 2):   # 3/5 = 60% court
                corruption      = "court"
                corrupted_year  = row["year"]
                corrupted_court = corrupt_court(row["court_id"], all_court_ids)
            else:
                corruption      = "year"
                corrupted_year  = corrupt_year(row["year"])
                corrupted_court = row["court_id"]

            local_seen.add(cs)
            citation_string = (
                f"{cs} ({corrupted_court})"
                if corruption == "court"
                else f"{cs} [{corrupted_year}]"
            )
            records.append({
                "citation":          citation_string,
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
            print(
                f"  [Type B] {len(records):>3}/{target}  "
                f"{citation_string}  [{corruption} corrupted]"
            )

    print(f"[Type B] Generated {len(records)} new corrupted citations.")
    save_checkpoint(records, CKPT_B_EXP)
    return records

# ── Type B (proposition) expansion ───────────────────────────────────────────

CKPT_B_PROP_EXP = Path(BENCHMARK_DIR) / "expand_checkpoint_type_b_proposition.json"

TYPE_B_PROP_PROMPT = """You are building a legal hallucination detection benchmark.

You will be given a real Fourth Amendment case citation and an excerpt from the opinion.
Your job is to write a plausible-sounding but factually WRONG proposition for this case.

Citation: {citation}

Opinion excerpt:
{excerpt}

Write a context paragraph (3-5 sentences) that:
- Cites this exact citation string naturally (as a real legal brief would)
- Uses correct legal vocabulary and sounds authoritative
- But materially misstates the holding — do one of:
    * Reverse the ruling (held constitutional → held unconstitutional, or vice versa)
    * Attribute a different legal standard (e.g. say "probable cause" when it was "reasonable suspicion")
    * Describe facts or a procedural outcome not present in the opinion

Return ONLY a JSON object, no preamble or markdown:
{{"citation": "{citation}", "context": "<wrong proposition paragraph>"}}
"""

def build_type_b_proposition_expansion(
    client: anthropic.Anthropic,
    df: pd.DataFrame,
    target: int,
    seen_citations: set[str],
) -> list[dict]:
    ckpt = load_checkpoint(CKPT_B_PROP_EXP)
    if ckpt is not None:
        return ckpt

    print(f"\n[Type B-Prop] Generating {target} proposition-hallucinated citations...")

    # Seed cases from context doc — high connectivity, easy to verify real holding
    PRIORITY_CITATIONS = {
        "92 F.4th 1279",
        "33 F.4th 296",
        "157 N.E.3d 406",
        "232 N.E.3d 419",
        "392 U.S. 1",
        "389 U.S. 347",
    }

    pool = df.sample(min(len(df), target * 25), random_state=13).reset_index(drop=True)
    records = []
    local_seen = set(seen_citations)

    for _, row in pool.iterrows():
        if len(records) >= target:
            break

        citations = extract_citations_from_text(row["plain_text"])

        for cit in citations:
            if len(records) >= target:
                break

            cs = cit["citation_string"]

            if not is_clean_reporter_citation(cs):
                continue
            if cs in local_seen:
                continue

            # Pull excerpt — first 1500 chars of plain_text as holding context
            excerpt = row["plain_text"][:1500].strip()
            if not excerpt:
                continue

            prompt = TYPE_B_PROP_PROMPT.format(
                citation=cs,
                excerpt=excerpt,
            )

            try:
                raw = call_claude(client, prompt)
                parsed = parse_json_from_response(raw)
                # parse_json_from_response returns list — handle both list and dict
                if isinstance(parsed, list):
                    parsed = parsed[0]
                wrong_context = parsed.get("context", "").strip()
            except Exception as e:
                print(f"  [Type B-Prop] API/parse error for {cs}: {e}")
                continue

            if not wrong_context:
                continue

            local_seen.add(cs)
            records.append({
                "citation":        cs,          # bare reporter string — no corruption
                "context":         wrong_context,
                "label":           "SUSPICIOUS",
                "subtype":         "B",
                "case_id":         int(row["case_id"]),
                "court_id":        row["court_id"],
                "year":            row["year"],
                "corruption_type": "proposition",
            })
            print(f"  [Type B-Prop] {len(records):>3}/{target}  {cs}")

    print(f"[Type B-Prop] Generated {len(records)} proposition-hallucinated citations.")
    save_checkpoint(records, CKPT_B_PROP_EXP)
    return records

# ── Type C expansion ──────────────────────────────────────────────────────────

def build_type_c_expansion(
    client: anthropic.Anthropic,
    target: int,
    seen_citations: set[str],
) -> list[dict]:
    ckpt = load_checkpoint(CKPT_C_EXP)
    if ckpt is not None:
        return ckpt

    print(f"\n[Type C] Generating {target} new plausible nonexistent citations...")
    batch_size = 10
    records    = []
    local_seen = set(seen_citations)

    while len(records) < target:
        remaining = target - len(records)
        n = min(batch_size, remaining)
        prompt = TYPE_C_PROMPT.format(n=n)
        raw = call_claude(client, prompt)
        try:
            parsed = parse_json_from_response(raw)
        except json.JSONDecodeError:
            print("  [Type C] JSON parse failed, retrying...")
            continue

        for item in parsed[:n]:
            cs = item.get("citation", "").strip()
            if not cs or cs in local_seen:
                continue
            local_seen.add(cs)
            records.append({
                "citation": cs,
                "context":  item.get("context", ""),
                "label":    "HALLUCINATED",
                "subtype":  "C",
                "case_id":  None,
                "court_id": None,
                "year":     None,
            })
            print(f"  [Type C] {len(records):>3}/{target}  {cs}")
            if len(records) >= target:
                break

    print(f"[Type C] Generated {len(records)} new plausible citations.")
    save_checkpoint(records, CKPT_C_EXP)
    return records


# ── Assembly ──────────────────────────────────────────────────────────────────

def print_summary(records: list[dict]) -> None:
    total  = len(records)
    real   = sum(1 for r in records if r["label"] == "REAL")
    hall   = sum(1 for r in records if r["label"] == "HALLUCINATED")
    type_a = sum(1 for r in records if r.get("subtype") == "A")
    type_b = sum(1 for r in records if r.get("subtype") == "B")
    type_c = sum(1 for r in records if r.get("subtype") == "C")
    print("\n─── Expanded Benchmark Summary ──────────────────────")
    print(f"  Total:                  {total}")
    print(f"  Real:                   {real}")
    print(f"  Hallucinated:           {hall}")
    print(f"    Type A (fabricated):  {type_a}")
    print(f"    Type B (corrupted):   {type_b}  ← weighted heavier")
    print(f"    Type C (plausible):   {type_c}")
    print("─────────────────────────────────────────────────────")


def main():
    random.seed(99)   # different seed from original to get different samples

    existing_records, seen_citations = load_existing_benchmark()

    print("Loading corpus...")
    df = load_corpus()
    print(f"  {len(df)} cases available.")

    print("Initializing Anthropic client...")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Generate new entries ──────────────────────────────────────────────────
    new_real   = build_real_expansion(df, NEW_REAL,   seen_citations)
    new_type_a = build_type_a_expansion(client, NEW_TYPE_A, seen_citations)
    new_type_b_meta = build_type_b_expansion(df, NEW_TYPE_B, seen_citations)
    new_type_b_prop = build_type_b_proposition_expansion(client, df, NEW_TYPE_B//2, seen_citations)
    new_type_c = build_type_c_expansion(client, NEW_TYPE_C, seen_citations)

    new_entries = new_real + new_type_a + new_type_b_meta + new_type_b_prop + new_type_c
    random.shuffle(new_entries)

    # ── Merge with existing ───────────────────────────────────────────────────
    all_records = existing_records + new_entries

    # Reassign benchmark_ids sequentially across full 500
    for i, record in enumerate(all_records):
        record["benchmark_id"] = i

    print_summary(all_records)

    # ── Save expanded benchmark ───────────────────────────────────────────────
    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\n✅  Saved {len(all_records)} entries → {BENCHMARK_PATH}")

    # ── Reset split cache so CV uses full 500 ─────────────────────────────────
    if SPLIT_CACHE_PATH.exists():
        SPLIT_CACHE_PATH.unlink()
        print(f"✅  Split cache cleared → {SPLIT_CACHE_PATH}")
        print("    Run  python -m benchmark.cross_validate  to start CV on full 500.")

    # Clean up expansion checkpoints
    for ckpt in [CKPT_REAL_EXP, CKPT_A_EXP, CKPT_B_EXP, CKPT_C_EXP]:
        if ckpt.exists():
            ckpt.unlink()
    print("✅  Expansion checkpoints cleaned up.")


if __name__ == "__main__":
    main()