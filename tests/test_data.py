import os
import json
from config import RAW_DIR, PROCESSED_DIR
import pandas as pd

# -------------------------------------------------------
# Raw cases
# -------------------------------------------------------

def test_raw_cases_exist():
    path = os.path.join(RAW_DIR, "batch_2015_present.json")
    assert os.path.exists(path), "batch_2015_present.json not found"

def test_raw_cases_count(raw_cases):
    assert len(raw_cases) == 1500, f"Expected 1500 cases, got {len(raw_cases)}"

def test_raw_cases_have_required_fields(raw_cases):
    required = ["caseName", "dateFiled", "court_id", "opinions", "cluster_id"]
    for field in required:
        assert field in raw_cases[0], f"Missing field: {field}"

def test_raw_cases_correct_court(raw_cases):
    ca9_cases = [c for c in raw_cases if c["court_id"] == "ca9"]
    assert len(ca9_cases) > 0, "No ca9 cases found"

# -------------------------------------------------------
# Merged enriched cases
# -------------------------------------------------------

def test_merged_cases_exist():
    path = os.path.join(RAW_DIR, "cases_merged.json")
    assert os.path.exists(path), "cases_merged.json not found"

def test_merged_cases_count(merged_cases):
    assert len(merged_cases) == 1353, f"Expected 1353 cases, got {len(merged_cases)}"

def test_merged_cases_have_full_text(merged_cases):
    has_text = [c for c in merged_cases if len(c.get("plain_text", "")) > 100]
    assert len(has_text) > 1200, f"Too few cases with text: {len(has_text)}"

def test_merged_cases_have_citations(merged_cases):
    has_citations = [c for c in merged_cases if len(c.get("opinions_cited", [])) > 0]
    assert len(has_citations) > 0, "No cases have citations"

def test_merged_cases_have_required_fields(merged_cases):
    required = ["case_id", "case_name", "date_filed", "plain_text", "opinions_cited"]
    for field in required:
        assert field in merged_cases[0], f"Missing field: {field}"

# -------------------------------------------------------
# Parquet
# -------------------------------------------------------

def test_parquet_exists():
    path = os.path.join(PROCESSED_DIR, "cases_enriched.parquet")
    assert os.path.exists(path), "cases_enriched.parquet not found"

def test_parquet_loads_correctly():
    path = os.path.join(PROCESSED_DIR, "cases_enriched.parquet")
    df = pd.read_parquet(path)
    assert len(df) > 0, "Parquet file is empty"
    assert "plain_text" in df.columns, "Missing plain_text column"
    assert "opinions_cited" in df.columns, "Missing opinions_cited column"

def test_parquet_no_null_case_ids():
    path = os.path.join(PROCESSED_DIR, "cases_enriched.parquet")
    df = pd.read_parquet(path)
    assert df["case_id"].isnull().sum() == 0, "Null case_ids found in parquet"

def test_parquet_row_count():
    path = os.path.join(PROCESSED_DIR, "cases_enriched.parquet")
    df = pd.read_parquet(path)
    assert len(df) == 1353, f"Expected 1353 rows, got {len(df)}"