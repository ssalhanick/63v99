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

# REPLACE WITH THIS
df.loc[missing_mask, "citations"] = df.loc[missing_mask, "case_id"].map(
    lambda cid: str(updates.get(cid, []))
)
df.to_parquet("data/processed/cases_enriched.parquet", index=False)