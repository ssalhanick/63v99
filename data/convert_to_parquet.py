import pandas as pd
import json
import os
from config import RAW_DIR, PROCESSED_DIR

print("Loading merged JSON...")
with open(os.path.join(RAW_DIR, "cases_merged.json")) as f:
    cases = json.load(f)

df = pd.DataFrame(cases)
print(f"Loaded {len(df)} cases with columns: {list(df.columns)}")

# Serialize list columns for Parquet compatibility
df["opinions_cited"] = df["opinions_cited"].apply(json.dumps)
df["citations"]      = df["citations"].apply(json.dumps)

os.makedirs(PROCESSED_DIR, exist_ok=True)
output_path = os.path.join(PROCESSED_DIR, "cases_enriched.parquet")
df.to_parquet(output_path, index=False)

print(f"\nDone. Saved {len(df)} cases to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
print(f"Columns: {list(df.columns)}")