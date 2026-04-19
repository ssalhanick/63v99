import pandas as pd
import ast

df = pd.read_parquet("data/processed/cases_enriched.parquet", 
                     columns=["case_id", "case_name", "citations"])

target_ids = [9352236, 4557235, 9912503, 9403162, 9942139]

for cid in target_ids:
    row = df[df["case_id"] == cid]
    if not row.empty:
        citations = row["citations"].values[0]
        print(f"{cid} | {row['case_name'].values[0]}")
        print(f"  citations: {citations}\n")