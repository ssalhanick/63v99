import pandas as pd

df = pd.read_parquet("data/processed/cases_enriched.parquet")
row = df[df["case_id"] == 9942139]
text = row["plain_text"].values[0]
print(text[:3000])