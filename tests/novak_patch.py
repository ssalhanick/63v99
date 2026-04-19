import pandas as pd

df = pd.read_parquet("data/processed/cases_enriched.parquet")
mask = df["case_id"] == 6336455
df.loc[mask, "citations"] = str(["33 F.4th 296"])
df.to_parquet("data/processed/cases_enriched.parquet", index=False)
print("Patched.")