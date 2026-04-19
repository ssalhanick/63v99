import pandas as pd

df = pd.read_parquet("data/processed/cases_enriched.parquet")
print(df.columns.tolist())

df2 = pd.read_parquet("data/processed/cases_cleaned.parquet")
print(df2.columns.tolist())