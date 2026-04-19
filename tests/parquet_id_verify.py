import pandas as pd
df = pd.read_parquet("data/processed/cases_enriched.parquet", columns=["case_id", "case_name", "citations"])
novak = df[df["case_id"] == 6336455]
print(novak[["case_id", "case_name", "citations"]].values)