import pandas as pd

df = pd.read_parquet("data/processed/cases_enriched.parquet", 
                     columns=["case_id", "plain_text"])

row = df[df["case_id"] == 9942139]
text = row["plain_text"].values[0]

# Rough token estimate: ~0.75 tokens per word
words = len(text.split())
tokens = int(words * 0.75)
print(f"Opinion words: {words:,}")
print(f"Estimated tokens: {tokens:,}")