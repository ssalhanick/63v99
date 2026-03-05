import json
import os
from config import ROOT_DIR

with open(os.path.join(ROOT_DIR, "data","raw", "cases_enriched.json")) as f:
    cases = json.load(f)

no_text = [c for c in cases if len(c.get("plain_text", "")) <= 100]

print(f"Total cases: {len(cases)}")
print(f"Cases WITH text: {len(cases) - len(no_text)}")
print(f"Cases WITHOUT text: {len(no_text)}")

# Look at the first case with no text to understand why
print(f"\n--- Sample case with no text ---")
sample = no_text[0]
for key, value in sample.items():
    if isinstance(value, str):
        print(f"{key}: {value[:100]}")
    else:
        print(f"{key}: {value}")