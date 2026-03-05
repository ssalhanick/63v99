import json
import os
from config import ROOT_DIR

with open(os.path.join(ROOT_DIR, "data", "raw", "cases_sample.json")) as f:
    cases = json.load(f)

print(f"Total cases: {len(cases)}")
print(f"\n--- First Case Fields ---")
first = cases[0]
for key, value in first.items():
    if isinstance(value, str):
        print(f"{key}: {value[:80]}")  # trim long strings
    else:
        print(f"{key}: {value}")