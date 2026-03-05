import json
import os
from config import RAW_DIR

batch_files = [
    "enriched_2015_present.json",
    "enriched_2010_2015.json",
]

all_cases = []
seen_ids  = set()

for filename in batch_files:
    path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(path):
        print(f"Skipping {filename} — not found")
        continue

    with open(path) as f:
        cases = json.load(f)

    before = len(all_cases)

    for case in cases:
        case_id = case["case_id"]
        if case_id not in seen_ids:
            seen_ids.add(case_id)
            all_cases.append(case)

    added = len(all_cases) - before
    dupes = len(cases) - added
    print(f"{filename}: {len(cases)} cases, {added} added, {dupes} duplicates removed")

output_path = os.path.join(RAW_DIR, "cases_merged.json")
with open(output_path, "w") as f:
    json.dump(all_cases, f, indent=2)

print(f"\nTotal unique cases: {len(all_cases)}")
print(f"Saved to: {output_path}")