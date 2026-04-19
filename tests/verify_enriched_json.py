import json, os

# Find your enriched batch files
raw_dir = "data/raw"
enriched_files = [f for f in os.listdir(raw_dir) if f.startswith("enriched_")]
print(enriched_files)

# Inspect the first case
with open(os.path.join(raw_dir, enriched_files[0])) as f:
    cases = json.load(f)

for case in cases[:10]:
    print(f"case_id: {case['case_id']} | cluster_id: {case['cluster_id']} | name: {case['case_name']}")