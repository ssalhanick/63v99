import json
import os
from config import ROOT_DIR

RAW_DIR = f'{ROOT_DIR}/data/raw'

batch_files = [
    "batch_2015_present.json",
    "batch_2010_2015.json",
    "batch_2001_2010.json"
]

for filename in batch_files:
    path = os.path.join(RAW_DIR, filename)

    if not os.path.exists(path):
        print(f'{filename}: NOT FOUND\n')
        continue
    
    with open(path) as f:
        cases = json.load(f)

    has_plain_text  = 0
    has_download_url = 0
    has_neither     = 0
    has_both        = 0

    for case in cases:
        opinions = case.get("opinions", [])
        if not opinions:
            has_neither += 1
            continue

        opinion = opinions[0]
        plain   = bool(opinion.get("plain_text", "").strip())
        pdf     = bool((opinion.get("download_url") or "").strip())

        if plain and pdf:
            has_both += 1
        elif plain:
            has_plain_text += 1
        elif pdf:
            has_download_url += 1
        else:
            has_neither += 1

    print(f"--- {filename} ---")
    print(f"Total cases:       {len(cases)}")
    print(f"Has plain text:       {has_plain_text}")
    print(f"Has download_url:  {has_download_url}")
    print(f"Has both:          {has_both}")
    print(f"Has neither:       {has_neither}")
    print()