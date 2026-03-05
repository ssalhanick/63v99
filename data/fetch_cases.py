import requests
import json
import os
import argparse
from tqdm import tqdm
from config import COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL, ROOT_DIR

# -------------------------------------------------------
# Argument parsing
# -------------------------------------------------------
parser = argparse.ArgumentParser(description="Fetch 9th Circuit Fourth Amendment cases from CourtListener")
parser.add_argument("--after",  type=str, required=True,  help="Fetch cases filed after this date (YYYY-MM-DD)")
parser.add_argument("--before", type=str, required=False, help="Fetch cases filed before this date (YYYY-MM-DD)")
parser.add_argument("--limit",  type=int, default=500,    help="Maximum number of cases to fetch (default: 500)")
args = parser.parse_args()

# -------------------------------------------------------
# Build output filename from date range
# -------------------------------------------------------
if args.before:
    filename = f"batch_{args.after[:4]}_{args.before[:4]}.json"
else:
    filename = f"batch_{args.after[:4]}_present.json"

output_path = os.path.join(ROOT_DIR, "data", "raw", filename)

TOKEN = COURTLISTENER_TOKEN
BASE_URL = COURTLISTENER_BASE_URL

headers = {"Authorization": f"Token {TOKEN}"}

params = {
    "q": "fourth amendment search seizure warrant probable cause unreasonable privacy",
    "filed_after": args.after,
    "order_by": "score desc",
    "page_size": 100,
    "format": "json"
}

if args.before:
    params["filed_before"] = args.before

print(f"Fetching up to {args.limit} cases...")
print(f"Date range: {args.after} → {args.before or 'present'}")
print(f"Output file: {filename}\n")

cases = []
url = f"{BASE_URL}/search/"

with tqdm(total=args.limit) as pbar:
    while url and len(cases) < args.limit:
        response = requests.get(url, headers=headers, params=params if "?" not in url else {})
        data = response.json()
        batch = data.get("results", [])
        cases.extend(batch)
        pbar.update(len(batch))
        url = data.get("next")

cases = cases[:args.limit]

with open(output_path, "w") as f:
    json.dump(cases, f, indent=2)

print(f"Done. {len(cases)} cases saved to {output_path}")