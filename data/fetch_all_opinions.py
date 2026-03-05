import requests
import json
import os
import time
import argparse
from tqdm import tqdm
from config import COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL, RAW_DIR

parser = argparse.ArgumentParser(description="Fetch full opinion text for a batch of cases")
parser.add_argument("--batch", type=str, required=True, help="Batch filename to enrich e.g. batch_2015_present.json")
args = parser.parse_args()

input_path  = os.path.join(RAW_DIR, args.batch)
output_name = args.batch.replace("batch_", "enriched_")
output_path = os.path.join(RAW_DIR, output_name)

TOKEN    = COURTLISTENER_TOKEN
BASE_URL = COURTLISTENER_BASE_URL
headers  = {"Authorization": f"Token {TOKEN}"}

with open(input_path) as f:
    cases = json.load(f)

print(f"Processing {len(cases)} cases from {args.batch}")
print(f"Output: {output_name}\n")

def fetch_with_retry(url, headers, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 429:
                print(f"\nRate limited — waiting 60 seconds...")
                time.sleep(60)
                continue
            return response
        except requests.exceptions.Timeout:
            wait = 10 * (attempt + 1)
            print(f"\nTimeout attempt {attempt + 1} — retrying in {wait}s...")
            time.sleep(wait)
        except requests.exceptions.ConnectionError:
            wait = 10 * (attempt + 1)
            print(f"\nConnection error attempt {attempt + 1} — retrying in {wait}s...")
            time.sleep(wait)
    return None

enriched_cases  = []
skipped_no_text = 0
skipped_no_opinion = 0

for case in tqdm(cases):
    try:
        if not case.get("opinions"):
            skipped_no_opinion += 1
            continue

        opinion_id = case["opinions"][0]["id"]
        url        = f"{BASE_URL}/opinions/{opinion_id}/"
        response   = fetch_with_retry(url, headers)

        if response is None:
            skipped_no_text += 1
            continue

        data       = response.json()
        plain_text = data.get("plain_text", "").strip()

        # Filter — only keep cases with real full text
        if len(plain_text) < 100:
            skipped_no_text += 1
            continue

        enriched = {
            "case_id":        opinion_id,
            "cluster_id":     case["cluster_id"],
            "case_name":      case["caseName"],
            "date_filed":     case["dateFiled"],
            "court_id":       case["court_id"],
            "citations":      case["citation"],
            "plain_text":     plain_text,
            "opinions_cited": data.get("opinions_cited", []),
            "cite_count":     case["citeCount"],
            "docket_number":  case["docketNumber"],
            "status":         case["status"],
        }

        enriched_cases.append(enriched)

        # Save progress after every case
        with open(output_path, "w") as f:
            json.dump(enriched_cases, f, indent=2)

        time.sleep(1)

    except Exception as e:
        print(f"\nError on {case.get('caseName')}: {e}")
        continue

print(f"\n--- Summary ---")
print(f"Total processed:     {len(cases)}")
print(f"Saved with text:     {len(enriched_cases)}")
print(f"Skipped (no text):   {skipped_no_text}")
print(f"Skipped (no opinion):{skipped_no_opinion}")
print(f"Output: {output_path}")