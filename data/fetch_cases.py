import requests
import json
import os
from config import COURTLISTENER_TOKEN

TOKEN = COURTLISTENER_TOKEN
BASE_URL = "https://www.courtlistener.com/api/rest/v4"

headers = {"Authorization": f"Token {TOKEN}"}

params = {
    "q": "fourth amendment search seizure",
    "court": "ca9",
    "filed_after": "2000-01-01",
    "order_by": "score desc",
    "page_size": 100,
    "format": "json"
}

cases = []
url = f"{BASE_URL}/search/"

while url and len(cases) < 500:
    response = requests.get(url, headers=headers, params=params if "?" not in url else {})
    data = response.json()
    print(data)  # add this line temporarily
    cases.extend(data["results"])
    url = data.get("next")
    print(f"Fetched {len(cases)} cases so far...")

with open("data/raw/cases_sample.json", "w") as f:
    json.dump(cases, f, indent=2)

print(f"Done. {len(cases)} cases saved to data/raw/cases_sample.json")