import requests
from config import COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL, LANDMARK_IDS

HEADERS = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}

for opinion_id in LANDMARK_IDS:
    url  = f"{COURTLISTENER_BASE_URL}/clusters/{opinion_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code == 200:
        data      = resp.json()
        name      = data.get("case_name", "Unknown")
        citations = data.get("citations", [])
        print(f"\n{opinion_id} | {name}")
        print(f"  citations: {citations}")
    else:
        print(f"{opinion_id} | ❌ status {resp.status_code}")