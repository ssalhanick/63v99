import requests
from config import COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL, LANDMARK_IDS

HEADERS = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}

for opinion_id in LANDMARK_IDS:
    url = f"{COURTLISTENER_BASE_URL}/opinions/{opinion_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    data = resp.json()
    text = data.get("plain_text", "").strip()
    print(f"{opinion_id} | chars={len(text)} | {'✅ has text' if len(text) > 100 else '❌ no text'}")

for opinion_id in LANDMARK_IDS:
    url = f"{COURTLISTENER_BASE_URL}/opinions/{opinion_id}/"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    data = resp.json()
    for field in ["html", "html_with_citations", "html_lawbox", "html_columbia"]:
        val = data.get(field, "")
        if val and len(val.strip()) > 100:
            print(f"{opinion_id} | {field} | chars={len(val)}")
            break
    else:
        print(f"{opinion_id} | ❌ no text in any field")