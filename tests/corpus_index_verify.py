# from detector.eyecite_parser import _CORPUS_INDEX
# print(_CORPUS_INDEX.get("969 F.3d 285"))

# import pandas as pd
# df = pd.read_parquet("data/processed/cases_enriched.parquet", columns=["case_id", "case_name", "citations"])
# novak = df[df["case_name"].str.contains("Novak", case=False, na=False)]
# print(novak[["case_id", "case_name", "citations"]])

import requests
from config import COURTLISTENER_TOKEN

headers = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}
resp = requests.get(
    "https://www.courtlistener.com/api/rest/v4/clusters/",
    headers=headers,
    params={"case_name_full": "Novak", "court": "ca6"}
)
import json
print(json.dumps(resp.json(), indent=2))