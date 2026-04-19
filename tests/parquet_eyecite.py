import pandas as pd
import ast
from detector.eyecite_parser import _CORPUS_INDEX

df = pd.read_parquet("data/processed/cases_enriched.parquet", 
                     columns=["case_id", "case_name", "citations"])

def parse_citations(val):
    if val is None or isinstance(val, float):
        return []
    if isinstance(val, list):
        # handle both ["33 F.4th 296"] and [{"cite": "33 F.4th 296", ...}]
        result = []
        for item in val:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # try common key names
                for key in ["cite", "citation", "reporter", "volume", "text"]:
                    if key in item:
                        result.append(item[key])
                        break
        return result
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parse_citations(parsed)  # recurse to handle nested structure
        except:
            return []
    return []

resolvable = []
unresolvable = []

for _, row in df.iterrows():
    citations = parse_citations(row["citations"])
    if not citations:
        unresolvable.append((row["case_id"], row["case_name"], "no citation string"))
        continue
    
    resolved = any(_CORPUS_INDEX.get(c) for c in citations)
    if resolved:
        resolvable.append((row["case_id"], row["case_name"], citations))
    else:
        unresolvable.append((row["case_id"], row["case_name"], citations))

for _, row in df.iterrows():
    val = row["citations"]
    if isinstance(val, list) and val and isinstance(val[0], dict):
        print("Dict citation example:", val[0])
        break

print(f"Resolvable:   {len(resolvable)} / {len(df)}")
print(f"Unresolvable: {len(unresolvable)} / {len(df)}")
print("\nSample unresolvable (have citation strings but don't resolve):")
has_string = [(i, n, c) for i, n, c in unresolvable if c != "no citation string"]
for row in has_string[:10]:
    print(f"  {row[0]} | {row[1]} | {row[2]}")