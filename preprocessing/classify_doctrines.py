"""
preprocessing/classify_doctrines.py

Step 5.3a — Doctrine keyword classification.

Scans the first 3,000 words of each case's plain_text for Fourth Amendment
doctrine keywords and assigns one or more doctrine IDs.

Input:  data/processed/cases_enriched.parquet
Output: data/processed/case_doctrines.parquet  (case_id, doctrine_ids)

Run:
  python -m preprocessing.classify_doctrines
"""

import logging
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH  = Path(PROCESSED_DIR) / "cases_enriched.parquet"
OUTPUT_PATH = Path(PROCESSED_DIR) / "case_doctrines.parquet"

# Fourth Amendment doctrine taxonomy and keywords
DOCTRINE_KEYWORDS = {
    "terry_stop":            ["terry stop", "investigatory stop", "articulable suspicion", "brief detention", "terry v. ohio"],
    "exclusionary_rule":     ["exclusionary rule", "fruit of the poisonous tree", "suppression of evidence"],
    "plain_view":            ["plain view", "inadvertent discovery", "immediately apparent"],
    "exigent_circumstances": ["exigent circumstances", "hot pursuit", "imminent destruction", "emergency exception"],
    "automobile_exception":  ["automobile exception", "vehicle search", "inventory search", "carroll doctrine"],
    "consent_search":        ["consent to search", "voluntary consent", "third party consent", "third-party consent"],
    "probable_cause":        ["probable cause", "reasonable belief", "totality of the circumstances", "totality of circumstances"],
    "good_faith":            ["good faith exception", "leon", "reasonably relied", "good-faith exception"],
    "curtilage":             ["curtilage", "open fields", "immediate vicinity"],
    "search_incident":       ["search incident to arrest", "wingspan", "chimel", "grab area"],
}

def classify_text(text: str) -> list[str]:
    """Return a list of doctrine IDs found in the text."""
    if not text or not isinstance(text, str):
        return []
        
    text_lower = text.lower()
    # Limit search to first ~3,000 words to catch main syllabi/introductions 
    # and speed up processing.
    text_slice = " ".join(text_lower.split()[:3000])
    
    doctrines_found = []
    for doctrine_id, keywords in DOCTRINE_KEYWORDS.items():
        if any(kw in text_slice for kw in keywords):
            doctrines_found.append(doctrine_id)
            
    return doctrines_found

def main() -> None:
    if not INPUT_PATH.exists():
        logger.error(f"Input file not found: {INPUT_PATH}")
        return
        
    logger.info(f"Loading cases from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH, columns=["case_id", "plain_text"])
    logger.info(f"Loaded {len(df)} cases.")
    
    records = []
    for _, row in df.iterrows():
        case_id = row["case_id"]
        doctrines = classify_text(row["plain_text"])
        if doctrines:
            records.append({
                "case_id": int(case_id),
                "doctrine_ids": doctrines
            })
            
    out_df = pd.DataFrame(records)
    logger.info(f"Found doctrines for {len(out_df)} cases.")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved mapped doctrines to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
