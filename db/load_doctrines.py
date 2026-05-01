"""
db/load_doctrines.py

Step 5.3b — Load Doctrine ontology into Neo4j.

Reads the mapped case doctrines and creates:
1. (:Doctrine) nodes for each unique doctrine
2. (c:Case)-[:APPLIES_DOCTRINE]->(d:Doctrine) relationships

Run:
  python -m db.load_doctrines
"""

import logging
import pandas as pd
from pathlib import Path
import sys
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_PATH = Path(PROCESSED_DIR) / "case_doctrines.parquet"

def _load_doctrines(tx, df: pd.DataFrame) -> None:
    # First, collect all unique doctrine IDs to create the Doctrine nodes
    unique_doctrines = set()
    for doctrines in df["doctrine_ids"]:
        unique_doctrines.update(doctrines)
        
    logger.info(f"Creating/merging {len(unique_doctrines)} Doctrine nodes...")
    for doc_id in unique_doctrines:
        # Create user-friendly display name (e.g. "terry_stop" -> "Terry Stop")
        name = " ".join([word.capitalize() for word in doc_id.split("_")])
        tx.run(
            """
            MERGE (d:Doctrine {id: $id})
            SET d.name = $name
            """,
            id=doc_id, name=name
        )

    # Now create the relationships
    logger.info("Creating APPLIES_DOCTRINE relationships in batches...")
    
    # Flatten the dataframe into (case_id, doctrine_id) tuples
    relationships = []
    for _, row in df.iterrows():
        case_id = int(row["case_id"])
        for doc_id in row["doctrine_ids"]:
            relationships.append({"case_id": case_id, "doc_id": doc_id})
            
    # Unwind in batches for efficiency
    batch_size = 1000
    for i in range(0, len(relationships), batch_size):
        batch = relationships[i:i + batch_size]
        result = tx.run(
            """
            UNWIND $batch AS rel
            MATCH (c:Case {id: rel.case_id})
            MATCH (d:Doctrine {id: rel.doc_id})
            MERGE (c)-[:APPLIES_DOCTRINE]->(d)
            RETURN count(*) AS added
            """,
            batch=batch
        )
        record = result.single()
        added = record["added"] if record else 0
        logger.info(f"  Processed batch {i}-{i+len(batch)-1}: {added} relationships added")

def main() -> None:
    if not INPUT_PATH.exists():
        logger.error(f"File not found: {INPUT_PATH}. Run classify_doctrines.py first.")
        return
        
    logger.info(f"Loading doctrine mappings from {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    
    logger.info("Connecting to Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            session.execute_write(_load_doctrines, df)
    finally:
        driver.close()
        
    logger.info("Successfully loaded Doctrine ontology into Neo4j.")

if __name__ == "__main__":
    main()
