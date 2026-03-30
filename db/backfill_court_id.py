"""
db/backfill_court_id.py
Week 8 — Backfill court_id from cases_enriched.parquet into Neo4j Case nodes.

Run from project root:
    python -m db.backfill_court_id
"""

import sys
import logging
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 200

CYPHER = """
UNWIND $rows AS row
MATCH (c:Case {id: row.case_id})
SET c.court_id = row.court_id
"""


def main():
    df = pd.read_parquet(
        Path(config.PROCESSED_DIR) / "cases_enriched.parquet",
        columns=["case_id", "court_id"],
    )
    df = df.dropna(subset=["court_id"])
    rows = df.to_dict("records")
    logger.info("Loaded %d rows from parquet", len(rows))

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )

    updated = 0
    try:
        with driver.session() as session:
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i : i + BATCH_SIZE]
                result = session.run(CYPHER, rows=batch)
                summary = result.consume()
                updated += summary.counters.properties_set
                logger.info(
                    "  batch %d/%d — properties set so far: %d",
                    i // BATCH_SIZE + 1,
                    -(-len(rows) // BATCH_SIZE),
                    updated,
                )
    finally:
        driver.close()

    logger.info("Done — %d court_id properties written to Neo4j", updated)


if __name__ == "__main__":
    main()