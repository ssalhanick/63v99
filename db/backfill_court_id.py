"""
db/backfill_court_id.py
Week 8 — Backfill court_id from cases_enriched.parquet into Neo4j Case nodes.

Run from project root:
    python -m db.backfill_court_id

Note: Landmark nodes are NOT in cases_enriched.parquet — they are loaded via
db/fetch_landmarks.py which now writes court_id directly. If any non-stub nodes
still show no court_id after this script, check for landmarks missing court_id
and patch them manually (all SCOTUS landmarks → court_id = 'scotus').
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

    # Post-run verification — warn if any non-stub nodes still lack court_id
    with driver.session() as session:
        result = session.run(
            "MATCH (c:Case {stub: false}) WHERE c.court_id IS NULL RETURN count(c) AS n"
        )
        remaining = result.single()["n"]
        if remaining == 0:
            logger.info("Verification passed — zero non-stub nodes missing court_id ✅")
        else:
            logger.warning(
                "⚠️  %d non-stub node(s) still missing court_id after backfill. "
                "These are likely landmark nodes not in the parquet. "
                "Check fetch_landmarks.py or patch manually.",
                remaining,
            )


if __name__ == "__main__":
    main()