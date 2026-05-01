"""
detector/existence_check.py

Layer 1 — Existence Check.

Given a case_id (CourtListener cluster ID), query Neo4j to determine whether
a Case node with that ID exists in the graph.

If the node does not exist → immediate HALLUCINATED verdict.
If the node exists → proceed to Layers 2 and 3.

Note: a case_id of None (unresolved by EyeCite/CourtListener) is treated as
non-existent and returns False immediately without querying Neo4j.
"""

import logging
from typing import Optional

from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

logger = logging.getLogger(__name__)


def check_existence(case_id: Optional[int], driver=None) -> tuple[bool, Optional[str]]:
    """
    Return (True, node_name) if a Case node with the given case_id exists in Neo4j.
    Return (False, None) if not found or if case_id is None.

    The node_name is used by the name check (Step 1.3) to compare party names
    in the citation string against the stored case name without a second query.

    Args:
        case_id: CourtListener cluster ID (integer). None → returns (False, None).
        driver:  optional Neo4j driver instance. If not provided, a new
                 driver is created and closed after the query. Pass a driver
                 when calling from the pipeline to reuse the connection.

    Returns:
        (True, node_name)  → node exists in graph (proceed to Layers 2 and 3)
        (False, None)      → node not found (HALLUCINATED verdict)
    """
    if case_id is None:
        logger.debug("case_id is None — skipping Neo4j lookup, returning False")
        return False, None

    close_after = driver is None
    if close_after:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (c:Case {id: $id}) RETURN c.id AS id, c.name AS name LIMIT 1",
                id=case_id,
            )
            record = result.single()
            exists    = record is not None
            node_name = str(record["name"]) if exists and record["name"] else None

        if exists:
            logger.info("Layer 1 PASS — case_id %d found in Neo4j (name=%r)", case_id, node_name)
        else:
            logger.info("Layer 1 FAIL — case_id %d not found in Neo4j", case_id)

        return exists, node_name

    except Exception as e:
        logger.error("Neo4j query error for case_id %s: %s", case_id, e)
        return False, None

    finally:
        if close_after:
            driver.close()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Terry v. Ohio — should exist (loaded as landmark)
    # Sokolow       — may or may not be in corpus
    # 9999999       — definitely does not exist

    test_cases = [
        (107729, "Terry v. Ohio (landmark — should exist)"),
        (112239, "United States v. Sokolow (corpus case)"),
        (4617563, "Amgen Inc. v. Sandoz Inc. (out-of-domain — should not exist)"),
        (9999999, "Fabricated ID — should not exist"),
        (None,   "None case_id — should return False without querying"),
    ]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    print(f"\n{'Case':<55} {'Exists':>6}")
    print("-" * 63)
    for case_id, label in test_cases:
        exists, node_name = check_existence(case_id, driver=driver)
        print(f"{label:<55} {str(exists):>6}  {node_name or ''}")

    driver.close()