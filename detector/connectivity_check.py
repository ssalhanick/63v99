"""
detector/connectivity_check.py

Layer 3 — Citation Connectivity Check.

Measures how well-connected a case is within the corpus citation network.
A real case cited in Fourth Amendment opinions will share citations with
other corpus cases. A hallucinated case has no footprint — zero or near-zero
shared citations.

Metric: citation density
  Count of distinct corpus cases (stub=false) that share at least one
  outbound citation with the target case.

  MATCH (target:Case {id: $id})-[:CITES]->(shared)<-[:CITES]-(corpus:Case {stub: false})
  RETURN count(DISTINCT shared) AS density

Verdict contribution:
  density >= CITATION_DENSITY_THRESHOLD → connected (REAL signal)
  density <  CITATION_DENSITY_THRESHOLD → not connected (SUSPICIOUS signal)

Note: case_id=None returns density=0, is_connected=False immediately.

Phase 4.1 addition:
  Also fetches `c.pagerank` — the GDS PageRank authority score written by
  db/compute_pagerank.py. Returns None if PageRank has not yet been computed.
"""

import logging
from typing import Optional

from neo4j import GraphDatabase

from config import (
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    CITATION_DENSITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class ConnectivityResult:
    density_score:  int             # number of shared citations with corpus cases
    is_connected:   bool            # density_score >= CITATION_DENSITY_THRESHOLD
    pagerank_score: Optional[float] # GDS PageRank authority score (None until computed)


# ---------------------------------------------------------------------------
# Main check
# ---------------------------------------------------------------------------

def check_connectivity(case_id: Optional[int], driver=None) -> ConnectivityResult:
    """
    Query Neo4j for citation density of the target case against the corpus.

    Args:
        case_id: CourtListener cluster ID. None → returns density=0 immediately.
        driver:  optional Neo4j driver. If not provided, one is created and
                 closed after the query. Pass a driver from the pipeline to
                 reuse the connection.

    Returns:
        ConnectivityResult with density_score and is_connected flag.
    """
    if case_id is None:
        logger.debug("case_id is None — skipping connectivity check")
        return ConnectivityResult(density_score=0, is_connected=False, pagerank_score=None)

    close_after = driver is None
    if close_after:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # Single query: fetch density + pagerank together to avoid
            # a second round-trip.  pagerank may be null if
            # db/compute_pagerank.py hasn't been run yet.
            result = session.run(
                """
                MATCH (target:Case {id: $id})-[:CITES]->(shared)
                      <-[:CITES]-(corpus:Case {stub: false})
                WITH count(DISTINCT shared) AS density
                MATCH (c:Case {id: $id})
                RETURN density, c.pagerank AS pagerank
                """,
                id=case_id,
            )
            record = result.single()
            density  = int(record["density"])            if record else 0
            pagerank = float(record["pagerank"])         if (record and record["pagerank"] is not None) else None

        is_connected = density >= CITATION_DENSITY_THRESHOLD

        if is_connected:
            logger.info(
                "Layer 3 PASS — case_id %d: density=%d pagerank=%s (threshold=%d)",
                case_id, density, f"{pagerank:.6f}" if pagerank is not None else "N/A",
                CITATION_DENSITY_THRESHOLD,
            )
        else:
            logger.info(
                "Layer 3 FAIL — case_id %d: density=%d pagerank=%s (threshold=%d)",
                case_id, density, f"{pagerank:.6f}" if pagerank is not None else "N/A",
                CITATION_DENSITY_THRESHOLD,
            )

        return ConnectivityResult(
            density_score=density,
            is_connected=is_connected,
            pagerank_score=pagerank,
        )

    except Exception as e:
        logger.error("Neo4j connectivity query error for case_id %s: %s", case_id, e)
        return ConnectivityResult(density_score=0, is_connected=False, pagerank_score=None)

    finally:
        if close_after:
            driver.close()


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Terry v. Ohio (107729) — landmark, likely isolated from corpus (by design)
    # Pick a corpus case that should have real connectivity
    # Fabricated ID — should have density 0
    test_cases = [
        (107729,  "Terry v. Ohio (landmark)"),
        (4617563, "Amgen v. Sandoz (out-of-domain)"),
        (9999999, "Fabricated ID"),
        (None,    "None case_id"),
    ]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Also test a few real corpus cases by sampling from Neo4j
    print("Sampling real corpus cases from Neo4j...")
    with driver.session() as session:
        result = session.run(
            "MATCH (c:Case {stub: false}) RETURN c.id AS id, c.name AS name LIMIT 5"
        )
        corpus_samples = [(r["id"], r["name"] or "Unknown") for r in result]

    all_cases = corpus_samples + test_cases

    print(f"\n{'Case':<55} {'Density':>7}  {'Connected':>9}")
    print("-" * 75)

    for case_id, label in all_cases:
        result = check_connectivity(case_id, driver=driver)
        print(
            f"{str(label):<55} {result.density_score:>7}  {str(result.is_connected):>9}"
        )

    driver.close()