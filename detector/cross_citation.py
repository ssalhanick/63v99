"""
detector/cross_citation.py

Phase 4, Steps 4.2 & 4.3 — Cross-citation graph analysis.

Provides two pairwise signals across all citations that passed Layer 1
within a single document:

  1. Jaccard similarity (Step 4.2)
     How many cases does pair (A, B) co-cite?
     jaccard(A, B) = |outNeighbors(A) ∩ outNeighbors(B)|
                     ─────────────────────────────────────
                     |outNeighbors(A) ∪ outNeighbors(B)|

     High Jaccard → the two cases are from the same legal neighbourhood
     Low Jaccard  → the citations are from different doctrinal areas
                    (weak SUSPICIOUS signal for co-citation incoherence)

  2. Shortest-path hop distance (Step 4.3)
     What is the minimum number of CITES hops between A and B (up to 5)?

     Short hop distance  → legally connected
     No path within 5   → disconnected (SUSPICIOUS signal)

Both functions accept a list of case IDs that have already been confirmed
to exist in Neo4j (i.e., all passed Layer 1). IDs that are None are skipped.

Usage (called from detector/pipeline.py after the per-citation loop):
    from detector.cross_citation import compute_cross_citation_signals
    signals = compute_cross_citation_signals(case_ids, driver=driver)
    # signals[case_id] = CrossCitationSignal(mean_jaccard, min_hop)
"""

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from detector.doctrine_check import get_doctrines, get_shared_doctrines

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CrossCitationSignal:
    """
    Aggregated cross-citation signals for a single case_id within one document.

    Attributes:
        mean_jaccard:   Mean Jaccard similarity of this case against all other
                        confirmed-real citations in the same document.
                        None if fewer than 2 confirmed cases in the document.
        min_hop_distance: Minimum hop distance from this case to any other
                          confirmed-real citation in the same document via CITES.
                          None if no path found within MAX_HOPS, or <2 cases.
        pair_count:     Number of other cases this case was paired with.
        has_doctrines:  True if the case has at least one doctrine label.
        mean_shared_doctrines: Mean number of shared doctrines with co-cited cases.
    """
    mean_jaccard:          Optional[float]
    min_hop_distance:      Optional[int]
    pair_count:            int
    has_doctrines:         bool
    mean_shared_doctrines: Optional[float]


# ---------------------------------------------------------------------------
# Jaccard (Step 4.2)
# ---------------------------------------------------------------------------

_JACCARD_QUERY = """
MATCH (a:Case {id: $id_a})-[:CITES]->(shared:Case)<-[:CITES]-(b:Case {id: $id_b})
WITH count(DISTINCT shared) AS intersection
MATCH (a:Case {id: $id_a})-[:CITES]->(out_a:Case)
WITH intersection, collect(DISTINCT out_a.id) AS a_neighbors
MATCH (b:Case {id: $id_b})-[:CITES]->(out_b:Case)
WITH intersection, a_neighbors, collect(DISTINCT out_b.id) AS b_neighbors
WITH intersection,
     [x IN a_neighbors WHERE x IN b_neighbors | x] AS shared_ids,
     a_neighbors + [x IN b_neighbors WHERE NOT x IN a_neighbors | x] AS union_ids
RETURN intersection,
       size(union_ids) AS union_size
"""


def _compute_jaccard_pair(session, id_a: int, id_b: int) -> float:
    """
    Return Jaccard similarity for the outbound citation neighbourhoods of
    case A and case B.

    Returns 0.0 if either case has no outbound citations.
    """
    result = session.run(_JACCARD_QUERY, id_a=id_a, id_b=id_b)
    record = result.single()
    if record is None:
        return 0.0
    intersection = int(record["intersection"])
    union_size   = int(record["union_size"])
    if union_size == 0:
        return 0.0
    return intersection / union_size


# ---------------------------------------------------------------------------
# Shortest path (Step 4.3)
# ---------------------------------------------------------------------------

MAX_HOPS = 5   # search up to 5 hops; no path beyond this → treat as disconnected

_SHORTEST_PATH_QUERY = f"""
MATCH path = shortestPath(
  (a:Case {{id: $id_a}})-[:CITES*..{MAX_HOPS}]-(b:Case {{id: $id_b}})
)
RETURN length(path) AS hops
"""


def _compute_shortest_path(session, id_a: int, id_b: int) -> Optional[int]:
    """
    Return the shortest-path hop distance between case A and case B,
    traversing CITES edges in either direction, up to MAX_HOPS hops.

    Returns None if no path is found within MAX_HOPS.
    """
    result = session.run(_SHORTEST_PATH_QUERY, id_a=id_a, id_b=id_b)
    record = result.single()
    if record is None:
        return None
    return int(record["hops"])


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def compute_cross_citation_signals(
    case_ids: list[Optional[int]],
    driver=None,
) -> dict[int, CrossCitationSignal]:
    """
    Compute pairwise Jaccard and shortest-path signals for all confirmed-real
    case IDs in a single document, then aggregate per case_id.

    Args:
        case_ids: List of case IDs that passed Layer 1.
                  None entries are silently skipped.
        driver:   Optional Neo4j driver. Created and closed internally if None.

    Returns:
        dict mapping case_id → CrossCitationSignal.
        If fewer than 2 non-None IDs are provided, every entry will have
        mean_jaccard=None and min_hop_distance=None.
    """
    # Filter to confirmed (non-None) IDs, deduplicate while preserving order
    seen: set[int] = set()
    valid_ids: list[int] = []
    for cid in case_ids:
        if cid is not None and cid not in seen:
            seen.add(cid)
            valid_ids.append(cid)

    # Build empty result for every valid ID
    result: dict[int, CrossCitationSignal] = {
        cid: CrossCitationSignal(
            mean_jaccard=None,
            min_hop_distance=None,
            pair_count=0,
            has_doctrines=False,
            mean_shared_doctrines=None,
        )
        for cid in valid_ids
    }

    if len(valid_ids) == 0:
        logger.debug(
            "cross_citation: no valid case_ids — skipping pairwise analysis"
        )
        return result

    close_after = driver is None
    if close_after:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # First, quickly check single-node doctrines
            has_doctrines_dict: dict[int, bool] = {
                cid: len(get_doctrines(session, cid)) > 0 
                for cid in valid_ids
            }
            
            # If fewer than 2 cases, we just populate the has_doctrines and return
            if len(valid_ids) < 2:
                for cid in valid_ids:
                    result[cid].has_doctrines = has_doctrines_dict[cid]
                return result

            # Accumulate per-node scores across all pairs
            jaccard_sums:   dict[int, float]         = {cid: 0.0  for cid in valid_ids}
            hop_mins:       dict[int, Optional[int]] = {cid: None for cid in valid_ids}
            pair_counts:    dict[int, int]            = {cid: 0    for cid in valid_ids}
            shared_doctrines_sums: dict[int, int]     = {cid: 0    for cid in valid_ids}

            for id_a, id_b in combinations(valid_ids, 2):
                # --- Jaccard ---
                jac = _compute_jaccard_pair(session, id_a, id_b)
                jaccard_sums[id_a] += jac
                jaccard_sums[id_b] += jac
                pair_counts[id_a]  += 1
                pair_counts[id_b]  += 1

                # --- Shortest path ---
                hops = _compute_shortest_path(session, id_a, id_b)
                if hops is not None:
                    hop_mins[id_a] = (
                        min(hop_mins[id_a], hops)
                        if hop_mins[id_a] is not None else hops
                    )
                    hop_mins[id_b] = (
                        min(hop_mins[id_b], hops)
                        if hop_mins[id_b] is not None else hops
                    )

                logger.debug(
                    "pair (%d, %d): jaccard=%.4f  hops=%s",
                    id_a, id_b, jac, hops,
                )
                
                # --- Shared Doctrines ---
                shared_docs = len(get_shared_doctrines(session, id_a, id_b))
                shared_doctrines_sums[id_a] += shared_docs
                shared_doctrines_sums[id_b] += shared_docs

            # Aggregate into CrossCitationSignal per case
            for cid in valid_ids:
                n = pair_counts[cid]
                mean_jac = jaccard_sums[cid] / n if n > 0 else None
                mean_shared = shared_doctrines_sums[cid] / n if n > 0 else None
                result[cid] = CrossCitationSignal(
                    mean_jaccard=mean_jac,
                    min_hop_distance=hop_mins[cid],
                    pair_count=n,
                    has_doctrines=has_doctrines_dict[cid],
                    mean_shared_doctrines=mean_shared,
                )
                logger.info(
                    "cross_citation case_id=%d: mean_jaccard=%.4f  min_hop=%s  pairs=%d  has_docs=%s  mean_shared=%.2f",
                    cid,
                    mean_jac if mean_jac is not None else 0.0,
                    hop_mins[cid],
                    n,
                    has_doctrines_dict[cid],
                    mean_shared if mean_shared is not None else 0.0,
                )

    except Exception as e:
        logger.error("cross_citation query error: %s", e)
        # Return the empty-signal result rather than crashing the pipeline

    finally:
        if close_after:
            driver.close()

    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    # Fourth Amendment landmarks — should be closely related
    # Terry v. Ohio, Katz v. United States, Mapp v. Ohio, United States v. Leon
    test_ids = [107729, 107564, 106285, 111262]

    print(f"\nRunning cross-citation analysis on {len(test_ids)} landmark cases …")
    signals = compute_cross_citation_signals(test_ids)

    print(f"\n{'case_id':<12} {'mean_jaccard':>14}  {'min_hop':>8}  {'pairs':>6}  {'has_docs':>9}  {'mean_shared':>12}")
    print("-" * 75)
    for cid, sig in signals.items():
        jac = f"{sig.mean_jaccard:.4f}" if sig.mean_jaccard is not None else "N/A"
        hop = str(sig.min_hop_distance) if sig.min_hop_distance is not None else "None"
        shared = f"{sig.mean_shared_doctrines:.2f}" if sig.mean_shared_doctrines is not None else "N/A"
        print(f"{cid:<12} {jac:>14}  {hop:>8}  {sig.pair_count:>6}  {str(sig.has_doctrines):>9}  {shared:>12}")

    # Also test with a fabricated ID — should return 0 Jaccard + no path
    print("\nAdding fabricated case ID 9999999 …")
    mixed_ids = [107729, 107564, 9999999]
    signals2 = compute_cross_citation_signals(mixed_ids)
    for cid, sig in signals2.items():
        jac = f"{sig.mean_jaccard:.4f}" if sig.mean_jaccard is not None else "N/A"
        hop = str(sig.min_hop_distance) if sig.min_hop_distance is not None else "None"
        print(f"  {cid}: jaccard={jac}  min_hop={hop}  has_docs={sig.has_doctrines}")
