"""
db/compute_pagerank.py

Phase 4, Step 4.1a — Compute and persist PageRank scores in Neo4j.

Uses NetworkX to compute PageRank from the CITES edge list pulled via Cypher,
then writes scores back to Case nodes as a `pagerank` property.

No Neo4j GDS plugin required — works with neo4j:5.15-community + APOC only.

Algorithm:
  1. Pull all CITES edges from Neo4j as (source_id, target_id) pairs.
  2. Build a directed NetworkX DiGraph.
  3. Run NetworkX PageRank (alpha=0.85, max_iter=100, convergence tol=1e-6).
  4. Write scores back to Neo4j in batches of 500 via parameterized UNWIND.

This is a one-time setup command, re-runnable at any time to refresh scores
after graph updates. The write is idempotent — re-running overwrites the
existing `pagerank` property.

Run from project root:
    python -m db.compute_pagerank

Verification:
    MATCH (c:Case {id: 107729}) RETURN c.pagerank  -- Terry v. Ohio should be high
    MATCH (c:Case) WHERE c.pagerank IS NOT NULL RETURN count(c)
"""

import logging
import sys
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# PageRank hyperparameters
DAMPING    = 0.85
MAX_ITER   = 100
TOL        = 1.0e-6
WRITE_BATCH = 500


# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------

def _fetch_edges(session) -> tuple[list[tuple[int, int]], set[int]]:
    """
    Pull all CITES edges from Neo4j.

    Returns:
        edges:    list of (source_id, target_id) integer tuples
        all_ids:  set of every node ID that appears in at least one edge
    """
    logger.info("Fetching CITES edges from Neo4j …")
    result = session.run(
        """
        MATCH (a:Case)-[:CITES]->(b:Case)
        RETURN a.id AS src, b.id AS tgt
        """
    )
    edges: list[tuple[int, int]] = []
    all_ids: set[int] = set()
    for record in result:
        src, tgt = record["src"], record["tgt"]
        if src is not None and tgt is not None:
            edges.append((int(src), int(tgt)))
            all_ids.add(int(src))
            all_ids.add(int(tgt))
    logger.info("Fetched %d edges, %d unique nodes", len(edges), len(all_ids))
    return edges, all_ids


def _fetch_all_case_ids(session) -> set[int]:
    """Fetch IDs of all Case nodes (including isolated nodes with no edges)."""
    result = session.run("MATCH (c:Case) WHERE c.id IS NOT NULL RETURN c.id AS id")
    return {int(r["id"]) for r in result}


# ---------------------------------------------------------------------------
# Pure-Python PageRank (no external dependency beyond stdlib)
# ---------------------------------------------------------------------------

def _compute_pagerank_python(
    edges: list[tuple[int, int]],
    all_node_ids: set[int],
    alpha: float = DAMPING,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
) -> dict[int, float]:
    """
    Power-iteration PageRank on a directed graph.

    Uses the standard random-surfer model:
      PR(v) = (1 - alpha) / N  +  alpha * sum(PR(u) / out_degree(u))  for u -> v

    Dangling nodes (zero out-degree) distribute their rank uniformly.

    Returns dict mapping node_id → pagerank score (scores sum to 1.0).
    """
    nodes = list(all_node_ids)
    N = len(nodes)
    if N == 0:
        return {}

    node_idx  = {nid: i for i, nid in enumerate(nodes)}

    # Build adjacency: in_edges[v] = list of (u, out_deg_u)
    out_deg: dict[int, int] = defaultdict(int)
    in_adj:  dict[int, list[int]] = defaultdict(list)   # in_adj[v] = [u, ...]
    for src, tgt in edges:
        if src in node_idx and tgt in node_idx:
            out_deg[src] += 1
            in_adj[tgt].append(src)

    dangling = {n for n in nodes if out_deg[n] == 0}

    # Initialize uniform
    pr = {n: 1.0 / N for n in nodes}
    base = (1.0 - alpha) / N

    for iteration in range(max_iter):
        dangling_sum = alpha * sum(pr[n] for n in dangling) / N
        new_pr: dict[int, float] = {}

        for v in nodes:
            rank = base + dangling_sum
            for u in in_adj[v]:
                rank += alpha * pr[u] / out_deg[u]
            new_pr[v] = rank

        # Check convergence (L1 norm)
        err = sum(abs(new_pr[n] - pr[n]) for n in nodes)
        pr = new_pr
        if err < N * tol:
            logger.info("PageRank converged in %d iterations (err=%.2e)", iteration + 1, err)
            break
    else:
        logger.warning("PageRank did not converge in %d iterations", max_iter)

    return pr


# ---------------------------------------------------------------------------
# Write back to Neo4j
# ---------------------------------------------------------------------------

def _write_pagerank(session, scores: dict[int, float], batch_size: int = WRITE_BATCH) -> int:
    """
    Write pagerank scores to Neo4j Case nodes in batches.

    Returns the total number of nodes updated.
    """
    items = [{"id": nid, "pr": score} for nid, score in scores.items()]
    total = 0

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        result = session.run(
            """
            UNWIND $batch AS row
            MATCH (c:Case {id: row.id})
            SET c.pagerank = row.pr
            RETURN count(*) AS updated
            """,
            batch=batch,
        )
        record = result.single()
        updated = int(record["updated"]) if record else 0
        total += updated
        logger.info(
            "  Wrote batch %d–%d: %d nodes updated",
            start, start + len(batch) - 1, updated,
        )

    return total


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify(session, scores: dict[int, float]) -> None:
    """Print top-10 cases by PageRank and landmark spot-checks."""
    top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n── Top 10 cases by PageRank ──")
    for i, (cid, pr) in enumerate(top10, 1):
        r = session.run(
            "MATCH (c:Case {id: $id}) RETURN c.name AS name",
            id=cid,
        ).single()
        name = ((r["name"] or "Unknown") if r else "Unknown")[:55]
        print(f"  {i:>2}. [{cid:>8}] {name:<57} pagerank={pr:.6f}")

    landmarks = [
        (107729, "Terry v. Ohio"),
        (107564, "Katz v. United States"),
        (106285, "Mapp v. Ohio"),
    ]
    print("\n── Landmark spot-checks ──")
    for cid, label in landmarks:
        pr = scores.get(cid)
        print(f"  {label} ({cid}): pagerank={pr}")

    # Coverage check against total node count
    r = session.run(
        "MATCH (c:Case) RETURN count(c) AS total, "
        "count(c.pagerank) AS with_pr"
    ).single()
    print(
        f"\n  Total Case nodes: {r['total']}  |  With pagerank written: {r['with_pr']}"
        f"  |  Computed: {len(scores)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_and_store_pagerank() -> None:
    """
    Full pipeline:
      1. Pull CITES edge list from Neo4j
      2. Compute PageRank via power iteration (pure Python — no GDS required)
      3. Write scores back to Case nodes in batches
      4. Verify output
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            # Step 1 — fetch graph
            edges, edge_node_ids = _fetch_edges(session)
            all_case_ids = _fetch_all_case_ids(session)
            # Include isolated nodes (no edges) so every Case gets a score
            all_node_ids = all_case_ids | edge_node_ids

            if not all_node_ids:
                logger.error("No Case nodes found — is the graph loaded?")
                return

            # Step 2 — compute PageRank
            logger.info(
                "Computing PageRank on %d nodes, %d edges (alpha=%.2f) …",
                len(all_node_ids), len(edges), DAMPING,
            )
            scores = _compute_pagerank_python(edges, all_node_ids)
            logger.info(
                "PageRank computed — min=%.6f  max=%.6f  mean=%.6f",
                min(scores.values()),
                max(scores.values()),
                sum(scores.values()) / len(scores),
            )

            # Step 3 — write
            logger.info("Writing %d scores to Neo4j in batches of %d …", len(scores), WRITE_BATCH)
            total_written = _write_pagerank(session, scores)
            logger.info("Total nodes written: %d", total_written)

            # Step 4 — verify
            _verify(session, scores)

    finally:
        driver.close()

    print("\n✅  PageRank scores written to Neo4j (property: `pagerank` on Case nodes).")
    print("    Re-run this script after any graph update to refresh scores.")
    print("    Next steps:")
    print("      python -m benchmark.evaluate")
    print("      python -m benchmark.train_scorer\n")


if __name__ == "__main__":
    compute_and_store_pagerank()
