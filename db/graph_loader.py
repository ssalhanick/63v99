import re
import ast
import os
import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

from config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    PROCESSED_DIR, LANDMARK_IDS
)

PARQUET_PATH = os.path.join(PROCESSED_DIR, "cases_enriched.parquet")
BATCH_SIZE   = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_opinion_id(url: str) -> int | None:
    if not isinstance(url, str):
        return None
    match = re.search(r"/(?:opinions?)/(\d+)/", url)
    return int(match.group(1)) if match else None


def safe_parse_list(val) -> list:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return []


# ---------------------------------------------------------------------------
# Batched Cypher transactions
# ---------------------------------------------------------------------------

def _batch_upsert_full_cases(tx, batch: list[dict]):
    tx.run("""
        UNWIND $batch AS row
        MERGE (c:Case {id: row.case_id})
        SET c.name  = row.name,
            c.year  = row.year,
            c.court = row.court,
            c.stub  = false
    """, batch=batch)


def _batch_upsert_stub_cases(tx, ids: list[int]):
    tx.run("""
        UNWIND $ids AS cid
        MERGE (c:Case {id: cid})
        ON CREATE SET c.stub = true
    """, ids=ids)


def _batch_upsert_citations(tx, edges: list[dict]):
    tx.run("""
        UNWIND $edges AS edge
        MATCH (a:Case {id: edge.from_id})
        MATCH (b:Case {id: edge.to_id})
        MERGE (a)-[:CITES]->(b)
    """, edges=edges)


def _mark_landmark(tx, case_id: int):
    tx.run("""
        MATCH (c:Case {id: $case_id})
        SET c.landmark = true
    """, case_id=case_id)


# ---------------------------------------------------------------------------
# Loading stages
# ---------------------------------------------------------------------------

def load_case_nodes(session, df: pd.DataFrame) -> set[int]:
    """Stage 1: Batch upsert all full Case nodes."""
    print("\n[1/4] Loading case nodes...")
    corpus_ids = set()
    batch = []

    for _, row in tqdm(df.iterrows(), total=len(df), unit="case"):
        case_id = int(row["case_id"])
        year    = None
        date    = row.get("date_filed")
        if pd.notna(date):
            try:
                year = int(str(date)[:4])
            except (ValueError, TypeError):
                pass

        batch.append({
            "case_id": case_id,
            "name":    str(row.get("case_name", "Unknown")),
            "year":    year,
            "court":   str(row.get("court_id", "unknown")),
        })
        corpus_ids.add(case_id)

        if len(batch) >= BATCH_SIZE:
            session.execute_write(_batch_upsert_full_cases, batch)
            batch.clear()

    if batch:
        session.execute_write(_batch_upsert_full_cases, batch)

    print(f"    ✅ {len(corpus_ids)} case nodes loaded.")
    return corpus_ids


def load_citation_edges(session, df: pd.DataFrame, corpus_ids: set[int]) -> dict:
    """Stage 2: Collect all edges in memory, then write stubs and edges in batches."""
    print("\n[2/4] Loading citation edges...")

    edges    = []
    stub_ids = set()
    skipped  = 0

    # Pass 1: collect everything into memory (fast — no DB calls)
    print("    Parsing citations...")
    for _, row in tqdm(df.iterrows(), total=len(df), unit="case"):
        from_id = int(row["case_id"])
        cited   = safe_parse_list(row.get("opinions_cited"))

        for url in cited:
            to_id = extract_opinion_id(url)
            if to_id is None:
                skipped += 1
                continue
            if to_id not in corpus_ids:
                stub_ids.add(to_id)
            edges.append({"from_id": from_id, "to_id": to_id})

    # Pass 2: write stubs
    stub_list = list(stub_ids)
    print(f"    Writing {len(stub_list)} stub nodes...")
    for i in range(0, len(stub_list), BATCH_SIZE):
        session.execute_write(_batch_upsert_stub_cases, stub_list[i:i+BATCH_SIZE])

    # Pass 3: write edges
    print(f"    Writing {len(edges)} citation edges...")
    for i in tqdm(range(0, len(edges), BATCH_SIZE), unit="batch"):
        session.execute_write(_batch_upsert_citations, edges[i:i+BATCH_SIZE])

    if skipped:
        print(f"    ⚠️  {skipped} URLs skipped (unparseable).")

    print(f"    ✅ Done — {len(edges)} edges, {len(stub_list)} stubs.")
    return {"edges": len(edges), "stubs": len(stub_list), "skipped": skipped}


def mark_landmarks(session) -> None:
    """Stage 3: Flag landmark cases."""
    print("\n[3/4] Marking landmark cases...")
    marked = 0
    for lid in LANDMARK_IDS:
        result = session.run(
            "MATCH (c:Case {id: $id}) RETURN c.name AS name", id=lid
        )
        record = result.single()
        if record:
            session.execute_write(_mark_landmark, lid)
            print(f"    ✅ Marked: {record['name']} (id={lid})")
            marked += 1
        else:
            print(f"    ⚠️  Landmark id={lid} not found in graph — skipping.")
    print(f"    {marked}/{len(LANDMARK_IDS)} landmark nodes marked.")


def verify_graph(session) -> None:
    """Stage 4: Summary of graph contents."""
    print("\n[4/4] Graph verification...")

    total_cases = session.run("MATCH (c:Case) RETURN count(c) AS n").single()["n"]
    full_cases  = session.run("MATCH (c:Case {stub: false}) RETURN count(c) AS n").single()["n"]
    stub_cases  = session.run("MATCH (c:Case {stub: true}) RETURN count(c) AS n").single()["n"]
    total_edges = session.run("MATCH ()-[r:CITES]->() RETURN count(r) AS n").single()["n"]
    landmarks   = session.run(
        "MATCH (c:Case {landmark: true}) RETURN c.name AS name, c.id AS id"
    ).data()

    print(f"    Total Case nodes : {total_cases}")
    print(f"      Full (corpus)  : {full_cases}")
    print(f"      Stubs          : {stub_cases}")
    print(f"    Total CITES edges: {total_edges}")
    print(f"    Landmark nodes   : {len(landmarks)}")
    for lm in landmarks:
        print(f"      - {lm['name']} (id={lm['id']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Verit — Neo4j Graph Loader")
    print(f"Source: {PARQUET_PATH}\n")

    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df)} cases loaded from parquet.")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            corpus_ids = load_case_nodes(session, df)
            stats      = load_citation_edges(session, df, corpus_ids)
            mark_landmarks(session)
            verify_graph(session)
    finally:
        driver.close()

    print("\n✅ Graph load complete.")


if __name__ == "__main__":
    main()