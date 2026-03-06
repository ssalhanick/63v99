import requests
from neo4j import GraphDatabase
from config import (
    COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    LANDMARK_IDS
)

HEADERS = {"Authorization": f"Token {COURTLISTENER_TOKEN}"}


# ---------------------------------------------------------------------------
# CourtListener fetch
# ---------------------------------------------------------------------------

def fetch_opinion(opinion_id: int) -> dict | None:
    """Fetch a single opinion by ID from CourtListener."""
    url = f"{COURTLISTENER_BASE_URL}/opinions/{opinion_id}/"
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code == 200:
        return r.json()
    print(f"    ⚠️  Failed to fetch opinion {opinion_id} — status {r.status_code}")
    return None


def fetch_cluster(cluster_url: str) -> dict | None:
    """Fetch the cluster (case metadata) for an opinion."""
    r = requests.get(cluster_url, headers=HEADERS, timeout=10)
    if r.status_code == 200:
        return r.json()
    return None


def extract_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Neo4j writes
# ---------------------------------------------------------------------------

def _upsert_landmark_node(tx, case_id: int, name: str, year: int | None, court: str):
    tx.run("""
        MERGE (c:Case {id: $case_id})
        SET c.name     = $name,
            c.year     = $year,
            c.court    = $court,
            c.stub     = false,
            c.landmark = true
    """, case_id=case_id, name=name, year=year, court=court)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Verit — Landmark Case Fetcher")
    print(f"Fetching {len(LANDMARK_IDS)} landmark cases from CourtListener...\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    loaded = 0
    for opinion_id in LANDMARK_IDS:
        print(f"  Fetching opinion id={opinion_id}...")

        opinion = fetch_opinion(opinion_id)
        if not opinion:
            continue

        # Get case name and date from the cluster
        cluster_url = opinion.get("cluster")
        name  = f"Unknown (opinion {opinion_id})"
        year  = None
        court = "scotus"

        if cluster_url:
            cluster = fetch_cluster(cluster_url)
            if cluster:
                name  = cluster.get("case_name", name)
                year  = extract_year(cluster.get("date_filed"))
                court = cluster.get("docket", {}).get("court_id", "scotus") if isinstance(cluster.get("docket"), dict) else "scotus"

        print(f"    → {name} ({year}) [{court}]")

        with driver.session() as session:
            session.execute_write(_upsert_landmark_node, opinion_id, name, year, court)
            print(f"    ✅ Loaded into Neo4j")
            loaded += 1

    driver.close()

    print(f"\n{loaded}/{len(LANDMARK_IDS)} landmark cases loaded.")
    print("Run the landmark verification query to confirm:")
    print("  MATCH (c:Case {{landmark: true}}) RETURN c.name, c.id, c.year")


if __name__ == "__main__":
    main()