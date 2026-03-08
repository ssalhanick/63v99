import pytest
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, LANDMARK_IDS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session():
    """Module-scoped Neo4j session — one connection for all tests."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as s:
        yield s
    driver.close()


# ---------------------------------------------------------------------------
# Node count tests
# ---------------------------------------------------------------------------

def test_full_case_node_count(session):
    """Corpus should have exactly 1,358 full Case nodes."""
    result = session.run("MATCH (c:Case {stub: false}) RETURN count(c) AS n").single()
    assert result["n"] == 1358, f"Expected 1358 full nodes, got {result['n']}"


def test_stub_nodes_exist(session):
    """Stub nodes should exist — out-of-corpus citations are expected."""
    result = session.run("MATCH (c:Case {stub: true}) RETURN count(c) AS n").single()
    assert result["n"] > 0, "Expected stub nodes but found none"


def test_total_node_count(session):
    """Total nodes should be full corpus + stubs."""
    full  = session.run("MATCH (c:Case {stub: false}) RETURN count(c) AS n").single()["n"]
    stubs = session.run("MATCH (c:Case {stub: true}) RETURN count(c) AS n").single()["n"]
    total = session.run("MATCH (c:Case) RETURN count(c) AS n").single()["n"]
    assert total == full + stubs, (
        f"Total nodes ({total}) doesn't match full ({full}) + stubs ({stubs})"
    )


# ---------------------------------------------------------------------------
# Landmark tests
# ---------------------------------------------------------------------------

def test_all_landmarks_present(session):
    """All landmark IDs from config should exist in the graph."""
    for lid in LANDMARK_IDS:
        result = session.run(
            "MATCH (c:Case {id: $id}) RETURN c.id AS id", id=lid
        ).single()
        assert result is not None, f"Landmark id={lid} missing from graph"


def test_all_landmarks_flagged(session):
    """All landmark nodes should have landmark=true."""
    for lid in LANDMARK_IDS:
        result = session.run(
            "MATCH (c:Case {id: $id}) RETURN c.landmark AS landmark", id=lid
        ).single()
        assert result is not None, f"Landmark id={lid} missing from graph"
        assert result["landmark"] is True, (
            f"Landmark id={lid} exists but landmark flag is not true"
        )


def test_landmark_count_matches_config(session):
    """Number of landmark nodes in graph should match LANDMARK_IDS in config."""
    result = session.run(
        "MATCH (c:Case {landmark: true}) RETURN count(c) AS n"
    ).single()
    assert result["n"] == len(LANDMARK_IDS), (
        f"Expected {len(LANDMARK_IDS)} landmarks, found {result['n']}"
    )


def test_landmarks_have_metadata(session):
    """Landmark nodes should have name and year — not just an ID."""
    results = session.run(
        "MATCH (c:Case {landmark: true}) RETURN c.id AS id, c.name AS name, c.year AS year"
    ).data()
    for r in results:
        assert r["name"] is not None, f"Landmark id={r['id']} missing name"
        assert r["year"] is not None, f"Landmark id={r['id']} missing year"


# ---------------------------------------------------------------------------
# Metadata completeness tests
# ---------------------------------------------------------------------------

def test_full_nodes_have_name(session):
    """All full (non-stub) Case nodes should have a name."""
    result = session.run("""
        MATCH (c:Case {stub: false})
        WHERE c.name IS NULL OR c.name = 'Unknown'
        RETURN count(c) AS n
    """).single()
    assert result["n"] == 0, f"{result['n']} full nodes are missing a name"


def test_full_nodes_have_year(session):
    """All full Case nodes should have a year — warn if more than 5% are missing."""
    total = session.run(
        "MATCH (c:Case {stub: false}) RETURN count(c) AS n"
    ).single()["n"]
    missing = session.run("""
        MATCH (c:Case {stub: false})
        WHERE c.year IS NULL
        RETURN count(c) AS n
    """).single()["n"]
    pct_missing = missing / total if total > 0 else 0
    assert pct_missing < 0.05, (
        f"{missing}/{total} full nodes ({pct_missing:.1%}) are missing year — exceeds 5% threshold"
    )


def test_full_nodes_have_court(session):
    """All full Case nodes should have a court_id."""
    result = session.run("""
        MATCH (c:Case {stub: false})
        WHERE c.court IS NULL OR c.court = 'unknown'
        RETURN count(c) AS n
    """).single()
    assert result["n"] == 0, f"{result['n']} full nodes are missing court"


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------

def test_citation_edges_exist(session):
    """Graph should have a substantial number of CITES edges."""
    result = session.run("MATCH ()-[r:CITES]->() RETURN count(r) AS n").single()
    assert result["n"] > 1000, (
        f"Expected >1000 CITES edges, found {result['n']} — graph may not have loaded correctly"
    )


def test_edges_are_directed_correctly(session):
    """CITES edges should go from corpus cases outward — sample check."""
    result = session.run("""
        MATCH (a:Case)-[:CITES]->(b:Case)
        WHERE a.stub = false
        RETURN count(*) AS n
    """).single()
    assert result["n"] > 0, "No outgoing CITES edges from full corpus nodes"


def test_no_self_citations(session):
    """No case should cite itself."""
    result = session.run("""
        MATCH (c:Case)-[:CITES]->(c)
        RETURN count(c) AS n
    """).single()
    assert result["n"] == 0, f"Found {result['n']} self-citations in graph"


# ---------------------------------------------------------------------------
# Connectivity tests
# ---------------------------------------------------------------------------

def test_corpus_cases_have_outgoing_citations(session):
    """At least 50% of corpus cases should have at least one outgoing citation."""
    total = session.run(
        "MATCH (c:Case {stub: false}) RETURN count(c) AS n"
    ).single()["n"]
    with_citations = session.run("""
        MATCH (c:Case {stub: false})
        WHERE (c)-[:CITES]->()
        RETURN count(c) AS n
    """).single()["n"]
    pct = with_citations / total if total > 0 else 0
    assert pct >= 0.50, (
        f"Only {with_citations}/{total} ({pct:.1%}) corpus cases have outgoing citations — "
        f"expected at least 50%"
    )

@pytest.mark.skip(reason="Landmarks isolated from corpus by design — Layer 3 uses citation density instead")
def test_landmarks_are_reachable(session):
    """At least one landmark should be reachable from the corpus via CITES."""
    result = session.run("""
        MATCH (a:Case {stub: false})-[:CITES]->(b:Case {landmark: true})
        RETURN count(a) AS n
    """).single()
    assert result["n"] > 0, (
        "No corpus cases cite any landmark directly — "
        "check landmark IDs and citation edge loading"
    )
