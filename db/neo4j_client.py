from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

def create_case(tx, case_id, name, year, court):
    tx.run("""
        MERGE (c:Case {id: $case_id})
        SET c.name = $name, c.year = $year, c.court = $court
    """, case_id=case_id, name=name, year=year, court=court)

def create_citation(tx, from_id, to_id):
    tx.run("""
        MATCH (a:Case {id: $from_id})
        MATCH (b:Case {id: $to_id})
        MERGE (a)-[:CITES]->(b)
    """, from_id=from_id, to_id=to_id)

def create_landmark(tx, case_id):
    tx.run("""
        MATCH (c:Case {id: $case_id})
        SET c.landmark = true
    """, case_id=case_id)

if __name__ == "__main__":
    with driver.session() as session:
        session.execute_write(
            create_case, "test-001", "Test v. United States", 2020, "ca9"
        )
        print("Connected successfully.")
    driver.close()