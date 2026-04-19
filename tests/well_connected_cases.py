from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

with driver.session() as session:
    result = session.run("""
        MATCH (c:Case)-[r]-()
        WHERE c.stub = false
        RETURN c.id, c.name, count(r) AS degree
        ORDER BY degree DESC
        LIMIT 10
    """)
    for row in result:
        print(row["c.id"], "|", row["c.name"], "| connections:", row["degree"])