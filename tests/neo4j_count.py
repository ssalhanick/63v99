from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

with driver.session() as session:
    result = session.run("MATCH (c:Case) RETURN count(c) AS total")
    print("Total cases in Neo4j:", result.single()["total"])