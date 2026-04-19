from detector.eyecite_parser import _CORPUS_INDEX
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from config import MILVUS_URI, MILVUS_COLLECTION, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, LANDMARK_IDS

LANDMARK_CITATIONS = {
    107729: "392 U.S. 1",
    107564: "389 U.S. 347",
    106285: "367 U.S. 643",
    111262: "468 U.S. 897",
    110959: "462 U.S. 213",
}

milvus = MilvusClient(uri=MILVUS_URI)
milvus.load_collection(MILVUS_COLLECTION)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

print(f"\n{'ID':<10} {'Citation':<20} {'Corpus':<8} {'Milvus':<8} {'Neo4j':<8}")
print("-" * 60)

for case_id, citation in LANDMARK_CITATIONS.items():
    corpus = "✅" if _CORPUS_INDEX.get(citation) else "❌"
    milvus_hit = milvus.query(
        collection_name=MILVUS_COLLECTION,
        filter=f"case_id == {case_id}",
        output_fields=["case_id"], limit=1
    )
    milvus_ok = "✅" if milvus_hit else "❌"
    with driver.session() as session:
        rec = session.run(
            "MATCH (c:Case {id: $id}) RETURN c.name AS name", id=case_id
        ).single()
    neo4j_ok = "✅" if rec else "❌"
    print(f"{case_id:<10} {citation:<20} {corpus:<8} {milvus_ok:<8} {neo4j_ok:<8}")

driver.close()