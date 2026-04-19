from pymilvus import MilvusClient
from config import MILVUS_URI, MILVUS_COLLECTION

client = MilvusClient(uri=MILVUS_URI)
client.load_collection(MILVUS_COLLECTION)

result = client.query(
    collection_name=MILVUS_COLLECTION,
    filter="case_id == 9942139",
    output_fields=["case_id", "embedding"],
    limit=1
)

emb = result[0]["embedding"]
print(f"case_id: {result[0]['case_id']}")
print(f"embedding type: {type(emb)}")
print(f"embedding length: {len(emb)}")
print(f"first 5 values: {emb[:5]}")