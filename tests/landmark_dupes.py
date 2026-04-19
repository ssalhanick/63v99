from pymilvus import MilvusClient
from config import MILVUS_URI, MILVUS_COLLECTION, LANDMARK_IDS

client = MilvusClient(uri=MILVUS_URI)
client.load_collection(MILVUS_COLLECTION)

for lid in LANDMARK_IDS:
    result = client.query(
        collection_name=MILVUS_COLLECTION,
        filter=f"case_id == {lid}",
        output_fields=["case_id"],
        limit=1
    )
    status = "✅ already in Milvus" if result else "❌ not in Milvus"
    print(f"{lid} | {status}")