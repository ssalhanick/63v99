import os
from dotenv import load_dotenv

load_dotenv()

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = f'{ROOT_DIR}/data/raw'
PROCESSED_DIR = f'{ROOT_DIR}/data/processed'

# Neo4j
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# CourtListener
COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN")
COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"

# Landmark Fourth Amendment case IDs
LANDMARK_IDS = [
    107729,   # Terry v. Ohio (1968)
    107564,   # Katz v. United States (1967)
    106285,   # Mapp v. Ohio (1961)
    111262,   # United States v. Leon (1984)
    110959,   # Illinois v. Gates (1983)
]


# Embedding
EMBEDDING_MODEL   = "nlpaueb/legal-bert-base-uncased"
EMBEDDING_DIM     = 768
MILVUS_URI        = "http://localhost:19530"
# MILVUS_DB_PATH  = os.path.join(ROOT_DIR, "milvus_verit.db")
MILVUS_COLLECTION = "case_embeddings"

# Vector pruning thresholds (Week 4)
MIN_TEXT_LENGTH   = 200    # characters — skip cases with very short plain_text
MAX_TEXT_LENGTH   = 50000  # characters — truncate before embedding

# HNSW index parameters (Week 4 — tune in Week 8 if recall drops below 95%)
HNSW_M                = 16    # bidirectional links per node
HNSW_EF_CONSTRUCTION  = 200   # build-time search width
HNSW_EF               = 50    # query-time search width

# ANN search parameters (Week 5)
TOP_K                  = 5     # candidates returned per query
SIMILARITY_THRESHOLD   = 0.75  # cosine similarity floor — tune on validation set in Week 8

# Hybrid search — Reciprocal Rank Fusion (Week 5)
# RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)
RRF_K                  = 60    # smoothing constant — standard default, rarely needs tuning
BM25_INDEX_PATH        = os.path.join(PROCESSED_DIR, "bm25_index.pkl")

# Connectivity (Layer 3 — Option B)
CITATION_DENSITY_THRESHOLD = 3   # minimum shared citations — tune on validation set in Week 8

# Cache (Week 6)
CACHE_TTL             = 3600   # seconds — TTL for query embedding + ANN result cache
CACHE_MAX_SIZE        = 512    # max entries in LRU cache