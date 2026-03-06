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