import pytest
import json
import os
from config import RAW_DIR

@pytest.fixture
def raw_cases():
    path = os.path.join(RAW_DIR, "batch_2015_present.json")
    with open(path) as f:
        return json.load(f)

@pytest.fixture
def merged_cases():
    path = os.path.join(RAW_DIR,"cases_merged.json")
    with open(path) as f:
        return json.load(f)