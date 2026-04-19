import os, re

search_terms = ["HALLUCINATED", "SUSPICIOUS", "REAL"]
search_dirs = ["api", "detector"]

for directory in search_dirs:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path) as f:
                    for i, line in enumerate(f, 1):
                        if any(term in line for term in search_terms):
                            print(f"{path}:{i}: {line.rstrip()}")