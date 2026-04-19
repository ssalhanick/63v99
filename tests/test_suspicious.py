import logging
from detector.semantic_check import _load_all, _embed, _case_specific_similarity
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from detector.pipeline import run_pipeline

text = """
In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held that 
officers may conduct a warrantless search of a vehicle without reasonable 
suspicion whenever a narcotics dog is present at the scene, regardless of 
whether the initial stop was lawful.
"""

results = run_pipeline(text)
for v in results:
    print(f"\nVerdict: {v.verdict}")
    print(f"L1 exists: {v.exists}")
    if v.semantic:
        print(f"L2 is_relevant: {v.semantic.is_relevant}")
        print(f"L2 rrf_score: {v.semantic.rrf_score}")
        print(f"L2 top_dense_score: {v.semantic.top_dense_score}")
    if v.connectivity:
        print(f"L3 is_connected: {v.connectivity.is_connected}")
        print(f"L3 density_score: {v.connectivity.density_score}")



_load_all()

correct = """
In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held that 
a trooper had reasonable suspicion to conduct a traffic stop where a vehicle's 
license plate lettering was partially obstructed, and that Utah's license plate 
maintenance law applied to out-of-state plates.
"""

wrong = """
In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held that 
officers may conduct a warrantless search of a vehicle without reasonable 
suspicion whenever a narcotics dog is present at the scene, regardless of 
whether the initial stop was lawful.
"""

correct_vec = _embed(correct)
wrong_vec   = _embed(wrong)

correct_sim = _case_specific_similarity(9942139, correct_vec)
wrong_sim   = _case_specific_similarity(9942139, wrong_vec)

print(f"Correct proposition similarity: {correct_sim:.4f}")
print(f"Wrong proposition similarity:   {wrong_sim:.4f}")
print(f"Gap: {correct_sim - wrong_sim:.4f}")