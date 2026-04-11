# benchmark/density_histogram.py
import json
import matplotlib.pyplot as plt

data = json.load(open("benchmark/eval_report.json", encoding="utf-8"))
entries = data["per_entry"]

real_density   = [e["density_score"] for e in entries if e["label"] == "REAL"         and e["density_score"] is not None]
halluc_density = [e["density_score"] for e in entries if e["label"] == "HALLUCINATED" and e["density_score"] is not None]

print(f"REAL entries: {len(real_density)}")
print(f"HALLUCINATED entries: {len(halluc_density)}")

plt.figure(figsize=(8, 4))
plt.hist(real_density,   bins=20, alpha=0.6, label="REAL",         color="#2e7d32")
plt.hist(halluc_density, bins=20, alpha=0.6, label="HALLUCINATED", color="#c62828")
plt.xlabel("Citation Density Score (Layer 3)")
plt.ylabel("Count")
plt.title("Citation Density Distribution — Real vs. Hallucinated")
plt.legend()
plt.tight_layout()
plt.savefig("visualization/density_histogram.png", dpi=150)
print("Saved → visualization/density_histogram.png")