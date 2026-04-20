"""
benchmark/report.py
Week 8 — Final evaluation on the held-out test set.

Run AFTER evaluate.py:
    python -m benchmark.report
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from benchmark.evaluate import (
    load_benchmark,
    stratified_split,
    run_inference,
    apply_verdict,
    compute_metrics,
    layer1_metrics,
    layer2_metrics,
    layer2b_metrics,
    layer3_metrics,
    layer4_metrics,
    print_summary,
    HALLUCINATED,
    SUSPICIOUS,
    REAL,
)

THRESHOLDS_PATH = Path(config.BENCHMARK_DIR) / "tuned_thresholds.json"
REPORT_OUT_PATH = Path(config.BENCHMARK_DIR) / "eval_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_thresholds() -> dict:
    if not THRESHOLDS_PATH.exists():
        logger.error("tuned_thresholds.json not found — run python -m benchmark.evaluate first.")
        sys.exit(1)
    with open(THRESHOLDS_PATH, "r") as f:
        t = json.load(f)
    logger.info(
        "Loaded thresholds — SIM=%.2f  RRF=%.3f  DENSITY=%d",
        t["sim_threshold"], t["rrf_threshold"], t["density_threshold"],
    )
    return t


def confusion_matrix_rows(results, sim, rrf, density) -> list[dict]:
    rows = []
    for r in results:
        pred = apply_verdict(r, sim, rrf, density)
        rows.append({
            "benchmark_id":   r["benchmark_id"],
            "label":          r["label"],
            "subtype":        r["subtype"],
            "prediction":     pred,
            "correct":        pred == r["label"],
            "exists":         r["exists"],
            "rrf_score":      r["rrf_score"],
            "dense_score":    r["dense_score"],
            "density_score":  r["density_score"],
            "metadata_valid": r.get("metadata_valid"),
            "meta_checked":   r.get("meta_checked"),
        })
    return rows


def suspicious_breakdown(rows: list[dict]) -> dict:
    suspicious = [r for r in rows if r["prediction"] == SUSPICIOUS]
    return {
        "count":             len(suspicious),
        "true_real":         sum(1 for r in suspicious if r["label"] == REAL),
        "true_hallucinated": sum(1 for r in suspicious if r["label"] == HALLUCINATED),
    }


def false_negative_analysis(rows: list[dict]) -> list[dict]:
    return [
        {
            "benchmark_id":  r["benchmark_id"],
            "subtype":       r["subtype"],
            "exists":        r["exists"],
            "rrf_score":     r["rrf_score"],
            "density_score": r["density_score"],
            "meta_checked":  r.get("meta_checked"),
            "metadata_valid":r.get("metadata_valid"),
        }
        for r in rows
        if r["label"] == HALLUCINATED and r["prediction"] == REAL
    ]


def false_positive_analysis(rows: list[dict]) -> list[dict]:
    return [
        {
            "benchmark_id":  r["benchmark_id"],
            "rrf_score":     r["rrf_score"],
            "density_score": r["density_score"],
        }
        for r in rows
        if r["label"] == REAL and r["prediction"] == HALLUCINATED
    ]


def print_full_report(thresholds, l1, l2, l2b, l3, l4, combined, suspicious, fn_list, fp_list, n_test):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  VERIT — WEEK 8 EVALUATION REPORT")
    print(f"  Test set: {n_test} entries  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{bar}")
    print(f"\n  Tuned thresholds:")
    print(f"    SIMILARITY_THRESHOLD       = {thresholds['sim_threshold']}")
    print(f"    RRF_THRESHOLD              = {thresholds['rrf_threshold']}")
    print(f"    CITATION_DENSITY_THRESHOLD = {thresholds['density_threshold']}")

    print_summary("Layer 1 — Existence     (test)", l1)
    print_summary("Layer 2 — Semantic      (test)", l2)
    print_summary("Layer 2b — LLM Context  (test)", l2b)
    print_summary("Layer 3 — Connectivity  (test)", l3)
    print_summary("Layer 4 — Metadata      (test)", l4)
    print_summary("Combined pipeline       (test)", combined)

    print(f"\n── SUSPICIOUS bucket ──────────────────────────────────")
    print(f"  Count            : {suspicious['count']}")
    print(f"  True REAL        : {suspicious['true_real']}")
    print(f"  True HALLUCINATED: {suspicious['true_hallucinated']}")

    print(f"\n── False Negatives (HALLUCINATED predicted as REAL) ───")
    if fn_list:
        for fn in fn_list:
            print(f"  id={fn['benchmark_id']:4d}  subtype={fn['subtype']}  "
                  f"meta_checked={fn['meta_checked']}  meta_valid={fn['metadata_valid']}  "
                  f"rrf={fn['rrf_score']}  density={fn['density_score']}")
    else:
        print("  None — all hallucinated citations caught ✅")

    print(f"\n── False Positives (REAL predicted as HALLUCINATED) ───")
    if fp_list:
        for fp in fp_list:
            print(f"  id={fp['benchmark_id']:4d}  rrf={fp['rrf_score']}  density={fp['density_score']}")
    else:
        print("  None ✅")

    print(f"\n{bar}\n")


def main():
    thresholds = load_thresholds()
    sim     = thresholds["sim_threshold"]
    rrf     = thresholds["rrf_threshold"]
    density = thresholds["density_threshold"]

    data = load_benchmark()
    _, test = stratified_split(data)

    logger.info("Connecting to Neo4j at %s", config.NEO4J_URI)
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )

    try:
        logger.info("Running inference on test set (%d entries)...", len(test))
        test_results = run_inference(test, driver)
    finally:
        driver.close()

    l1       = layer1_metrics(test_results)
    l2       = layer2_metrics(test_results, sim, rrf)
    l2b      = layer2b_metrics(test_results)
    l3       = layer3_metrics(test_results, density)
    l4       = layer4_metrics(test_results)
    combined = compute_metrics(test_results, sim, rrf, density)

    rows       = confusion_matrix_rows(test_results, sim, rrf, density)
    suspicious = suspicious_breakdown(rows)
    fn_list    = false_negative_analysis(rows)
    fp_list    = false_positive_analysis(rows)

    print_full_report(
        thresholds, l1, l2, l2b, l3, l4, combined,
        suspicious, fn_list, fp_list, n_test=len(test),
    )

    report = {
        "generated_at":   datetime.now().isoformat(),
        "test_set_size":  len(test),
        "thresholds": {
            "sim_threshold":     sim,
            "rrf_threshold":     rrf,
            "density_threshold": density,
        },
        "layer_metrics": {"layer1": l1, "layer2": l2, "layer2b": l2b, "layer3": l3, "layer4": l4},
        "combined_metrics":     combined,
        "suspicious_breakdown": suspicious,
        "false_negatives":      fn_list,
        "false_positives":      fp_list,
        "per_entry":            rows,
    }

    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved → %s", REPORT_OUT_PATH)


if __name__ == "__main__":
    main()