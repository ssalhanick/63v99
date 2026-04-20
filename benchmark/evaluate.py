"""
benchmark/evaluate.py
Week 8 — Threshold tuning on the validation split (with Layer 4 metadata check).

Run from project root:
    python -m benchmark.evaluate
"""

import json
import logging
import sys
import itertools
from pathlib import Path
import time

from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from detector.existence_check import check_existence
from detector.semantic_check import semantic_check
from detector.connectivity_check import check_connectivity
from detector.metadata_check import check_metadata
from detector.llm_check import llm_check


BENCHMARK_PATH      = Path(config.BENCHMARK_DIR) / "benchmark.json"
THRESHOLDS_OUT_PATH = Path(config.BENCHMARK_DIR) / "tuned_thresholds.json"
SPLIT_CACHE_PATH    = Path(config.BENCHMARK_DIR) / "split_indices.json"

HALLUCINATED = "HALLUCINATED"
SUSPICIOUS   = "SUSPICIOUS"
REAL         = "REAL"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SIM_THRESHOLDS     = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
RRF_THRESHOLDS     = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035]
DENSITY_THRESHOLDS = [1, 2, 3, 4, 5]


def load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d benchmark entries from %s", len(data), BENCHMARK_PATH)
    return data


def stratified_split(data: list[dict], test_size: float = 0.20, seed: int = 42):
    """80/20 stratified split. Cached on first run — never reshuffles."""
    if SPLIT_CACHE_PATH.exists():
        with open(SPLIT_CACHE_PATH, "r") as f:
            cache = json.load(f)
        val_ids  = set(cache["val_ids"])
        test_ids = set(cache["test_ids"])
        val  = [d for d in data if d["benchmark_id"] in val_ids]
        test = [d for d in data if d["benchmark_id"] in test_ids]
        logger.info("Loaded cached split — val=%d  test=%d", len(val), len(test))
        return val, test

    strata  = [f"{d['label']}_{d['subtype']}" for d in data]
    indices = list(range(len(data)))
    val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=strata, random_state=seed
    )
    val  = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]

    with open(SPLIT_CACHE_PATH, "w") as f:
        json.dump(
            {"val_ids":  [d["benchmark_id"] for d in val],
             "test_ids": [d["benchmark_id"] for d in test]},
            f, indent=2,
        )
    logger.info("Created and cached new split — val=%d  test=%d", len(val), len(test))
    return val, test


def run_entry(entry: dict, driver) -> dict:
    """Run all four layers. Returns raw scores for threshold-free re-evaluation."""
    case_id         = entry.get("case_id")
    context_text    = entry["context"]
    citation_string = entry["citation"]

    exists = check_existence(case_id, driver=driver) if case_id is not None else False

    if not exists:
        return {
            "benchmark_id":     entry["benchmark_id"],
            "label":            entry["label"],
            "subtype":          entry["subtype"],
            "corruption_type":  entry.get("corruption_type"),
            "exists":           False,
            "rrf_score":        None,
            "dense_score":      None,
            "density_score":    None,
            "metadata_valid":   None,
            "meta_checked":     False,
            "meta_year_match":  None,
            "meta_court_match": None,
            "llm_accurate":     None,
            "llm_checked":      False,
        }

    sem  = semantic_check(context_text)
    conn = check_connectivity(case_id, driver=driver)
    meta = check_metadata(case_id, citation_string, driver=driver)

    # Only call LLM check on proposition-type entries to control cost
    corruption_type = entry.get("corruption_type")
    if corruption_type == "proposition":
        llm  = llm_check(case_id, context_text)
        time.sleep(1)
        llm_accurate = llm.is_accurate
        llm_checked  = True
    else:
        llm_accurate = None
        llm_checked  = False

    return {
         "benchmark_id":     entry["benchmark_id"],
        "label":            entry["label"],
        "subtype":          entry["subtype"],
        "corruption_type":  corruption_type,
        "exists":           True,
        "rrf_score":        sem.rrf_score,
        "dense_score":      sem.top_dense_score,
        "density_score":    conn.density_score,
        "metadata_valid":   meta.is_valid,
        "meta_checked":     meta.checked,
        "meta_year_match":  meta.year_match,
        "meta_court_match": meta.court_match,
        "llm_accurate":     llm_accurate,
        "llm_checked":      llm_checked,
    }


def run_inference(entries: list[dict], driver) -> list[dict]:
    results = []
    n = len(entries)
    for i, entry in enumerate(entries, 1):
        if i % 10 == 0 or i == n:
            logger.info("  inference %d / %d", i, n)
        results.append(run_entry(entry, driver))
    return results


def apply_verdict(result: dict, sim_thresh: float, rrf_thresh: float, density_thresh: int) -> str:
    if not result["exists"]:
        return HALLUCINATED

    # Layer 4 — metadata mismatch catches Type B
    if result.get("meta_checked") and result.get("metadata_valid") is False:
        return HALLUCINATED
    
    # Layer 2b — LLM proposition check
    if result.get("llm_checked") and result.get("llm_accurate") is False:
        return SUSPICIOUS

    rrf_score     = result["rrf_score"]
    dense_score   = result["dense_score"]
    density_score = result["density_score"]

    l2_pass = (
        (rrf_score   is not None and rrf_score   >= rrf_thresh) or
        (dense_score is not None and dense_score >= sim_thresh)
    )
    l3_pass = density_score is not None and density_score >= density_thresh

    if l2_pass and l3_pass:           return REAL
    elif not l2_pass and not l3_pass: return HALLUCINATED
    else:                             return SUSPICIOUS


def compute_metrics(results, sim_thresh, rrf_thresh, density_thresh) -> dict:
    tp = fp = fn = tn = 0
    subtype_counts: dict[str, dict] = {}

    for r in results:
        pred      = apply_verdict(r, sim_thresh, rrf_thresh, density_thresh)
        truth     = r["label"]
        pred_pos  = pred in (HALLUCINATED, SUSPICIOUS)
        truth_pos = truth in (HALLUCINATED, SUSPICIOUS)

        if pred_pos and truth_pos:       tp += 1
        elif pred_pos and not truth_pos: fp += 1
        elif not pred_pos and truth_pos: fn += 1
        else:                            tn += 1

        st = r["subtype"] or "REAL"
        if st not in subtype_counts:
            subtype_counts[st] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        if pred_pos and truth_pos:       subtype_counts[st]["tp"] += 1
        elif pred_pos and not truth_pos: subtype_counts[st]["fp"] += 1
        elif not pred_pos and truth_pos: subtype_counts[st]["fn"] += 1
        else:                            subtype_counts[st]["tn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(results) if results else 0.0

    subtype_f1 = {}
    for st, c in subtype_counts.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
        subtype_f1[st] = round(2 * p * r / (p + r) if (p + r) > 0 else 0.0, 4)

    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "accuracy": round(accuracy, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "subtype_f1": subtype_f1,
    }


def layer1_metrics(results: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for r in results:
        pred_pos  = not r["exists"]
        truth_pos = r["label"] in (HALLUCINATED, SUSPICIOUS)
        if pred_pos and truth_pos:       tp += 1
        elif pred_pos and not truth_pos: fp += 1
        elif not pred_pos and truth_pos: fn += 1
        else:                            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def layer2_metrics(results: list[dict], sim_thresh: float, rrf_thresh: float) -> dict:
    """Isolated: evaluated only on exists=True entries so L1 doesn't inflate recall."""
    tp = fp = fn = tn = 0
    for r in results:
        if not r["exists"]:
            continue
        rrf_score   = r.get("rrf_score")
        dense_score = r.get("dense_score")
        l2_fail   = (
            (rrf_score   is None or rrf_score   < rrf_thresh) and
            (dense_score is None or dense_score < sim_thresh)
        )
        pred_pos  = l2_fail
        truth_pos = r["label"] in (HALLUCINATED, SUSPICIOUS)
        if pred_pos and truth_pos:       tp += 1
        elif pred_pos and not truth_pos: fp += 1
        elif not pred_pos and truth_pos: fn += 1
        else:                            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def layer2b_metrics(results: list[dict]) -> dict:
    """Isolated: evaluated only on exists=True, llm_checked=True entries."""
    tp = fp = fn = tn = 0
    for r in results:
        if not r["exists"] or not r.get("llm_checked"):
            continue
        pred_pos  = r.get("llm_accurate") is False
        truth_pos = r["label"] in (HALLUCINATED, SUSPICIOUS)
        if pred_pos and truth_pos:       tp += 1
        elif pred_pos and not truth_pos: fp += 1
        elif not pred_pos and truth_pos: fn += 1
        else:                            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def layer3_metrics(results: list[dict], density_thresh: int) -> dict:
    """Isolated: evaluated only on exists=True entries."""
    tp = fp = fn = tn = 0
    for r in results:
        if not r["exists"]:
            continue
        density_score = r.get("density_score")
        l3_fail   = density_score is None or density_score < density_thresh
        pred_pos  = l3_fail
        truth_pos = r["label"] in (HALLUCINATED, SUSPICIOUS)
        if pred_pos and truth_pos:       tp += 1
        elif pred_pos and not truth_pos: fp += 1
        elif not pred_pos and truth_pos: fn += 1
        else:                            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def layer4_metrics(results: list[dict]) -> dict:
    """Isolated: evaluated only on exists=True, meta_checked=True entries."""
    tp = fp = fn = tn = 0
    for r in results:
        if not r["exists"] or not r.get("meta_checked"):
            continue
        pred_pos  = r.get("metadata_valid") is False
        truth_pos = r["label"] in (HALLUCINATED, SUSPICIOUS)
        if pred_pos and truth_pos:       tp += 1
        elif pred_pos and not truth_pos: fp += 1
        elif not pred_pos and truth_pos: fn += 1
        else:                            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def sweep_thresholds(results: list[dict]) -> tuple[dict, list[dict]]:
    logger.info(
        "Sweeping %d threshold combinations...",
        len(SIM_THRESHOLDS) * len(RRF_THRESHOLDS) * len(DENSITY_THRESHOLDS),
    )
    best       = None
    best_f1    = -1.0
    all_scores = []

    for sim, rrf, density in itertools.product(
        SIM_THRESHOLDS, RRF_THRESHOLDS, DENSITY_THRESHOLDS
    ):
        metrics = compute_metrics(results, sim, rrf, density)
        all_scores.append({
            "sim_threshold": sim, "rrf_threshold": rrf,
            "density_threshold": density, **metrics,
        })
        if (metrics["f1"] > best_f1 or
            (metrics["f1"] == best_f1 and best is not None and
             metrics["precision"] > best["precision"])):
            best_f1 = metrics["f1"]
            best = {"sim_threshold": sim, "rrf_threshold": rrf,
                    "density_threshold": density, **metrics}

    logger.info(
        "Best val F1=%.4f at SIM=%.2f  RRF=%.3f  DENSITY=%d",
        best["f1"], best["sim_threshold"],
        best["rrf_threshold"], best["density_threshold"],
    )
    return best, all_scores


def print_summary(label: str, metrics: dict):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    if "accuracy" in metrics:
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    if "subtype_f1" in metrics:
        print(f"  Subtype F1:")
        for st, f1 in sorted(metrics["subtype_f1"].items()):
            print(f"    {st:12s}: {f1:.4f}")


def main():
    data = load_benchmark()
    val, _ = stratified_split(data)

    logger.info("Connecting to Neo4j at %s", config.NEO4J_URI)
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )

    try:
        logger.info("Running inference on val set (%d entries)...", len(val))
        val_results = run_inference(val, driver)

        rrf_start = getattr(config, "RRF_THRESHOLD", 0.020)

        print("\n── Layer-isolated metrics (starting thresholds) ──")
        print_summary("Layer 1 — Existence   (val)", layer1_metrics(val_results))
        print_summary("Layer 2 — Semantic    (val)", layer2_metrics(val_results, config.SIMILARITY_THRESHOLD, rrf_start))
        print_summary("Layer 2b — LLM Prop   (val)", layer2b_metrics(val_results))
        print_summary("Layer 3 — Connectivity(val)", layer3_metrics(val_results, config.CITATION_DENSITY_THRESHOLD))
        print_summary("Layer 4 — Metadata    (val)", layer4_metrics(val_results))

        best, all_scores = sweep_thresholds(val_results)

        out = {
            "sim_threshold":     best["sim_threshold"],
            "rrf_threshold":     best["rrf_threshold"],
            "density_threshold": best["density_threshold"],
            "val_metrics": {
                "precision": best["precision"], "recall": best["recall"],
                "f1": best["f1"], "accuracy": best["accuracy"],
            },
            "sweep_log": sorted(all_scores, key=lambda x: -x["f1"])[:20],
        }
        with open(THRESHOLDS_OUT_PATH, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Tuned thresholds saved → %s", THRESHOLDS_OUT_PATH)

        print_summary(
            f"Combined (tuned)  SIM={best['sim_threshold']}  "
            f"RRF={best['rrf_threshold']}  DENSITY={best['density_threshold']}  (val)",
            best,
        )
        print(f"\n✅  Run  python -m benchmark.report  to evaluate on the held-out test set.\n")

    finally:
        driver.close()


if __name__ == "__main__":
    main()