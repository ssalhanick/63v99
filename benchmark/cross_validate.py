"""
benchmark/cross_validate.py
Week 8 stress test — 10-fold stratified cross-validation on the full benchmark.

Checkpointed — safe to interrupt and resume. Completed folds are saved to
benchmark/cv_checkpoint.json and skipped on restart.

Usage:
    python -m benchmark.cross_validate
    python -m benchmark.cross_validate --dry-run
    python -m benchmark.cross_validate --reset   # clear checkpoint and restart
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from neo4j import GraphDatabase
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from benchmark.evaluate import (
    load_benchmark,
    run_inference,
    compute_metrics,
    layer1_metrics,
    layer2_metrics,
    layer3_metrics,
    layer4_metrics,
)

THRESHOLDS_PATH    = Path(config.BENCHMARK_DIR) / "tuned_thresholds.json"
CV_REPORT_PATH     = Path(config.BENCHMARK_DIR) / "cv_report.json"
CV_CHECKPOINT_PATH = Path(config.BENCHMARK_DIR) / "cv_checkpoint.json"
N_FOLDS            = 10
ANOMALY_THRESHOLD  = 0.05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_thresholds() -> dict:
    if not THRESHOLDS_PATH.exists():
        logger.error("tuned_thresholds.json not found — run evaluate.py first.")
        sys.exit(1)
    with open(THRESHOLDS_PATH) as f:
        t = json.load(f)
    logger.info(
        "Loaded thresholds — SIM=%.2f  RRF=%.3f  DENSITY=%d",
        t["sim_threshold"], t["rrf_threshold"], t["density_threshold"],
    )
    return t


def make_strata_labels(data: list[dict]) -> list[str]:
    return [f"{d['label']}_{d['subtype']}" for d in data]


def stats(values: list[float]) -> dict:
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 4),
        "std":  round(float(arr.std()),  4),
        "min":  round(float(arr.min()),  4),
        "max":  round(float(arr.max()),  4),
    }


def load_checkpoint() -> list[dict]:
    """Load completed fold logs from checkpoint. Returns empty list if none."""
    if CV_CHECKPOINT_PATH.exists():
        with open(CV_CHECKPOINT_PATH) as f:
            fold_logs = json.load(f)
        logger.info(
            "Resuming from checkpoint — %d/%d folds already complete",
            len(fold_logs), N_FOLDS,
        )
        return fold_logs
    return []


def save_checkpoint(fold_logs: list[dict]) -> None:
    with open(CV_CHECKPOINT_PATH, "w") as f:
        json.dump(fold_logs, f, indent=2)
    logger.info("Checkpoint saved — %d/%d folds complete", len(fold_logs), N_FOLDS)


def print_cv_summary(layer_name: str, fold_metrics: list[dict]):
    precisions = [m["precision"] for m in fold_metrics]
    recalls    = [m["recall"]    for m in fold_metrics]
    f1s        = [m["f1"]        for m in fold_metrics]

    p = stats(precisions)
    r = stats(recalls)
    f = stats(f1s)

    print(f"\n{'='*60}")
    print(f"  {layer_name}")
    print(f"{'='*60}")
    print(f"  Precision : {p['mean']:.4f} ± {p['std']:.4f}  [{p['min']:.4f} – {p['max']:.4f}]")
    print(f"  Recall    : {r['mean']:.4f} ± {r['std']:.4f}  [{r['min']:.4f} – {r['max']:.4f}]")
    print(f"  F1        : {f['mean']:.4f} ± {f['std']:.4f}  [{f['min']:.4f} – {f['max']:.4f}]")


def flag_anomalous_folds(fold_f1s: list[float]) -> list[int]:
    mean_f1 = np.mean(fold_f1s)
    return [i for i, f1 in enumerate(fold_f1s) if mean_f1 - f1 > ANOMALY_THRESHOLD]


def main(dry_run: bool = False, reset: bool = False):
    if reset and CV_CHECKPOINT_PATH.exists():
        CV_CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleared — starting from fold 1")

    data       = load_benchmark()
    thresholds = load_thresholds()
    sim        = thresholds["sim_threshold"]
    rrf        = thresholds["rrf_threshold"]
    density    = thresholds["density_threshold"]

    strata  = make_strata_labels(data)
    indices = list(range(len(data)))
    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    if dry_run:
        print(f"\n── Dry run — validating {N_FOLDS}-fold split on {len(data)} entries ──")
        for fold, (_, test_idx) in enumerate(skf.split(indices, strata), 1):
            fold_data  = [data[i] for i in test_idx]
            fold_strat = [strata[i] for i in test_idx]
            label_dist: dict[str, int] = {}
            for s in fold_strat:
                label_dist[s] = label_dist.get(s, 0) + 1
            print(f"  Fold {fold:2d}: {len(fold_data):3d} entries  {label_dist}")
        print("\n✅  Split looks valid. Remove --dry-run to run full CV.")
        return

    # Load checkpoint — skip already-completed folds
    fold_logs      = load_checkpoint()
    completed_folds = {log["fold"] for log in fold_logs}

    logger.info(
        "Starting %d-fold CV on %d entries (~%.0f min estimated)",
        N_FOLDS, len(data), len(data) * 1.5 * N_FOLDS / 60,
    )

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    )

    try:
        for fold, (_, test_idx) in enumerate(skf.split(indices, strata), 1):
            if fold in completed_folds:
                logger.info("Fold %d/%d — skipping (already complete)", fold, N_FOLDS)
                continue

            fold_data = [data[i] for i in test_idx]
            logger.info(
                "── Fold %d/%d  (%d entries) ──────────────────────",
                fold, N_FOLDS, len(fold_data),
            )

            results = run_inference(fold_data, driver)

            l1 = layer1_metrics(results)
            l2 = layer2_metrics(results, sim, rrf)
            l3 = layer3_metrics(results, density)
            l4 = layer4_metrics(results)
            cb = compute_metrics(results, sim, rrf, density)

            fold_logs.append({
                "fold":     fold,
                "n":        len(fold_data),
                "combined": cb,
                "layer1":   l1,
                "layer2":   l2,
                "layer3":   l3,
                "layer4":   l4,
            })

            logger.info(
                "  Fold %d done — P=%.4f  R=%.4f  F1=%.4f",
                fold, cb["precision"], cb["recall"], cb["f1"],
            )

            # Save after every fold — safe to interrupt after this point
            save_checkpoint(fold_logs)

    finally:
        driver.close()

    if len(fold_logs) < N_FOLDS:
        logger.warning(
            "Only %d/%d folds complete. Re-run to resume from fold %d.",
            len(fold_logs), N_FOLDS, len(fold_logs) + 1,
        )
        return

    # ── All folds done — print results ────────────────────────────────────────
    fold_logs_sorted = sorted(fold_logs, key=lambda x: x["fold"])

    fold_results: dict[str, list[dict]] = {
        "combined": [], "layer1": [], "layer2": [], "layer3": [], "layer4": []
    }
    for log in fold_logs_sorted:
        for key in fold_results:
            fold_results[key].append(log[key])

    print(f"\n{'='*60}")
    print(f"  VERIT — {N_FOLDS}-FOLD CROSS-VALIDATION RESULTS")
    print(f"  Dataset: {len(data)} entries  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    print_cv_summary("Layer 1 — Existence",    fold_results["layer1"])
    print_cv_summary("Layer 2 — Semantic",     fold_results["layer2"])
    print_cv_summary("Layer 3 — Connectivity", fold_results["layer3"])
    print_cv_summary("Layer 4 — Metadata",     fold_results["layer4"])
    print_cv_summary("Combined pipeline",       fold_results["combined"])

    combined_f1s = [m["f1"] for m in fold_results["combined"]]
    anomalous    = flag_anomalous_folds(combined_f1s)

    print(f"\n── Fold F1 scores (combined) ─────────────────────────")
    for i, f1 in enumerate(combined_f1s, 1):
        flag = " ⚠ ANOMALY" if (i - 1) in anomalous else ""
        print(f"  Fold {i:2d}: {f1:.4f}{flag}")

    if anomalous:
        print(f"\n⚠  {len(anomalous)} anomalous fold(s) detected "
              f"(F1 dropped >{ANOMALY_THRESHOLD} below mean).")
        print("   Inspect fold_logs in cv_report.json for details.")
    else:
        print(f"\n✅  No anomalous folds — F1 is stable across all {N_FOLDS} folds.")

    report = {
        "generated_at":      datetime.now().isoformat(),
        "n_folds":           N_FOLDS,
        "dataset_size":      len(data),
        "thresholds":        thresholds,
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "anomalous_folds":   anomalous,
        "summary": {
            layer: {
                "precision": stats([m["precision"] for m in metrics]),
                "recall":    stats([m["recall"]    for m in metrics]),
                "f1":        stats([m["f1"]        for m in metrics]),
            }
            for layer, metrics in fold_results.items()
        },
        "fold_logs": fold_logs_sorted,
    }

    with open(CV_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("CV report saved → %s", CV_REPORT_PATH)

    # Clean up checkpoint on successful completion
    if CV_CHECKPOINT_PATH.exists():
        CV_CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleaned up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate split logic without running inference")
    parser.add_argument("--reset", action="store_true",
                        help="Clear checkpoint and restart from fold 1")
    args = parser.parse_args()
    main(dry_run=args.dry_run, reset=args.reset)