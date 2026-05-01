"""
benchmark/train_scorer.py

Phase 3, Step 3.1c — Train the calibrated logistic regression scorer.

Reads the raw_scores.csv produced by benchmark/evaluate.py and trains a
logistic regression that outputs P(hallucinated) for each citation.

The trained model replaces the hard boolean threshold fusion in pipeline.py.

Run from project root AFTER running benchmark/evaluate.py:
    python -m benchmark.train_scorer

Outputs:
    benchmark/scorer.pkl  — trained (scaler, model) bundle
    Prints cross-validation AUC, calibration check, and feature weights.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_SCORES_PATH = Path(config.BENCHMARK_DIR) / "raw_scores.csv"
SCORER_PATH     = Path(getattr(config, "SCORER_PATH", "benchmark/scorer.pkl"))

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURES = [
    "exists",          # bool → 0/1  (L1)
    "rrf_score",       # float        (L2a)
    "dense_score",     # float        (L2a)
    "case_sim",        # float | None (L2a, requires case_id)
    "density_score",   # int   | None (L3)
    "pagerank_score",  # float | None (L3 + Phase 4.1 — GDS PageRank)
    "metadata_valid",  # bool  | None (L4)
    "name_score",      # float | None (Step 1.3)
    "temporal_valid",  # bool  | None (Step 2.3)
]


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load and preprocess raw_scores.csv into (X, y, benchmark_ids).

    Missing values are filled with conservative defaults:
      - numeric scores  → 0.0  (treated as weakest signal)
      - boolean fields  → 1.0  (treated as valid / not failing)
    """
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    # Encode label: HALLUCINATED or SUSPICIOUS → 1, REAL → 0
    y = (df["label"].isin(["HALLUCINATED", "SUSPICIOUS"])).astype(int).values

    # Build feature matrix
    X_df = pd.DataFrame()
    X_df["exists"]         = df["exists"].astype(float)
    X_df["rrf_score"]      = df["rrf_score"].fillna(0.0).astype(float)
    X_df["dense_score"]    = df["dense_score"].fillna(0.0).astype(float)
    X_df["case_sim"]       = df["case_sim"].fillna(0.0).astype(float)
    X_df["density_score"]  = df["density_score"].fillna(0.0).astype(float)
    # pagerank: fill missing with 0.0 (lowest authority — conservative)
    X_df["pagerank_score"] = df["pagerank_score"].fillna(0.0).astype(float) if "pagerank_score" in df.columns else 0.0
    # Boolean fields: NaN means check didn't run → treat as passing (1.0)
    X_df["metadata_valid"] = df["metadata_valid"].fillna(1.0).astype(float)
    X_df["name_score"]     = df["name_score"].fillna(1.0).astype(float)
    X_df["temporal_valid"] = df["temporal_valid"].fillna(1.0).astype(float)

    X = X_df.values
    ids = df["benchmark_id"].tolist()
    return X, y, ids


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",   # compensate for REAL-heavy benchmark
        max_iter=1000,
        random_state=42,
    )

    # Cross-validate before final fit
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    logger.info(
        "5-fold CV AUC: %.4f ± %.4f  (min=%.4f  max=%.4f)",
        aucs.mean(), aucs.std(), aucs.min(), aucs.max(),
    )

    # Final fit on all data
    model.fit(X_scaled, y)

    return scaler, model


def print_weights(model: LogisticRegression, scaler: StandardScaler) -> None:
    print("\n── Feature weights (logistic regression coefficients) ──")
    print(f"  {'Feature':<20} {'Coeff':>10}  {'Scaled σ':>10}")
    print("  " + "-" * 44)
    for feat, coef, scale in zip(FEATURES, model.coef_[0], scaler.scale_):
        # Positive coef → higher value → more likely hallucinated
        # Negative coef → higher value → more likely real
        print(f"  {feat:<20} {coef:>+10.4f}  (σ={scale:.4f})")
    print(f"  {'Intercept':<20} {model.intercept_[0]:>+10.4f}")


def print_calibration(
    model: LogisticRegression, scaler: StandardScaler,
    X: np.ndarray, y: np.ndarray,
) -> None:
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\n── Classification report (full dataset, threshold=0.5) ──")
    print(classification_report(y, preds, target_names=["REAL", "HALLUCINATED/SUSPICIOUS"]))
    print(f"  ROC-AUC: {roc_auc_score(y, probs):.4f}")

    # Calibration buckets
    print("\n── Calibration check (predicted P vs. actual rate) ──")
    print(f"  {'P range':<15} {'n':>6}  {'actual rate':>12}")
    print("  " + "-" * 36)
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        mask = (probs >= lo) & (probs < hi)
        n = mask.sum()
        if n > 0:
            actual = y[mask].mean()
            print(f"  [{lo:.1f}, {hi:.1f})      {n:>6}  {actual:>12.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not RAW_SCORES_PATH.exists():
        logger.error(
            "raw_scores.csv not found at %s — run 'python -m benchmark.evaluate' first.",
            RAW_SCORES_PATH,
        )
        sys.exit(1)

    X, y, _ids = load_features(RAW_SCORES_PATH)
    logger.info(
        "Feature matrix: %d samples × %d features  (%d positive / %d negative)",
        X.shape[0], X.shape[1], y.sum(), (1 - y).sum(),
    )

    scaler, model = train(X, y)
    print_weights(model, scaler)
    print_calibration(model, scaler, X, y)

    # Save bundle
    SCORER_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"scaler": scaler, "model": model, "features": FEATURES}
    with open(SCORER_PATH, "wb") as f:
        pickle.dump(bundle, f)
    logger.info("Scorer saved → %s", SCORER_PATH)
    print(f"\n✅  Scorer trained and saved to {SCORER_PATH}")
    print(f"   Run  python -m benchmark.report  to evaluate on the held-out test set.\n")


if __name__ == "__main__":
    main()
