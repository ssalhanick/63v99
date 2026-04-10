"""
visualization/umap_viz.py
Week 9 — UMAP dimensionality reduction of corpus embeddings.

Provides:
    load_umap_figure()  — cached Plotly figure of full corpus in 2D
    overlay_citations() — add submitted citation points as a highlighted overlay

Designed to be imported by frontend/app.py and cached with @st.cache_resource.

Standalone usage (generates HTML file for writeup):
    python -m visualization.umap_viz
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR

log = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path(PROCESSED_DIR) / "embeddings.parquet"
METADATA_PATH   = Path(PROCESSED_DIR) / "cases_cleaned.parquet"

METADATA_COLS = ["case_id", "case_name", "court_id", "date_filed", "cite_count"]

# UMAP parameters — match PROJECT_CONTEXT.md spec
UMAP_PARAMS = {
    "n_neighbors": 15,
    "min_dist":    0.1,
    "metric":      "cosine",
    "random_state": 42,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_embeddings() -> tuple[np.ndarray, list[int]]:
    """
    Load embeddings.parquet and return (matrix, case_ids).
    matrix shape: (n_cases, 768)
    """
    log.info("Loading embeddings from %s", EMBEDDINGS_PATH)
    df = pd.read_parquet(EMBEDDINGS_PATH, columns=["case_id", "embedding"])
    case_ids = df["case_id"].tolist()
    matrix   = np.vstack(df["embedding"].values)
    log.info("  Loaded %d embeddings, shape %s", len(case_ids), matrix.shape)
    return matrix, case_ids


def _load_metadata(case_ids: list[int]) -> pd.DataFrame:
    """Load metadata for the embedded cases, indexed by case_id."""
    log.info("Loading metadata from %s", METADATA_PATH)
    df = pd.read_parquet(METADATA_PATH, columns=METADATA_COLS)
    df = df[df["case_id"].isin(set(case_ids))].set_index("case_id")
    df["year"] = df["date_filed"].str[:4].fillna("unknown")
    log.info("  Metadata loaded: %d cases", len(df))
    return df


# ── Dimensionality reduction ──────────────────────────────────────────────────

def _run_umap(matrix: np.ndarray) -> np.ndarray:
    """
    Apply StandardScaler then UMAP to reduce 768-dim embeddings to 2D.

    StandardScaler first — zero mean, unit variance per dimension.
    Without this, high-variance BERT dimensions dominate the UMAP distance
    calculation and distort the 2D layout.

    Returns array of shape (n_cases, 2).
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from umap import UMAP
    except ImportError:
        raise ImportError(
            "umap-learn and scikit-learn required. "
            "Run: pip install umap-learn scikit-learn"
        )

    log.info("Applying StandardScaler...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)   # (n, 768), zero mean unit variance

    log.info("Running UMAP (n_neighbors=%d, metric=%s)...", UMAP_PARAMS["n_neighbors"], UMAP_PARAMS["metric"])
    reducer = UMAP(**UMAP_PARAMS)
    coords  = reducer.fit_transform(scaled)  # (n, 2)
    log.info("  UMAP complete, output shape: %s", coords.shape)
    return coords


# ── Plotly figure builder ─────────────────────────────────────────────────────

def _circuit_label(court_id: str) -> str:
    """Map CourtListener court_id to a readable circuit label for coloring."""
    mapping = {
        "ca1": "1st Cir", "ca2": "2nd Cir", "ca3": "3rd Cir",
        "ca4": "4th Cir", "ca5": "5th Cir", "ca6": "6th Cir",
        "ca7": "7th Cir", "ca8": "8th Cir", "ca9": "9th Cir",
        "ca10": "10th Cir", "ca11": "11th Cir", "cadc": "DC Cir",
        "cafc": "Fed Cir", "scotus": "SCOTUS",
    }
    return mapping.get(str(court_id).lower(), "State/Other")


def build_corpus_figure(color_by: str = "circuit"):
    """
    Build and return a Plotly scatter figure of the full corpus in 2D UMAP space.

    Args:
        color_by: "circuit" | "year" — controls point coloring

    Returns:
        plotly.graph_objects.Figure
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required. Run: pip install plotly")

    matrix, case_ids = _load_embeddings()
    metadata         = _load_metadata(case_ids)
    coords           = _run_umap(matrix)

    # Assemble plot dataframe
    rows = []
    for i, cid in enumerate(case_ids):
        x, y = float(coords[i, 0]), float(coords[i, 1])
        if cid in metadata.index:
            row = metadata.loc[cid]
            name     = str(row["case_name"]) if pd.notna(row["case_name"]) else "Unknown"
            court_id = str(row["court_id"])  if pd.notna(row["court_id"])  else "unknown"
            year     = str(row["year"])
            cite_count = int(row["cite_count"]) if pd.notna(row.get("cite_count")) else 0
        else:
            name = "Unknown"; court_id = "unknown"; year = "unknown"; cite_count = 0

        rows.append({
            "x":        x,
            "y":        y,
            "case_id":  cid,
            "name":     name,
            "court_id": court_id,
            "circuit":  _circuit_label(court_id),
            "year":     year,
            "cite_count": cite_count,
            "hover": f"{name}<br>{_circuit_label(court_id)}, {year}<br>cite_count: {cite_count}",
        })

    df = pd.DataFrame(rows)

    color_col = "circuit" if color_by == "circuit" else "year"

    fig = px.scatter(
        df,
        x="x", y="y",
        color=color_col,
        hover_name="name",
        hover_data={
            "x": False, "y": False,
            "court_id": True, "year": True, "cite_count": True,
        },
        title=f"Verit Corpus — UMAP Embedding Space (colored by {color_by})",
        labels={"x": "UMAP-1", "y": "UMAP-2"},
        template="plotly_white",
        opacity=0.75,
    )

    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        height=600,
        legend_title_text=color_col.capitalize(),
        font=dict(family="Inter, sans-serif", size=12),
    )

    return fig, coords, case_ids, df


def overlay_submitted_citations(
    base_fig,
    coords:       np.ndarray,
    case_ids:     list[int],
    verdicts:     list[dict],
) -> None:
    """
    Add submitted citation points as an overlay on the base corpus figure.
    Colors: green=REAL, yellow=SUSPICIOUS, red=HALLUCINATED.

    Modifies base_fig in place.

    Args:
        base_fig:   Plotly figure from build_corpus_figure()
        coords:     UMAP 2D coordinates from the same build call
        case_ids:   List of corpus case_ids in same order as coords
        verdicts:   List of verdict dicts from the API response
                    (must have case_id and verdict fields)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required.")

    color_map = {
        "REAL":         "#2e7d32",
        "SUSPICIOUS":   "#f57c00",
        "HALLUCINATED": "#c62828",
    }
    symbol_map = {
        "REAL":         "star",
        "SUSPICIOUS":   "diamond",
        "HALLUCINATED": "x",
    }

    id_to_coord = {cid: coords[i] for i, cid in enumerate(case_ids)}

    for v in verdicts:
        cid     = v.get("case_id")
        verdict = v.get("verdict", "UNKNOWN")
        name    = v.get("citation_string", "Unknown")

        if cid is None or cid not in id_to_coord:
            continue   # hallucinated cases won't have coords — skip

        x, y = float(id_to_coord[cid][0]), float(id_to_coord[cid][1])

        base_fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers",
            marker=dict(
                color=color_map.get(verdict, "gray"),
                size=14,
                symbol=symbol_map.get(verdict, "circle"),
                line=dict(color="white", width=1.5),
            ),
            name=f"{verdict}: {name[:40]}",
            hovertext=f"{name}<br>Verdict: {verdict}",
            hoverinfo="text",
            showlegend=True,
        ))


# ── Standalone export ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(__file__).parent
    for color_by in ("circuit", "year"):
        log.info("Building UMAP figure colored by %s...", color_by)
        fig, _, _, _ = build_corpus_figure(color_by=color_by)
        out_path = output_dir / f"umap_{color_by}.html"
        fig.write_html(str(out_path))
        log.info("Saved → %s", out_path)

    print("\n✅  UMAP figures saved to visualization/")
    print("    Open umap_circuit.html and umap_year.html in your browser.")