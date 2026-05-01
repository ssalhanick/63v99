"""
detector/pipeline.py

Hallucination detection pipeline — orchestrates all three layers.

Entry point: run_pipeline(text) → list[CitationVerdict]

Flow per citation:
    1. EyeCite extracts + resolves citations from raw text
    2. Layer 1 (existence_check)     — does the case exist in Neo4j?
    3. Layer 2a (semantic_check)     — is the context Fourth Amendment domain?
    4. Layer 2b (llm_check)          — does the proposition match this case's holding?
    5. Layer 3 (connectivity_check)  — does the case have citation footprint?

Verdict logic:
    Layer 1 FAIL                         → HALLUCINATED (stop)
    Layer 1 PASS, L2a PASS, L2b PASS, L3 PASS → REAL
    Layer 1 PASS, any of L2a/L2b/L3 FAIL     → SUSPICIOUS

Phase 4 additions:
    After per-citation loop — cross_citation.compute_cross_citation_signals()
    stamps each verdict with mean_jaccard and min_hop_distance.

Usage:
    python -m detector.pipeline
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, SCORER_PATH
from detector.eyecite_parser import parse_citations, ResolvedCitation
from detector.existence_check import check_existence
from detector.semantic_check import semantic_check, SemanticResult
from detector.llm_check import llm_check, LLMResult
from detector.connectivity_check import check_connectivity, ConnectivityResult
from detector.metadata_check import check_metadata, MetadataResult
from detector.name_check import check_name, NameCheckResult
from detector.temporal_check import check_temporal, TemporalResult
from detector.cross_citation import (
    compute_cross_citation_signals,
    CrossCitationSignal,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Learned scorer — lazy loaded from SCORER_PATH (set in config.py)
# Falls back to boolean threshold logic if the pkl is not yet available.
# ---------------------------------------------------------------------------

_scorer_bundle = None   # {"scaler": ..., "model": ..., "features": [...]}
_scorer_loaded = False  # sentinel so we only attempt load once


def _get_scorer() -> Optional[dict]:
    """Return the scorer bundle (scaler + model) or None if not available."""
    global _scorer_bundle, _scorer_loaded
    if _scorer_loaded:
        return _scorer_bundle
    _scorer_loaded = True
    path = Path(SCORER_PATH)
    if path.exists():
        with open(path, "rb") as f:
            _scorer_bundle = pickle.load(f)
        logger.info("Scorer loaded from %s", path)
    else:
        logger.info(
            "Scorer not found at %s — using boolean verdict logic. "
            "Run python -m benchmark.train_scorer to enable calibrated scoring.",
            path,
        )
    return _scorer_bundle


# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

REAL         = "REAL"
SUSPICIOUS   = "SUSPICIOUS"
HALLUCINATED = "HALLUCINATED"

# Probability thresholds for the learned scorer
_P_HALLUCINATED = 0.70   # P(hallucinated) >= this → HALLUCINATED
_P_SUSPICIOUS   = 0.40   # P(hallucinated) >= this → SUSPICIOUS

@dataclass
class CitationVerdict:
    # From EyeCite parser
    citation_string: str
    case_name:       Optional[str]
    case_id:         Optional[int]

    # Layer results
    exists:          bool                           # Layer 1
    semantic:        Optional[SemanticResult]       # Layer 2a (None if L1/L4 failed)
    llm_result:      Optional[LLMResult]            # Layer 2b (None if L1/L4/L2a failed)
    connectivity:    Optional[ConnectivityResult]   # Layer 3  (None if L1/L4 failed)
    metadata:        Optional[MetadataResult]       # Layer 4  (None if L1 failed)
    name_check:      Optional[NameCheckResult]      # Name check (None if L1 failed)
    temporal:        Optional[TemporalResult]       # Temporal check (None if L1 failed)

    # Final verdict
    verdict:         str                            # REAL | SUSPICIOUS | HALLUCINATED

    # Context passed to layers
    context_text:    str

    # Top corpus matches from Layer 2a
    top_matches:     list[dict] = field(default_factory=list)

    # Phase 4 — cross-citation graph signals (None if only one citation in document)
    cross_jaccard_score:  Optional[float] = None   # mean Jaccard vs. co-citations
    min_hop_distance:     Optional[int]   = None   # shortest CITES path to nearest co-citation
    has_doctrines:        bool            = False  # has >= 1 doctrine label
    mean_shared_doctrines: Optional[float] = None  # mean shared doctrines with co-citations


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _compute_verdict(
    exists:       bool,
    semantic:     Optional[SemanticResult],
    llm_result:   Optional[LLMResult],
    connectivity: Optional[ConnectivityResult],
    metadata:     Optional[MetadataResult],
    name_check:   Optional[NameCheckResult],
    temporal:     Optional[TemporalResult],
) -> str:
    """
    Compute the final verdict using the learned scorer if available,
    otherwise fall back to boolean threshold fusion.

    Hard rules (always applied regardless of scorer):
      - Layer 1 FAIL  → HALLUCINATED
      - Layer 4 FAIL  → HALLUCINATED
    """
    if not exists:
        return HALLUCINATED

    # L4: metadata mismatch is always a hard HALLUCINATED signal
    if metadata is not None and metadata.checked and not metadata.is_valid:
        return HALLUCINATED

    # L2b: LLM explicitly rejected the proposition -> hard SUSPICIOUS signal
    if llm_result is not None and not llm_result.skipped and not llm_result.is_accurate:
        return SUSPICIOUS

    # --- Attempt learned scorer ---
    bundle = _get_scorer()
    if bundle is not None:
        scaler, model = bundle["scaler"], bundle["model"]
        x = np.array([[
            float(exists),
            semantic.rrf_score          if semantic                       else 0.0,
            semantic.top_dense_score    if semantic                       else 0.0,
            semantic.case_sim           if semantic and semantic.case_sim is not None else 0.0,
            float(connectivity.density_score) if connectivity             else 0.0,
            float(connectivity.pagerank_score) if (connectivity and connectivity.pagerank_score is not None) else 0.0,
            float(metadata.is_valid)    if metadata  and metadata.checked else 1.0,
            name_check.score            if name_check and name_check.checked else 1.0,
            float(temporal.is_valid)    if temporal  and temporal.checked else 1.0,
        ]])
        # Scorer may have been trained without pagerank_score (Phase 3 model).
        # Trim the feature vector to match the stored model's expected width.
        expected_n = scaler.n_features_in_
        if x.shape[1] > expected_n:
            x = x[:, :expected_n]
        p_hallucinated = model.predict_proba(scaler.transform(x))[0][1]
        logger.info("Scorer P(hallucinated)=%.4f", p_hallucinated)
        if p_hallucinated >= _P_HALLUCINATED:
            return HALLUCINATED
        if p_hallucinated >= _P_SUSPICIOUS:
            return SUSPICIOUS
        return REAL

    # --- Boolean fallback ---
    l2a      = semantic.is_relevant          if semantic                              else False
    l2b      = llm_result.is_accurate        if llm_result and not llm_result.skipped else True
    l3       = connectivity.is_connected     if connectivity                           else False
    name_ok  = name_check.is_valid           if name_check and name_check.checked     else True
    temp_ok  = temporal.is_valid             if temporal   and temporal.checked       else True

    if l2a and l2b and l3 and name_ok and temp_ok:
        return REAL
    else:
        return SUSPICIOUS


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(text: str, court_filter: Optional[list[str]] = None) -> list[CitationVerdict]:
    """
    Run the full hallucination detection pipeline on raw AI-generated text.

    Args:
        text: raw legal text potentially containing AI-generated citations

    Returns:
        List of CitationVerdict objects, one per unique full citation found.
        Empty list if no full citations are found.
    """
    logger.info("Pipeline starting — extracting citations from input text")

    citations: list[ResolvedCitation] = parse_citations(text)

    if not citations:
        logger.info("No full citations found in input text")
        return []

    logger.info("Processing %d citation(s)", len(citations))

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    verdicts = []

    try:
        for citation in citations:
            logger.info("--- Checking: %s ---", citation.citation_string)

            # Layer 1 — existence
            exists, node_name = check_existence(citation.case_id, driver=driver)

            if not exists:
                logger.info("Layer 1 FAIL → HALLUCINATED: %s", citation.citation_string)
                verdicts.append(CitationVerdict(
                    citation_string = citation.citation_string,
                    case_name       = citation.case_name,
                    case_id         = citation.case_id,
                    exists          = False,
                    semantic        = None,
                    llm_result      = None,
                    connectivity    = None,
                    metadata        = None,
                    name_check      = None,
                    temporal        = None,
                    verdict         = HALLUCINATED,
                    context_text    = citation.context_text,
                    top_matches     = [],
                ))
                continue

            # Layer 4 — metadata validation (runs before LLM to save tokens on Type B)
            meta = check_metadata(citation.case_id, citation.citation_string, driver=driver)

            # Name check — runs alongside L4 (uses node_name from existence_check)
            name = check_name(citation.citation_string, node_name)

            # Temporal check — reuses year fields already fetched by L4 (no extra DB query)
            temporal = check_temporal(
                cited_year  = meta.cited_year,
                actual_year = meta.actual_year,
            )

            if meta.checked and not meta.is_valid:
                logger.info(
                    "Layer 4 FAIL → HALLUCINATED: %s (year=%s court=%s)",
                    citation.citation_string, meta.cited_year, meta.cited_court,
                )
                verdicts.append(CitationVerdict(
                    citation_string = citation.citation_string,
                    case_name       = citation.case_name,
                    case_id         = citation.case_id,
                    exists          = True,
                    semantic        = None,
                    llm_result      = None,
                    connectivity    = None,
                    metadata        = meta,
                    name_check      = name,
                    temporal        = temporal,
                    verdict         = HALLUCINATED,
                    context_text    = citation.context_text,
                    top_matches     = [],
                ))
                continue

            # Layer 2a — semantic domain check
            semantic = semantic_check(
                citation.context_text,
                case_id=citation.case_id,
                court_filter=court_filter,
            )

            # Layer 2b — LLM proposition accuracy check
            # Only run if 2a passes — saves tokens on clearly off-domain text
            if semantic.is_relevant:
                llm_result = llm_check(citation.case_id, citation.context_text)
            else:
                logger.info("Layer 2a FAIL — skipping LLM check for %s", citation.citation_string)
                llm_result = None

            # Layer 3 — citation connectivity
            connectivity = check_connectivity(citation.case_id, driver=driver)

            # Final verdict
            verdict = _compute_verdict(exists, semantic, llm_result, connectivity, meta, name, temporal)

            logger.info(
                "Verdict: %s | L1=%s L4=%s L2a=%s (rrf=%.4f) L2b=%s L3=%s (density=%d) name=%.2f temp=%s",
                verdict,
                exists,
                meta.is_valid if meta and meta.checked else "skip",
                semantic.is_relevant,
                semantic.rrf_score,
                llm_result.is_accurate if llm_result and not llm_result.skipped else "skipped",
                connectivity.is_connected,
                connectivity.density_score,
                name.score if name and name.checked else 1.0,
                temporal.reason if temporal and not temporal.is_valid else "ok",
            )

            verdicts.append(CitationVerdict(
                citation_string = citation.citation_string,
                case_name       = citation.case_name,
                case_id         = citation.case_id,
                exists          = exists,
                semantic        = semantic,
                llm_result      = llm_result,
                connectivity    = connectivity,
                metadata        = meta,
                name_check      = name,
                temporal        = temporal,
                verdict         = verdict,
                context_text    = citation.context_text,
                top_matches     = semantic.top_matches,
            ))

    finally:
        driver.close()

    # -----------------------------------------------------------------------
    # Phase 4 — cross-citation post-processing
    # -----------------------------------------------------------------------
    # Collect all case_ids that passed Layer 1 and got a ConnectivityResult.
    # (We only run cross-citation on cases that exist in the graph.)
    l1_pass_ids: list[Optional[int]] = [
        v.case_id for v in verdicts if v.exists and v.case_id is not None
    ]

    if len(l1_pass_ids) >= 2:
        logger.info(
            "Phase 4 — running cross-citation analysis on %d confirmed case_ids",
            len(l1_pass_ids),
        )
        # Re-use the driver (already closed above) — open a fresh one just for
        # the cross-citation batch; this keeps the per-citation loop clean.
        xc_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        try:
            xc_signals = compute_cross_citation_signals(l1_pass_ids, driver=xc_driver)
        finally:
            xc_driver.close()

        # Stamp each verdict with its cross-citation signals
        for verdict in verdicts:
            if verdict.case_id is not None and verdict.case_id in xc_signals:
                sig = xc_signals[verdict.case_id]
                verdict.cross_jaccard_score = sig.mean_jaccard
                verdict.min_hop_distance    = sig.min_hop_distance
                verdict.has_doctrines       = sig.has_doctrines
                verdict.mean_shared_doctrines = sig.mean_shared_doctrines
                logger.info(
                    "Phase 4 — %s: jaccard=%.4f  min_hop=%s  docs=%s  shared=%.2f",
                    verdict.citation_string,
                    sig.mean_jaccard if sig.mean_jaccard is not None else 0.0,
                    sig.min_hop_distance,
                    sig.has_doctrines,
                    sig.mean_shared_doctrines if sig.mean_shared_doctrines is not None else 0.0,
                )
    else:
        logger.info(
            "Phase 4 — fewer than 2 confirmed citations — skipping cross-citation analysis"
        )

    return verdicts


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample = """
    The Fourth Amendment protects individuals from unreasonable searches and seizures.
    In Terry v. Ohio, 392 U.S. 1 (1968), the Court held that a brief investigatory stop
    is permissible when an officer has reasonable articulable suspicion. This standard
    was later extended in United States v. Sokolow, 490 U.S. 1 (1989), which addressed
    drug courier profiles. A fabricated citation to United States v. Torres, 923 F.3d 1027
    (9th Cir. 2019) would not resolve to a valid opinion.
    """

    results = run_pipeline(sample)

    print(f"\n{'='*75}")
    print(f"{'Citation':<25} {'Verdict':<14} {'L1':>4} {'L2a':>5} {'L2b':>5} {'L3':>5} {'Density':>7}")
    print("-" * 75)
    for v in results:
        l2a     = v.semantic.is_relevant if v.semantic else "-"
        l2b     = v.llm_result.is_accurate if v.llm_result and not v.llm_result.skipped else "skip"
        l3      = v.connectivity.is_connected if v.connectivity else "-"
        density = v.connectivity.density_score if v.connectivity else "-"
        print(
            f"{v.citation_string:<25} {v.verdict:<14} "
            f"{str(v.exists):>4} {str(l2a):>5} {str(l2b):>5} {str(l3):>5} {str(density):>7}"
        )
    print(f"{'='*75}")