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

Usage:
    python -m detector.pipeline
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from detector.eyecite_parser import parse_citations, ResolvedCitation
from detector.existence_check import check_existence
from detector.semantic_check import semantic_check, SemanticResult
from detector.llm_check import llm_check, LLMResult
from detector.connectivity_check import check_connectivity, ConnectivityResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

REAL         = "REAL"
SUSPICIOUS   = "SUSPICIOUS"
HALLUCINATED = "HALLUCINATED"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CitationVerdict:
    # From EyeCite parser
    citation_string: str
    case_name:       Optional[str]
    case_id:         Optional[int]

    # Layer results
    exists:          bool                         # Layer 1
    semantic:        Optional[SemanticResult]     # Layer 2a (None if L1 failed)
    llm_result:      Optional[LLMResult]          # Layer 2b (None if L1/L2a failed)
    connectivity:    Optional[ConnectivityResult] # Layer 3  (None if L1 failed)

    # Final verdict
    verdict:         str                          # REAL | SUSPICIOUS | HALLUCINATED

    # Context passed to layers
    context_text:    str

    # Top corpus matches from Layer 2a
    top_matches:     list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _compute_verdict(
    exists:       bool,
    semantic:     Optional[SemanticResult],
    llm_result:   Optional[LLMResult],
    connectivity: Optional[ConnectivityResult],
) -> str:
    """
    Layer 1 FAIL → HALLUCINATED immediately.
    Layer 1 PASS → worst case is SUSPICIOUS (case is real, proposition may be wrong).

    REAL requires all of: L2a PASS, L2b PASS (or skipped), L3 PASS.
    Anything else → SUSPICIOUS.
    """
    if not exists:
        return HALLUCINATED

    l2a = semantic.is_relevant          if semantic     else False
    l2b = llm_result.is_accurate        if llm_result and not llm_result.skipped else True
    l3  = connectivity.is_connected     if connectivity else False

    if l2a and l2b and l3:
        return REAL
    else:
        return SUSPICIOUS


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(text: str) -> list[CitationVerdict]:
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
            exists = check_existence(citation.case_id, driver=driver)

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
                    verdict         = HALLUCINATED,
                    context_text    = citation.context_text,
                    top_matches     = [],
                ))
                continue

            # Layer 2a — semantic domain check
            semantic = semantic_check(citation.context_text)

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
            verdict = _compute_verdict(exists, semantic, llm_result, connectivity)

            logger.info(
                "Verdict: %s | L1=%s L2a=%s (rrf=%.4f) L2b=%s L3=%s (density=%d)",
                verdict,
                exists,
                semantic.is_relevant,
                semantic.rrf_score,
                llm_result.is_accurate if llm_result and not llm_result.skipped else "skipped",
                connectivity.is_connected,
                connectivity.density_score,
            )

            verdicts.append(CitationVerdict(
                citation_string = citation.citation_string,
                case_name       = citation.case_name,
                case_id         = citation.case_id,
                exists          = exists,
                semantic        = semantic,
                llm_result      = llm_result,
                connectivity    = connectivity,
                verdict         = verdict,
                context_text    = citation.context_text,
                top_matches     = semantic.top_matches,
            ))

    finally:
        driver.close()

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