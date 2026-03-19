"""
detector/pipeline.py

Hallucination detection pipeline — orchestrates all three layers.

Entry point: run_pipeline(text) → list[CitationVerdict]

Flow per citation:
    1. EyeCite extracts + resolves citations from raw text
    2. Layer 1 (existence_check)     — does the case exist in Neo4j?
    3. Layer 2 (semantic_check)      — is the context semantically consistent?
    4. Layer 3 (connectivity_check)  — does the case have citation footprint?

Verdict logic:
    Layer 1 FAIL                        → HALLUCINATED (stop, skip Layers 2+3)
    Layer 1 PASS, Layer 2+3 both PASS   → REAL
    Layer 1 PASS, Layer 2+3 both FAIL   → HALLUCINATED
    Layer 1 PASS, Layer 2 XOR Layer 3   → SUSPICIOUS

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
    exists:          bool                        # Layer 1
    semantic:        Optional[SemanticResult]    # Layer 2 (None if Layer 1 failed)
    connectivity:    Optional[ConnectivityResult]# Layer 3 (None if Layer 1 failed)

    # Final verdict
    verdict:         str                         # REAL | SUSPICIOUS | HALLUCINATED

    # Context passed to Layer 2
    context_text:    str

    # Top corpus matches from Layer 2 (for RAG context in LLM layer)
    top_matches:     list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _compute_verdict(
    exists: bool,
    semantic: Optional[SemanticResult],
    connectivity: Optional[ConnectivityResult],
) -> str:
    """
    Combine layer results into a final verdict.

    Layer 1 FAIL → HALLUCINATED immediately.
    Both Layer 2 and Layer 3 PASS → REAL.
    Both Layer 2 and Layer 3 FAIL → HALLUCINATED.
    One passes, one fails → SUSPICIOUS.
    """
    if not exists:
        return HALLUCINATED

    l2 = semantic.is_relevant if semantic else False
    l3 = connectivity.is_connected if connectivity else False

    if l2 and l3:
        return REAL
    elif not l2 and not l3:
        return HALLUCINATED
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

    # Parse + resolve citations
    citations: list[ResolvedCitation] = parse_citations(text)

    if not citations:
        logger.info("No full citations found in input text")
        return []

    logger.info("Processing %d citation(s)", len(citations))

    # Shared Neo4j driver — reused across all citations
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    verdicts = []

    try:
        for citation in citations:
            logger.info("--- Checking: %s ---", citation.citation_string)

            # Layer 1 — existence
            exists = check_existence(citation.case_id, driver=driver)

            if not exists:
                # Short-circuit — no point running Layers 2 and 3
                logger.info(
                    "Layer 1 FAIL → HALLUCINATED: %s", citation.citation_string
                )
                verdicts.append(CitationVerdict(
                    citation_string = citation.citation_string,
                    case_name       = citation.case_name,
                    case_id         = citation.case_id,
                    exists          = False,
                    semantic        = None,
                    connectivity    = None,
                    verdict         = HALLUCINATED,
                    context_text    = citation.context_text,
                    top_matches     = [],
                ))
                continue

            # Layer 2 — semantic relevance
            semantic = semantic_check(citation.context_text)

            # Layer 3 — citation connectivity
            connectivity = check_connectivity(citation.case_id, driver=driver)

            # Final verdict
            verdict = _compute_verdict(exists, semantic, connectivity)

            logger.info(
                "Verdict: %s | L1=%s L2=%s (rrf=%.4f, dense=%.4f) L3=%s (density=%d)",
                verdict,
                exists,
                semantic.is_relevant,
                semantic.rrf_score,
                semantic.top_dense_score,
                connectivity.is_connected,
                connectivity.density_score,
            )

            verdicts.append(CitationVerdict(
                citation_string = citation.citation_string,
                case_name       = citation.case_name,
                case_id         = citation.case_id,
                exists          = exists,
                semantic        = semantic,
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
    import json
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

    print(f"\n{'='*65}")
    print(f"{'Citation':<25} {'Verdict':<14} {'L1':>4} {'L2':>5} {'L3':>5} {'Density':>7}")
    print("-" * 65)
    for v in results:
        l2 = v.semantic.is_relevant if v.semantic else "-"
        l3 = v.connectivity.is_connected if v.connectivity else "-"
        density = v.connectivity.density_score if v.connectivity else "-"
        print(
            f"{v.citation_string:<25} {v.verdict:<14} "
            f"{str(v.exists):>4} {str(l2):>5} {str(l3):>5} {str(density):>7}"
        )
    print(f"{'='*65}")