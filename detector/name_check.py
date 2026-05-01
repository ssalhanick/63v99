"""
detector/name_check.py

Phase 1, Step 1.3 — Party Name Consistency Check.

For a citation that passes Layer 1 (case exists in Neo4j), compares the party
names in the citation string against the case name stored on the Neo4j Case node.

Catches a class of hallucination that Layers 1-4 miss:
    - Correct case_id (case exists)
    - Wrong party names (e.g. "United States v. Smith" for a case about Jones)

Uses rapidfuzz token_sort_ratio to handle minor formatting differences between
citation strings ("Smith v. Jones") and node names ("JONES v. SMITH" or
"Jones v. Smith, No. 19-1234").

Verdict contribution:
    is_valid = True   → name is consistent (no signal added)
    is_valid = False  → name mismatch below NAME_SCORE_THRESHOLD → SUSPICIOUS signal
    checked  = False  → no party names extractable (citation is pure reporter format)

Called by detector/pipeline.py after Layer 1 passes and node_name is available.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from config import NAME_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NameCheckResult:
    cited_name:  Optional[str]   # party names extracted from citation string
    node_name:   Optional[str]   # case name from Neo4j node
    score:       float           # rapidfuzz token_sort_ratio (0.0 – 1.0)
    is_valid:    bool            # False → SUSPICIOUS signal
    checked:     bool            # False if no party names extractable


# ---------------------------------------------------------------------------
# Party name extraction
# ---------------------------------------------------------------------------

def _extract_party_names(citation_string: str) -> Optional[str]:
    """
    Extract the party-name portion from a citation string.

    Handles:
      - "United States v. Jones, 392 U.S. 1 (1968)"   → "United States v. Jones"
      - "Smith v. Johnson, 923 F.3d 1027 (9th Cir. 2019)" → "Smith v. Johnson"
      - "392 U.S. 1"                                   → None (pure reporter)
      - "United States v. Jones"                       → "United States v. Jones"

    Returns None if no " v. " pattern is found (pure reporter citation).
    """
    # Match "X v. Y" at the start of the string, stopping before a comma or reporter
    match = re.match(
        r"^([A-Za-z][^,]+?\s+v\.\s+[A-Za-z][^,\d]+?)(?:,|\s+\d|\s*$)",
        citation_string.strip(),
    )
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_name(
    citation_string: str,
    node_name: Optional[str],
) -> NameCheckResult:
    """
    Compare party names in the citation string against the Neo4j node name.

    Args:
        citation_string: raw citation as it appears in the AI-generated text
        node_name:       case name from the Neo4j Case node (may be None)

    Returns:
        NameCheckResult — is_valid=False if score < NAME_SCORE_THRESHOLD
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        logger.warning(
            "rapidfuzz not installed — name check skipped. Run: pip install rapidfuzz"
        )
        return NameCheckResult(
            cited_name=None, node_name=node_name,
            score=1.0, is_valid=True, checked=False
        )

    cited_name = _extract_party_names(citation_string)

    # Cannot check — pure reporter citation like "392 U.S. 1"
    if not cited_name:
        logger.debug("Name check skipped — no party names in: %s", citation_string)
        return NameCheckResult(
            cited_name=None, node_name=node_name,
            score=1.0, is_valid=True, checked=False
        )

    # Node name unavailable — cannot penalize
    if not node_name:
        logger.debug("Name check skipped — no node_name for: %s", citation_string)
        return NameCheckResult(
            cited_name=cited_name, node_name=None,
            score=1.0, is_valid=True, checked=False
        )

    score = fuzz.token_sort_ratio(cited_name.lower(), node_name.lower()) / 100.0
    is_valid = score >= NAME_SCORE_THRESHOLD

    if not is_valid:
        logger.info(
            "Name MISMATCH: cited=%r node=%r score=%.2f (threshold=%.2f)",
            cited_name, node_name, score, NAME_SCORE_THRESHOLD,
        )
    else:
        logger.debug(
            "Name OK: cited=%r node=%r score=%.2f",
            cited_name, node_name, score,
        )

    return NameCheckResult(
        cited_name=cited_name,
        node_name=node_name,
        score=round(score, 3),
        is_valid=is_valid,
        checked=True,
    )
