"""
detector/temporal_check.py

Phase 2, Step 2.3 — Temporal Precedential Logic.

Two lightweight checks that catch impossible or suspect citations:

  1. Future-citation check:
     If the year in the citation string is in the future relative to today,
     the citation cannot be real → SUSPICIOUS signal.

  2. Year-inversion check:
     If the year extracted from the citation string pre-dates the year the
     cited case was actually decided (stored on the Neo4j Case node), the
     citation is logically impossible → SUSPICIOUS signal.
     Example: citing "United States v. Jones, 565 U.S. 400 (2009)" when
     the Jones decision was issued in 2012.

Both checks reuse MetadataResult.actual_year and MetadataResult.cited_year
which are already fetched by metadata_check.py (Layer 4) — no additional
database round trip required.

Called by detector/pipeline.py after Layer 4 has run.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

_CURRENT_YEAR: int = date.today().year


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TemporalResult:
    cited_year:  Optional[int]   # year extracted from citation string
    actual_year: Optional[int]   # year case was actually decided (from Neo4j)
    is_valid:    bool            # False → SUSPICIOUS signal
    checked:     bool            # False if neither year could be determined
    reason:      Optional[str]   # "future_citation" | "year_inversion" | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_temporal(
    cited_year:  Optional[int],
    actual_year: Optional[int],
) -> TemporalResult:
    """
    Validate citation year against the actual case year and the current date.

    Args:
        cited_year:  year extracted from the citation string by metadata_check
        actual_year: year the case was actually decided, from the Neo4j node

    Returns:
        TemporalResult with is_valid=False when a temporal anomaly is detected.
        checked=False when neither year is available (cannot evaluate).
    """
    # Cannot check if no year information is available
    if cited_year is None and actual_year is None:
        logger.debug("Temporal check skipped — no year data available")
        return TemporalResult(
            cited_year=None, actual_year=None,
            is_valid=True, checked=False, reason=None,
        )

    # Check 1: Future citation — cited year is after the current calendar year
    if cited_year is not None and cited_year > _CURRENT_YEAR:
        logger.info(
            "Temporal FAIL: cited_year=%d is in the future (current=%d)",
            cited_year, _CURRENT_YEAR,
        )
        return TemporalResult(
            cited_year=cited_year, actual_year=actual_year,
            is_valid=False, checked=True, reason="future_citation",
        )

    # Check 2: Year inversion — citation year pre-dates the actual decision
    if cited_year is not None and actual_year is not None:
        if cited_year < actual_year:
            logger.info(
                "Temporal FAIL: cited_year=%d predates actual_year=%d (year inversion)",
                cited_year, actual_year,
            )
            return TemporalResult(
                cited_year=cited_year, actual_year=actual_year,
                is_valid=False, checked=True, reason="year_inversion",
            )

    logger.debug(
        "Temporal OK: cited_year=%s actual_year=%s", cited_year, actual_year
    )
    return TemporalResult(
        cited_year=cited_year, actual_year=actual_year,
        is_valid=True, checked=(cited_year is not None or actual_year is not None),
        reason=None,
    )
