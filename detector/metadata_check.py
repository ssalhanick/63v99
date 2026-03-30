"""
detector/metadata_check.py
Week 8 — Layer 4: Metadata Validation

Checks whether the year and court mentioned in a citation string match the
actual properties stored on the Neo4j Case node. Catches Type B hallucinations
(real case, wrong year or court) that Layers 1–3 cannot detect because the
underlying case is real and well-connected.

Only runs when Layer 1 returns True (case exists). Skipped for numeric-only
citations like "392 U.S. 1" where no year/court can be extracted.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# ── court ID normalisation ────────────────────────────────────────────────────
# Maps common citation court strings → CourtListener court_id values
COURT_ALIASES: dict[str, list[str]] = {
    "ca1":  ["1st cir", "first circuit"],
    "ca2":  ["2nd cir", "second circuit"],
    "ca3":  ["3rd cir", "third circuit"],
    "ca4":  ["4th cir", "fourth circuit"],
    "ca5":  ["5th cir", "fifth circuit"],
    "ca6":  ["6th cir", "sixth circuit"],
    "ca7":  ["7th cir", "seventh circuit"],
    "ca8":  ["8th cir", "eighth circuit"],
    "ca9":  ["9th cir", "ninth circuit"],
    "ca10": ["10th cir", "tenth circuit"],
    "ca11": ["11th cir", "eleventh circuit"],
    "cadc": ["d.c. cir", "dc cir"],
    "cafc": ["fed. cir", "federal circuit"],
}

# Reverse lookup: alias → court_id
_ALIAS_TO_COURT: dict[str, str] = {
    alias: court_id
    for court_id, aliases in COURT_ALIASES.items()
    for alias in aliases
}


@dataclass
class MetadataResult:
    checked:        bool   # False if citation string had no extractable metadata
    year_match:     Optional[bool]   # None if year not in citation string
    court_match:    Optional[bool]   # None if court not in citation string
    cited_year:     Optional[int]    # year extracted from citation string
    cited_court:    Optional[str]    # court extracted from citation string
    actual_year:    Optional[int]    # year from Neo4j node
    actual_court:   Optional[str]    # court_id from Neo4j node
    is_valid:       bool             # False → flag as SUSPICIOUS


def _extract_year_from_citation(citation_string: str) -> Optional[int]:
    """
    Extract a 4-digit year from a citation string.
    Matches patterns like:
      - "(9th Cir. 2019)"
      - "(2022)"
      - "923 F.3d 1027 (4th Cir. 2019)"
    Returns None if no year found (e.g. pure reporter cites like "392 U.S. 1").
    """
    match = re.search(r"\b(19[0-9]{2}|20[0-9]{2})\b", citation_string)
    if match:
        return int(match.group(1))
    return None


def _extract_court_from_citation(citation_string: str) -> Optional[str]:
    """
    Extract and normalise a court identifier from a citation string.

    Two strategies, tried in order:

    1. Direct CourtListener ID match — catches benchmark Type B corruptions where
       a court_id like 'ca11', 'cadc', 'nh', 'lactapp' was injected into the
       trailing parenthetical of a reporter citation, e.g. "476 U.S. 207 (ca11)".
       Pattern: last parenthetical group that contains only word chars (no year,
       no spaces, no circuit-style text).

    2. Alias match — catches natural-language court strings in formatted citations
       like "923 F.3d 1027 (4th Cir. 2019)" → "ca4".
    """
    # Strategy 1 — bare court_id in trailing parenthetical
    # Matches e.g. "(ca11)", "(nh)", "(cadc)", "(lactapp)"
    # Excludes parentheticals that contain a 4-digit year or spaces (those are
    # natural-language court+year groups handled by strategy 2).
    bare_match = re.search(r"\(([a-z][a-z0-9]{1,10})\)\s*$", citation_string.strip().lower())
    if bare_match:
        candidate = bare_match.group(1)
        # Reject if it looks like a year (shouldn't happen given regex, but be safe)
        if not re.fullmatch(r"(19|20)\d{2}", candidate):
            logger.debug("Extracted bare court_id from citation: %s", candidate)
            return candidate

    # Strategy 2 — human-readable court alias
    lower = citation_string.lower()
    for alias, court_id in _ALIAS_TO_COURT.items():
        if alias in lower:
            return court_id

    return None


def _fetch_node_properties(case_id: int, driver) -> Optional[dict]:
    """
    Fetch year and court_id from the Neo4j Case node.
    Returns None if node not found (shouldn't happen — Layer 1 already confirmed existence).
    """
    query = """
        MATCH (c:Case {id: $id})
        RETURN c.year AS year, c.court_id AS court_id
        LIMIT 1
    """
    with driver.session() as session:
        result = session.run(query, id=case_id)
        record = result.single()
        if record is None:
            return None
        return {
            "year":     int(record["year"]) if record["year"] is not None else None,
            "court_id": record["court_id"],
        }


def check_metadata(
    case_id: int,
    citation_string: str,
    driver,
    year_tolerance: int = 0,
) -> MetadataResult:
    """
    Compare year and court extracted from the citation string against the
    Neo4j node properties for the given case_id.

    Args:
        case_id:          CourtListener opinion ID (Layer 1 confirmed it exists)
        citation_string:  Raw citation as it appears in the AI-generated text
        driver:           Active Neo4j driver
        year_tolerance:   Allow ±N years before flagging. Default 0 (exact match).
                          Set to 1 if you want to allow off-by-one year errors.

    Returns:
        MetadataResult — is_valid=False if any mismatch detected
    """
    cited_year  = _extract_year_from_citation(citation_string)
    cited_court = _extract_court_from_citation(citation_string)

    # Nothing to check — pure reporter citation like "392 U.S. 1"
    if cited_year is None and cited_court is None:
        logger.debug(
            "Metadata check skipped — no extractable year/court in: %s",
            citation_string,
        )
        return MetadataResult(
            checked=False,
            year_match=None,
            court_match=None,
            cited_year=None,
            cited_court=None,
            actual_year=None,
            actual_court=None,
            is_valid=True,   # can't flag what we can't check
        )

    props = _fetch_node_properties(case_id, driver)
    if props is None:
        logger.warning("Metadata check: node not found for case_id=%d", case_id)
        return MetadataResult(
            checked=False,
            year_match=None,
            court_match=None,
            cited_year=cited_year,
            cited_court=cited_court,
            actual_year=None,
            actual_court=None,
            is_valid=True,   # can't penalise if node fetch failed
        )

    actual_year  = props["year"]
    actual_court = props["court_id"]

    # Year match
    year_match: Optional[bool] = None
    if cited_year is not None and actual_year is not None:
        year_match = abs(cited_year - actual_year) <= year_tolerance

    # Court match
    court_match: Optional[bool] = None
    if cited_court is not None and actual_court is not None:
        court_match = cited_court == actual_court

    # is_valid: True only if every available check passes
    checks = [c for c in [year_match, court_match] if c is not None]
    is_valid = all(checks) if checks else True

    if not is_valid:
        logger.info(
            "Metadata MISMATCH on case_id=%d: "
            "cited_year=%s actual_year=%s | cited_court=%s actual_court=%s",
            case_id, cited_year, actual_year, cited_court, actual_court,
        )
    else:
        logger.debug(
            "Metadata OK for case_id=%d (year=%s court=%s)",
            case_id, cited_year, cited_court,
        )

    return MetadataResult(
        checked=True,
        year_match=year_match,
        court_match=court_match,
        cited_year=cited_year,
        cited_court=cited_court,
        actual_year=actual_year,
        actual_court=actual_court,
        is_valid=is_valid,
    )