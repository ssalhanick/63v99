"""
detector/eyecite_parser.py

Extract citation strings from raw AI-generated text using EyeCite, then resolve
each citation to a CourtListener cluster ID via the citation-lookup API.

Output per citation:
    - citation_string  : formatted string, e.g. "392 U.S. 1"
    - case_name        : name as returned by CourtListener (or None)
    - case_id          : integer cluster ID — matches Neo4j node id (or None if unresolvable)
    - context_text     : sentence window around the citation in the source text
    - reporter         : e.g. "U.S.", "F.3d"
    - volume           : int
    - page             : int

Note on case_id:
    CourtListener's citation-lookup returns cluster IDs. Neo4j nodes are keyed on
    cluster ID (not opinion ID), so we use cluster_id directly as case_id. This
    matches how graph_loader.py loaded cases into Neo4j.
"""

import re
import logging
import requests
import time
from dataclasses import dataclass
from typing import Optional

from eyecite import get_citations
from eyecite.models import FullCaseCitation

from config import COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ResolvedCitation:
    citation_string: str
    reporter: str
    volume: int
    page: int
    case_name: Optional[str]
    case_id: Optional[int]      # CourtListener cluster ID = Neo4j node id (None if unresolved)
    context_text: str           # surrounding sentence window


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

CONTEXT_WINDOW = 3  # sentences on each side of the citation


def _extract_context(full_text: str, citation_string: str, window: int = CONTEXT_WINDOW) -> str:
    """
    Return a window of sentences surrounding the first occurrence of
    citation_string in full_text. Falls back to a character window if
    sentence splitting doesn't find enough material.
    """
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    target_idx = None
    for i, sent in enumerate(sentences):
        if citation_string in sent:
            target_idx = i
            break

    if target_idx is None:
        # Fallback: character window around first occurrence
        idx = full_text.find(citation_string)
        if idx == -1:
            return citation_string
        start = max(0, idx - 300)
        end = min(len(full_text), idx + len(citation_string) + 300)
        return full_text[start:end].strip()

    start = max(0, target_idx - window)
    end = min(len(sentences), target_idx + window + 1)
    return " ".join(sentences[start:end]).strip()


# ---------------------------------------------------------------------------
# CourtListener resolution
# ---------------------------------------------------------------------------

HEADERS = {
    "Authorization": f"Token {COURTLISTENER_TOKEN}",
    "Content-Type": "application/json",
}

LOOKUP_URL = f"{COURTLISTENER_BASE_URL}/citation-lookup/"

_API_DELAY = 0.5  # seconds — polite delay between API calls


def _resolve_citation(volume: int, reporter: str, page: int) -> tuple[Optional[int], Optional[str]]:
    """
    POST to CourtListener /citation-lookup/ for exact citation match.

    Endpoint: POST /api/rest/v4/citation-lookup/
    Body: {"text": "392 U.S. 1"}

    Response is a list; each item has a clusters[] array. We take the first
    cluster's id (cluster ID) as case_id — this matches the id field on Neo4j
    Case nodes loaded by graph_loader.py.

    Returns: (cluster_id, case_name) — both None if not found.
    """
    citation_string = f"{volume} {reporter} {page}"

    try:
        resp = requests.post(
            LOOKUP_URL,
            headers=HEADERS,
            json={"text": citation_string},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            logger.debug("No citation-lookup result for %s", citation_string)
            return None, None

        clusters = data[0].get("clusters", [])
        if not clusters:
            logger.debug("No clusters returned for %s", citation_string)
            return None, None

        cluster = clusters[0]
        cluster_id = cluster.get("id")
        case_name = cluster.get("case_name")
        return cluster_id, case_name

    except requests.RequestException as e:
        logger.warning("CourtListener API error for %s: %s", citation_string, e)
        return None, None
    finally:
        time.sleep(_API_DELAY)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_citations(text: str) -> list[ResolvedCitation]:
    """
    Extract all full case citations from text, resolve each to a CourtListener
    cluster ID, and return a list of ResolvedCitation objects.

    Only FullCaseCitation objects are processed — short form citations like
    "id." or "supra" are skipped because they can't be independently resolved.

    Args:
        text: raw AI-generated legal text

    Returns:
        List of ResolvedCitation objects (unresolved citations have case_id=None)
    """
    found = get_citations(text)
    resolved = []
    seen = set()  # deduplicate by (volume, reporter, page)

    for citation in found:
        if not isinstance(citation, FullCaseCitation):
            logger.debug("Skipping non-full citation: %s", type(citation).__name__)
            continue

        try:
            volume = int(citation.groups["volume"])
            reporter = citation.groups["reporter"]
            page = int(citation.groups["page"])
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Could not parse citation fields: %s", e)
            continue

        key = (volume, reporter, page)
        if key in seen:
            continue
        seen.add(key)

        citation_string = f"{volume} {reporter} {page}"
        context = _extract_context(text, citation_string)

        logger.info("Resolving citation: %s", citation_string)
        case_id, case_name = _resolve_citation(volume, reporter, page)

        resolved.append(ResolvedCitation(
            citation_string=citation_string,
            reporter=reporter,
            volume=volume,
            page=page,
            case_name=case_name,
            case_id=case_id,
            context_text=context,
        ))

    logger.info("Parsed %d unique full citations from input text", len(resolved))
    return resolved


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

    results = parse_citations(sample)
    for r in results:
        print(json.dumps({
            "citation_string": r.citation_string,
            "case_name": r.case_name,
            "case_id": r.case_id,
            "context_text": r.context_text[:100] + "...",
        }, indent=2))