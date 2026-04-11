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
import ast
import requests
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eyecite import get_citations
from eyecite.models import FullCaseCitation

from config import COURTLISTENER_TOKEN, COURTLISTENER_BASE_URL, PROCESSED_DIR

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
# Corpus citation index  (fallback when CourtListener API returns None)
# ---------------------------------------------------------------------------

def _build_corpus_index() -> dict[str, tuple[int, str]]:
    """
    Build a lookup dict from the local parquet corpus:
        citation_string  ->  (case_id, case_name)

    Used as a fallback when the CourtListener live API does not return a
    cluster_id for a citation that is nonetheless present in our corpus.
    Each case's `citations` column is a list of reporter strings, e.g.
    ["901 F.3d 1042", "2018 WL 123456"].  Any of those strings can be used
    to find the case_id.

    Returns an empty dict if the parquet file is missing.
    """
    parquet_path = Path(PROCESSED_DIR) / "cases_enriched.parquet"
    if not parquet_path.exists():
        logger.warning("Corpus parquet not found at %s — fallback disabled", parquet_path)
        return {}

    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=["case_id", "case_name", "citations"])
    except Exception as e:
        logger.warning("Could not load corpus parquet for citation index: %s", e)
        return {}

    index: dict[str, tuple[int, str]] = {}
    for _, row in df.iterrows():
        cid  = int(row["case_id"])
        name = str(row["case_name"]) if row["case_name"] else ""
        raw  = row["citations"]

        # citations may be a list already or a JSON/repr string
        if isinstance(raw, list):
            cit_list = raw
        elif isinstance(raw, str):
            try:
                cit_list = ast.literal_eval(raw)
            except Exception:
                cit_list = [raw]
        else:
            cit_list = []

        for cit_str in cit_list:
            if isinstance(cit_str, str) and cit_str.strip():
                index[cit_str.strip()] = (cid, name)

    logger.info("Corpus citation index built: %d citation strings -> case_ids", len(index))
    return index


# Built once at module import — shared by all calls to _resolve_citation
_CORPUS_INDEX: dict[str, tuple[int, str]] = _build_corpus_index()


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
    Resolve a citation to a (case_id, case_name) pair compatible with Neo4j.

    Two-stage lookup — corpus index is checked FIRST:

      1. Corpus parquet index  — exact match against the reporter strings scraped
                                 into Verit. case_ids here are guaranteed to match
                                 Neo4j (both loaded from the same CourtListener
                                 scrape). This is the primary source of truth.

      2. CourtListener live API — fallback for citations not in our corpus.
                                  CourtListener may have re-indexed or re-merged
                                  clusters since our scrape, so its cluster IDs
                                  can differ from the Neo4j node IDs. Only used
                                  when the corpus index has no match.

    Returns: (case_id, case_name) — both None if neither stage resolves it.
    """
    citation_string = f"{volume} {reporter} {page}"

    # ---- Stage 1: Corpus index (primary — guaranteed Neo4j-compatible IDs) --
    if citation_string in _CORPUS_INDEX:
        cid, name = _CORPUS_INDEX[citation_string]
        logger.info(
            "Resolved '%s' via corpus index -> case_id=%d ('%s')",
            citation_string, cid, name,
        )
        return cid, name

    logger.debug("'%s' not in corpus index — trying CourtListener API", citation_string)

    # ---- Stage 2: CourtListener API (fallback for out-of-corpus citations) --
    try:
        resp = requests.post(
            LOOKUP_URL,
            headers=HEADERS,
            json={"text": citation_string},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        clusters = data[0].get("clusters", []) if data else []
        if clusters:
            cluster    = clusters[0]
            cluster_id = cluster.get("id")
            case_name  = cluster.get("case_name")
            if cluster_id is not None:
                logger.info(
                    "Resolved '%s' via CourtListener API -> case_id=%d",
                    citation_string, cluster_id,
                )
                return cluster_id, case_name

        logger.debug("CourtListener API returned no cluster for '%s'", citation_string)

    except requests.RequestException as e:
        logger.warning("CourtListener API error for '%s': %s", citation_string, e)
    finally:
        time.sleep(_API_DELAY)

    logger.info("Could not resolve '%s' via corpus index or CourtListener API", citation_string)
    return None, None



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