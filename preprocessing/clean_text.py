"""
preprocessing/clean_text.py

Cleans raw legal opinion text from CourtListener before embedding.

Steps:
  1. Strip court headers and footers (docket lines, attorney listings, procedural boilerplate)
  2. Normalize citation strings → [CITATION] token
  3. Fix encoding artifacts, collapse excessive whitespace

Input:  data/processed/cases_enriched.parquet
Output: data/processed/cases_cleaned.parquet

Run:
  python -m preprocessing.clean_text
"""

import re
import logging
import unicodedata
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH  = Path(PROCESSED_DIR) / "cases_enriched.parquet"
OUTPUT_PATH = Path(PROCESSED_DIR) / "cases_cleaned.parquet"

# ---------------------------------------------------------------------------
# Citation normalization patterns
# Covers: U.S., F.3d, F.2d, F.Supp, S.Ct., L.Ed., F.4th, etc.
# ---------------------------------------------------------------------------
_REPORTER_PATTERN = re.compile(
    r"""
    \d{1,3}                         # volume
    \s+
    (?:
        U\.S\.|S\.Ct\.|L\.Ed\.(?:2d)?|F\.(?:2d|3d|4th|Supp(?:\.2d|\.3d)?)|
        A\.(?:2d|3d)|B\.R\.|F\.R\.D\.|Fed\.\s*Cl\.|Fed\.\s*Appx?\.|
        WL                          # Westlaw citations
    )
    \s+
    \d+                             # page
    (?:\s*\(\w[\w\s.,]*\d{4}\))?   # optional (court year) suffix
    """,
    re.VERBOSE,
)

# Case name pattern: "Word v. Word" or "In re Word"
_CASE_NAME_PATTERN = re.compile(
    r"""
    (?:
        (?:[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)*)\s+v\.\s+
        (?:[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)*)
        |
        In\s+re\s+[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)*
    )
    (?:,\s*\d{1,3}\s+\S+\.?\s+\d+(?:\s*\(\w[\w\s.,]*\d{4}\))?)?
    """,
    re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Header / footer patterns
# ---------------------------------------------------------------------------

# Lines that look like docket numbers
_DOCKET_LINE = re.compile(r"^(?:No\.|Case No\.|Docket No\.)\s+[\w\-]+.*$", re.MULTILINE)

# "Filed:" / "Argued:" / "Decided:" date lines
_PROCEDURAL_DATE = re.compile(
    r"^(?:Filed|Argued|Decided|Submitted|Rehearing Denied)[:\s]+\w.{0,40}$",
    re.MULTILINE | re.IGNORECASE,
)

# Attorney / counsel block patterns
_COUNSEL_BLOCK = re.compile(
    r"(?:Counsel|Attorney|Argued by|Submitted by|For (?:Appellant|Appellee|Petitioner|Respondent)).*?\n",
    re.IGNORECASE,
)

# Page number markers injected by PDF extraction (e.g. "*1234" or "Page 5")
_PAGE_MARKER = re.compile(r"^\*\d+\s*$|^Page\s+\d+\s*$", re.MULTILINE)

# Court header lines (e.g. "UNITED STATES COURT OF APPEALS FOR THE NINTH CIRCUIT")
_COURT_HEADER = re.compile(
    r"^UNITED STATES (?:COURT OF APPEALS|DISTRICT COURT).*$",
    re.MULTILINE | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------

def _fix_encoding(text: str) -> str:
    """Normalize unicode, replace common OCR artifacts."""
    # Normalize to NFC (canonical composition)
    text = unicodedata.normalize("NFC", text)
    # Common OCR substitutions
    replacements = {
        "\x92": "'",   # curly apostrophe → straight
        "\x93": '"',
        "\x94": '"',
        "\x96": "-",   # en-dash
        "\x97": "-",   # em-dash
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ", # non-breaking space
        "\ufffd": "",  # replacement character
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _strip_headers_footers(text: str) -> str:
    """Remove procedural boilerplate that adds noise without doctrinal content."""
    text = _DOCKET_LINE.sub("", text)
    text = _PROCEDURAL_DATE.sub("", text)
    text = _COUNSEL_BLOCK.sub("", text)
    text = _PAGE_MARKER.sub("", text)
    text = _COURT_HEADER.sub("", text)
    return text


def _normalize_citations(text: str) -> str:
    """
    Replace citation strings with [CITATION] token.
    Order matters: replace full "Name v. Name, reporter" first,
    then bare reporter strings, to avoid partial matches.
    """
    text = _CASE_NAME_PATTERN.sub("[CITATION]", text)
    text = _REPORTER_PATTERN.sub("[CITATION]", text)
    # Collapse consecutive [CITATION] tokens (e.g. after name + reporter both matched)
    text = re.sub(r"(\[CITATION\]\s*){2,}", "[CITATION] ", text)
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines and trailing spaces."""
    # Replace 3+ consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single opinion text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _fix_encoding(text)
    text = _strip_headers_footers(text)
    text = _normalize_citations(text)
    text = _normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info(f"Loading {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    log.info(f"Loaded {len(df):,} cases")

    original_nulls = df["plain_text"].isna().sum()
    log.info(f"Cases with null plain_text before cleaning: {original_nulls:,}")

    log.info("Cleaning text …")
    df["plain_text"] = df["plain_text"].fillna("").apply(clean_text)

    # Replace empty strings (after cleaning) with NaN for downstream filtering
    df["plain_text"] = df["plain_text"].replace("", None)

    post_nulls = df["plain_text"].isna().sum()
    log.info(f"Cases with empty/null plain_text after cleaning: {post_nulls:,}")
    log.info(f"Cases retained with text: {len(df) - post_nulls:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()