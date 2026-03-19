"""
api/main.py

FastAPI endpoint for the Verit hallucination detection pipeline.

Endpoints:
    POST /check-citation   — run pipeline on raw legal text, return verdicts
    GET  /health           — liveness check

Usage:
    uvicorn api.main:app --reload --port 8000

Example request:
    curl -X POST http://localhost:8000/check-citation \
         -H "Content-Type: application/json" \
         -d '{"text": "In Terry v. Ohio, 392 U.S. 1 (1968), the Court held..."}'

Security notes:
    - Input capped at MAX_TEXT_LENGTH (50,000 chars) to prevent runaway inference
    - CORS restricted to Streamlit frontend origin (localhost:8501)
    - No authentication — single-user local deployment
    - Rate limiting not yet implemented — see PROJECT_CONTEXT.md for production roadmap
      (recommended: slowapi + per-API-key limits when monetizing)
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from detector.pipeline import run_pipeline, CitationVerdict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Maximum input size — prevents runaway legal-bert inference on oversized payloads
MAX_TEXT_LENGTH = 50_000  # characters

app = FastAPI(
    title="Verit — Legal Citation Hallucination Detector",
    description="Detects hallucinated citations in AI-generated legal text using a three-layer pipeline.",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# CORS — restrict to Streamlit frontend origin
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],   # Streamlit default port
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CheckCitationRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": (
                    "In Terry v. Ohio, 392 U.S. 1 (1968), the Court held that "
                    "a brief investigatory stop requires reasonable articulable suspicion."
                )
            }
        }


class TopMatch(BaseModel):
    case_id:     Optional[int]
    case_name:   Optional[str]
    court_id:    Optional[str]
    date_filed:  Optional[str]
    cite_count:  Optional[int]
    rrf_score:   float
    dense_score: float
    bm25_score:  float


class CitationResult(BaseModel):
    citation_string: str
    case_name:       Optional[str]
    case_id:         Optional[int]
    verdict:         str                  # REAL | SUSPICIOUS | HALLUCINATED
    exists:          bool                 # Layer 1
    semantic_score:  Optional[float]      # Layer 2 — top RRF score
    dense_score:     Optional[float]      # Layer 2 — top cosine similarity
    density_score:   Optional[int]        # Layer 3 — shared citation count
    top_matches:     list[TopMatch]       # Layer 2 corpus candidates (for RAG)


class CheckCitationResponse(BaseModel):
    citation_count: int
    citations:      list[CitationResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdict_to_response(v: CitationVerdict) -> CitationResult:
    """Convert a CitationVerdict dataclass to a Pydantic response model."""
    top_matches = [
        TopMatch(
            case_id     = m.get("case_id"),
            case_name   = m.get("case_name"),
            court_id    = m.get("court_id"),
            date_filed  = m.get("date_filed"),
            cite_count  = m.get("cite_count"),
            rrf_score   = m.get("rrf_score", 0.0),
            dense_score = m.get("dense_score", 0.0),
            bm25_score  = m.get("bm25_score", 0.0),
        )
        for m in v.top_matches
    ]

    return CitationResult(
        citation_string = v.citation_string,
        case_name       = v.case_name,
        case_id         = v.case_id,
        verdict         = v.verdict,
        exists          = v.exists,
        semantic_score  = v.semantic.rrf_score        if v.semantic      else None,
        dense_score     = v.semantic.top_dense_score  if v.semantic      else None,
        density_score   = v.connectivity.density_score if v.connectivity else None,
        top_matches     = top_matches,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "service": "verit"}


@app.post("/check-citation", response_model=CheckCitationResponse)
def check_citation(request: CheckCitationRequest):
    """
    Run the three-layer hallucination detection pipeline on raw legal text.

    Returns one verdict per unique full citation found in the text.
    Citations with no resolvable full form (id., supra, etc.) are skipped.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty")

    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds {MAX_TEXT_LENGTH:,} character limit ({len(request.text):,} chars received)",
        )

    logger.info("Received /check-citation request (%d chars)", len(request.text))

    try:
        verdicts = run_pipeline(request.text)
    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    results = [_verdict_to_response(v) for v in verdicts]

    return CheckCitationResponse(
        citation_count = len(results),
        citations      = results,
    )


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)