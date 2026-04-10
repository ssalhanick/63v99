"""
frontend/llm.py
Week 9 — Claude Haiku integration for Verit.

Provides two streaming generators:
    stream_explanation() — plain-English verdict explanation grounded in top_matches
    stream_correction()  — suggested real case (HALLUCINATED only)

Called directly from Streamlit via st.write_stream().
RAG context: top_matches from Layer 2 are passed as system prompt context so
Haiku's explanations are anchored to the actual corpus, not training data alone.
"""

import logging
from typing import Generator

import anthropic

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

log = logging.getLogger(__name__)

# ── Client (module-level singleton) ───────────────────────────────────────────
_client: anthropic.Anthropic | None = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ── RAG context builder ───────────────────────────────────────────────────────

def _format_corpus_context(top_matches: list[dict]) -> str:
    """
    Format top_matches into a numbered list for the system prompt.
    Only includes matches with a case_name — skips stub nodes.
    """
    lines = []
    for i, m in enumerate(top_matches, 1):
        name  = m.get("case_name") or "Unknown"
        court = m.get("court_id")  or "unknown court"
        year  = (m.get("date_filed") or "")[:4] or "unknown year"
        score = m.get("rrf_score", 0.0)
        lines.append(f"  {i}. {name} ({court}, {year}) — similarity score: {score:.4f}")
    return "\n".join(lines) if lines else "  No corpus matches retrieved."


# ── System prompts ────────────────────────────────────────────────────────────

def _explanation_system_prompt(top_matches: list[dict]) -> str:
    corpus_context = _format_corpus_context(top_matches)
    return f"""You are a legal citation verification assistant for Verit, a system that \
detects hallucinated citations in AI-generated Fourth Amendment legal text.

The following are real Fourth Amendment cases retrieved from the Verit corpus \
that are most semantically similar to the citation being evaluated:

{corpus_context}

Your job is to explain the verdict in 2-3 plain-English sentences that a lawyer \
could understand. Be specific — reference the corpus cases above when relevant. \
Do not use bullet points. Do not repeat the citation string or verdict label. \
Do not hedge excessively. Be direct and informative."""


def _correction_system_prompt(top_matches: list[dict]) -> str:
    corpus_context = _format_corpus_context(top_matches)
    return f"""You are a legal citation correction assistant for Verit, a system that \
detects hallucinated citations in AI-generated Fourth Amendment legal text.

The following are real Fourth Amendment cases from the Verit corpus that are \
most semantically similar to the hallucinated citation's context:

{corpus_context}

Suggest the single most appropriate real case from the list above as a replacement \
for the hallucinated citation. Format your response as:

Suggested replacement: [Case Name], [reporter citation if known] — [one sentence \
explaining why this case fits the context].

If none of the corpus cases are a good fit, say so briefly. Do not fabricate citations."""


# ── Streaming generators ──────────────────────────────────────────────────────

def stream_explanation(
    citation_string: str,
    verdict: str,
    semantic_score: float | None,
    density_score:  int   | None,
    top_matches:    list[dict],
) -> Generator[str, None, None]:
    """
    Stream a plain-English explanation of a verdict from Claude Haiku.

    Yields text chunks as they arrive — pass directly to st.write_stream().

    Args:
        citation_string: The citation as it appeared in the AI-generated text
        verdict:         REAL | SUSPICIOUS | HALLUCINATED
        semantic_score:  Layer 2 RRF score (None if Layer 1 short-circuited)
        density_score:   Layer 3 citation density (None if Layer 1 short-circuited)
        top_matches:     Layer 2 corpus candidates with case metadata
    """
    sem_str     = f"{semantic_score:.4f}" if semantic_score is not None else "N/A"
    density_str = str(density_score)      if density_score  is not None else "N/A"

    user_message = (
        f"Citation: {citation_string}\n"
        f"Verdict: {verdict}\n"
        f"Semantic relevance score: {sem_str}\n"
        f"Citation network density: {density_str}\n\n"
        f"Explain why this citation received a {verdict} verdict."
    )

    client = _get_client()

    try:
        with client.messages.stream(
            model      = ANTHROPIC_MODEL,
            max_tokens = 300,
            system     = _explanation_system_prompt(top_matches),
            messages   = [{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    except anthropic.APIError as e:
        log.error("Haiku API error during explanation: %s", e)
        yield f"[Explanation unavailable — API error: {e}]"


def stream_correction(
    citation_string: str,
    context_text:   str,
    top_matches:    list[dict],
) -> Generator[str, None, None]:
    """
    Stream a suggested real-case correction for a HALLUCINATED citation.

    Yields text chunks as they arrive — pass directly to st.write_stream().

    Args:
        citation_string: The hallucinated citation string
        context_text:    The surrounding paragraph from the AI-generated text
        top_matches:     Layer 2 corpus candidates to draw correction from
    """
    user_message = (
        f"The following citation was flagged as HALLUCINATED:\n"
        f"  {citation_string}\n\n"
        f"Context in which it appeared:\n"
        f"  {context_text[:400]}{'...' if len(context_text) > 400 else ''}\n\n"
        f"Suggest the best real replacement from the corpus cases provided."
    )

    client = _get_client()

    try:
        with client.messages.stream(
            model      = ANTHROPIC_MODEL,
            max_tokens = 200,
            system     = _correction_system_prompt(top_matches),
            messages   = [{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    except anthropic.APIError as e:
        log.error("Haiku API error during correction: %s", e)
        yield f"[Correction unavailable — API error: {e}]"