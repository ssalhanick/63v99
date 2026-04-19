"""
detector/llm_check.py

Layer 2b — LLM proposition accuracy check.

Given a resolved case_id and the proposition text from the citation context,
fetches the case opinion from the parquet and asks Claude Haiku whether the
proposition accurately reflects what the case actually held.

Verdict contribution:
    is_accurate = True  → proposition matches case holding (REAL signal)
    is_accurate = False → proposition does not match (SUSPICIOUS signal)
    skipped = True      → LLM call failed, fall back to semantic_check result

Called by detector/pipeline.py after Layer 2a (semantic_check) passes.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from config import PROCESSED_DIR, ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPINION_WORD_LIMIT  = 2000          # first N words of opinion sent to LLM
HAIKU_MODEL         = "claude-haiku-4-5-20251001"
MAX_TOKENS          = 256
API_URL             = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION   = "2023-06-01"

PARQUET_PATH = Path(PROCESSED_DIR) / "cases_enriched.parquet"

SYSTEM_PROMPT = """You are a legal citation verifier. You will be given a legal 
proposition and an excerpt from a court opinion. Your job is to determine whether 
the proposition accurately reflects what the case actually held.

Be strict — if the proposition misstates the holding, reverses the ruling, 
attributes a different legal standard, or describes facts not present in the 
opinion, mark it as inaccurate.

Respond with a JSON object only, no preamble or markdown:
{"accurate": true, "reason": "one sentence explanation"}
or
{"accurate": false, "reason": "one sentence explanation"}"""

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    is_accurate:  bool    # True = proposition matches case holding
    reason:       str     # Haiku's one-sentence explanation
    tokens_used:  int     # input + output tokens for cost tracking
    case_id:      int
    skipped:      bool = False  # True if LLM call failed — pipeline uses fallback


# ---------------------------------------------------------------------------
# Module-level parquet cache — loaded once
# ---------------------------------------------------------------------------

_opinion_df = None

def _load_opinions():
    global _opinion_df
    if _opinion_df is not None:
        return
    logger.info("Loading opinion text from parquet...")
    _opinion_df = pd.read_parquet(
        PARQUET_PATH,
        columns=["case_id", "plain_text"]
    ).set_index("case_id")
    logger.info("  Opinion parquet loaded: %d cases", len(_opinion_df))


def _get_opinion_excerpt(case_id: int) -> str | None:
    """
    Fetch and trim the opinion text for a case to OPINION_WORD_LIMIT words.
    Returns None if case not found or plain_text is empty.
    """
    _load_opinions()
    if case_id not in _opinion_df.index:
        logger.warning("case_id %d not found in opinion parquet", case_id)
        return None

    text = _opinion_df.loc[case_id, "plain_text"]
    if not isinstance(text, str) or not text.strip():
        logger.warning("case_id %d has empty plain_text", case_id)
        return None

    words = text.split()
    trimmed = " ".join(words[:OPINION_WORD_LIMIT])
    return trimmed


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_haiku(proposition: str, opinion_excerpt: str) -> dict:
    """
    Call Claude Haiku with the proposition + opinion excerpt.
    Returns parsed JSON dict with 'accurate' and 'reason' keys.
    Raises on network error or bad response.
    """
    user_message = (
        f"PROPOSITION:\n{proposition}\n\n"
        f"CASE OPINION (excerpt):\n{opinion_excerpt}"
    )

    payload = {
        "model":      HAIKU_MODEL,
        "max_tokens": MAX_TOKENS,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": user_message}],
    }

    headers = {
        "Content-Type":      "application/json",
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
    }

    resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()

    data        = resp.json()
    raw_text    = data["content"][0]["text"].strip()
    tokens_used = data["usage"]["input_tokens"] + data["usage"]["output_tokens"]

    # Strip markdown fences if present
    clean = re.sub(r"```json|```", "", raw_text).strip()
    parsed = json.loads(clean)

    return {"accurate": parsed["accurate"], "reason": parsed["reason"],
            "tokens_used": tokens_used}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def llm_check(case_id: int, proposition: str) -> LLMResult:
    """
    Run the LLM proposition accuracy check for a resolved citation.

    Args:
        case_id:     resolved case ID from Layer 1
        proposition: the full context text surrounding the citation

    Returns:
        LLMResult with is_accurate, reason, tokens_used, and skipped flag.
        If skipped=True, the pipeline falls back to semantic_check result.
    """
    # Fetch opinion excerpt
    opinion_excerpt = _get_opinion_excerpt(case_id)
    if opinion_excerpt is None:
        logger.warning("LLM check skipped — no opinion text for case_id %d", case_id)
        return LLMResult(
            is_accurate=True, reason="skipped — no opinion text",
            tokens_used=0, case_id=case_id, skipped=True
        )

    # Call Haiku
    try:
        result = _call_haiku(proposition, opinion_excerpt)
        logger.info(
            "LLM check — case_id %d: accurate=%s tokens=%d | %s",
            case_id, result["accurate"], result["tokens_used"], result["reason"]
        )
        return LLMResult(
            is_accurate  = result["accurate"],
            reason       = result["reason"],
            tokens_used  = result["tokens_used"],
            case_id      = case_id,
            skipped      = False,
        )

    except requests.exceptions.Timeout:
        logger.warning("LLM check timed out for case_id %d — skipping", case_id)
    except json.JSONDecodeError as e:
        logger.warning("LLM response parse error for case_id %d: %s — skipping", case_id, e)
    except Exception as e:
        logger.warning("LLM check failed for case_id %d: %s — skipping", case_id, e)

    return LLMResult(
        is_accurate=True, reason="skipped — LLM error",
        tokens_used=0, case_id=case_id, skipped=True
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Hoskins v. Withers — correct proposition
    correct = (
        "In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held "
        "that a trooper had reasonable suspicion to conduct a traffic stop where a "
        "vehicle's license plate lettering was partially obstructed, and that Utah's "
        "license plate maintenance law applied to out-of-state plates."
    )

    # Hoskins v. Withers — wrong proposition (Type B)
    wrong = (
        "In Hoskins v. Withers, 92 F.4th 1279 (10th Cir. 2024), the court held "
        "that officers may conduct a warrantless search of a vehicle without "
        "reasonable suspicion whenever a narcotics dog is present at the scene, "
        "regardless of whether the initial stop was lawful."
    )

    print("\n--- Correct proposition ---")
    r1 = llm_check(9942139, correct)
    print(f"  accurate={r1.is_accurate} | {r1.reason} | tokens={r1.tokens_used}")

    print("\n--- Wrong proposition (Type B) ---")
    r2 = llm_check(9942139, wrong)
    print(f"  accurate={r2.is_accurate} | {r2.reason} | tokens={r2.tokens_used}")