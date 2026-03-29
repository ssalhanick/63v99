"""
frontend/app.py

Streamlit frontend for the Verit hallucination detection pipeline.

Week 7 scope:
    - Text input for AI-generated legal text
    - POST to FastAPI /check-citation endpoint
    - Verdict table with badge, semantic score, density score
    - Expandable detail per citation showing top corpus matches

Week 9 additions (not yet implemented):
    - Claude Haiku LLM explanations per verdict
    - Suggested corrections for hallucinated citations

Usage:
    streamlit run frontend/app.py
    (FastAPI must be running on localhost:8000)
"""

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Verit — Legal Citation Checker",
    page_icon="⚖️",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL         = "http://localhost:8000/check-citation"
REQUEST_TIMEOUT = 60   # seconds — legal-bert inference can be slow

# Verdict badge config
VERDICT_BADGE = {
    "REAL":         ("🟢", "green",  "Real"),
    "SUSPICIOUS":   ("🟡", "orange", "Suspicious"),
    "HALLUCINATED": ("🔴", "red",    "Hallucinated"),
}

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-real        { color: #2e7d32; font-weight: 700; }
    .verdict-suspicious  { color: #e65100; font-weight: 700; }
    .verdict-hallucinated{ color: #c62828; font-weight: 700; }
    .citation-string     { font-family: monospace; font-size: 0.95rem; }
    .match-row           { padding: 4px 0; border-bottom: 1px solid #eee; }
    .score-label         { color: #666; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚖️ Verit")
st.caption("Legal Citation Hallucination Detector · Fourth Amendment Corpus")
st.divider()


# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("Paste AI-generated legal text")
input_text = st.text_area(
    label="Input text",
    placeholder=(
        "Paste a paragraph of AI-generated legal text containing citations.\n\n"
        "Example: In Terry v. Ohio, 392 U.S. 1 (1968), the Court held that a police "
        "officer may stop and briefly detain a person based on reasonable suspicion..."
    ),
    height=200,
    label_visibility="collapsed",
)

check_button = st.button("🔍 Check Citations", type="primary", use_container_width=True)
st.divider()


# ── Helper functions ──────────────────────────────────────────────────────────

def call_api(text: str) -> dict:
    """POST to FastAPI /check-citation and return parsed JSON response."""
    response = requests.post(
        API_URL,
        json={"text": text},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def render_verdict_badge(verdict: str) -> str:
    """Return an HTML badge string for the given verdict."""
    emoji, color, label = VERDICT_BADGE.get(
        verdict, ("⚪", "gray", verdict)
    )
    return f'<span style="color:{color}; font-weight:700;">{emoji} {label}</span>'


def render_score_bar(score: float | None, label: str, threshold: float | None = None) -> None:
    """Render a labeled progress bar for a numeric score."""
    if score is None:
        st.markdown(f'<span class="score-label">{label}: —</span>', unsafe_allow_html=True)
        return
    clamped = max(0.0, min(1.0, float(score)))
    color   = "normal"
    if threshold is not None:
        color = "normal" if clamped >= threshold else "inverse"
    st.markdown(f'<span class="score-label">{label}: {score:.3f}</span>', unsafe_allow_html=True)
    st.progress(clamped)


def render_citation_result(idx: int, citation: dict) -> None:
    """Render a single citation result card with expandable detail."""
    verdict         = citation.get("verdict", "UNKNOWN")
    citation_string = citation.get("citation_string", "Unknown citation")
    semantic_score  = citation.get("semantic_score")
    density_score   = citation.get("density_score")
    existence       = citation.get("existence", False)
    top_matches     = citation.get("top_matches", [])

    emoji, color, label = VERDICT_BADGE.get(verdict, ("⚪", "gray", verdict))

    # ── Citation card ─────────────────────────────────────────────────────────
    with st.container():
        col_verdict, col_citation, col_semantic, col_density = st.columns([1.5, 4, 1.5, 1.5])

        with col_verdict:
            st.markdown(
                f'<span style="color:{color}; font-size:1.1rem; font-weight:700;">'
                f'{emoji} {label}</span>',
                unsafe_allow_html=True
            )

        with col_citation:
            st.markdown(
                f'<span class="citation-string">{citation_string}</span>',
                unsafe_allow_html=True
            )

        with col_semantic:
            if semantic_score is not None:
                st.metric(
                    label="Semantic",
                    value=f"{semantic_score:.3f}",
                    delta=None,
                )
            else:
                st.metric(label="Semantic", value="—")

        with col_density:
            if density_score is not None:
                st.metric(
                    label="Density",
                    value=int(density_score),
                    delta=None,
                )
            else:
                st.metric(label="Density", value="—")

        # ── Existence indicator ───────────────────────────────────────────────
        exists_label = "✅ Found in graph" if existence else "❌ Not found in graph"
        st.caption(exists_label)

        # ── Expandable top matches ────────────────────────────────────────────
        if top_matches:
            with st.expander(f"Top corpus matches ({len(top_matches)})"):
                # Table header
                header_cols = st.columns([4, 1.5, 1.5])
                header_cols[0].markdown("**Case**")
                header_cols[1].markdown("**Score**")
                header_cols[2].markdown("**Case ID**")
                st.divider()

                for match in top_matches:
                    match_cols = st.columns([4, 1.5, 1.5])

                    case_name  = match.get("case_name") or match.get("name") or "Unknown"
                    match_score = match.get("score") or match.get("rrf_score")
                    case_id    = match.get("case_id") or match.get("id")

                    match_cols[0].markdown(f"*{case_name}*")

                    if match_score is not None:
                        match_cols[1].markdown(f"`{float(match_score):.3f}`")
                    else:
                        match_cols[1].markdown("—")

                    if case_id is not None:
                        match_cols[2].markdown(f"`{case_id}`")
                    else:
                        match_cols[2].markdown("—")
        else:
            with st.expander("Top corpus matches"):
                st.caption("No matches returned.")

        st.divider()


def render_summary(citations: list[dict]) -> None:
    """Render a summary bar showing verdict counts."""
    total       = len(citations)
    real        = sum(1 for c in citations if c.get("verdict") == "REAL")
    suspicious  = sum(1 for c in citations if c.get("verdict") == "SUSPICIOUS")
    hallucinated= sum(1 for c in citations if c.get("verdict") == "HALLUCINATED")

    st.subheader("Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Citations", total)
    col2.metric("🟢 Real",         real)
    col3.metric("🟡 Suspicious",   suspicious)
    col4.metric("🔴 Hallucinated", hallucinated)
    st.divider()


# ── Main logic ────────────────────────────────────────────────────────────────
if check_button:
    if not input_text.strip():
        st.warning("Please paste some legal text before checking.")
    else:
        with st.spinner("Running citation checks..."):
            try:
                result   = call_api(input_text)
                citations = result.get("citations", [])

                if not citations:
                    st.info("No citations were extracted from the provided text.")
                else:
                    render_summary(citations)

                    st.subheader("Citation Details")
                    # Column headers
                    h1, h2, h3, h4 = st.columns([1.5, 4, 1.5, 1.5])
                    h1.markdown("**Verdict**")
                    h2.markdown("**Citation**")
                    h3.markdown("**Semantic**")
                    h4.markdown("**Density**")
                    st.divider()

                    for idx, citation in enumerate(citations):
                        render_citation_result(idx, citation)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the Verit API. "
                    "Make sure FastAPI is running on localhost:8000."
                )
            except requests.exceptions.Timeout:
                st.error(
                    "Request timed out. The text may be too long or the server is busy."
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8rem; margin-top:2rem;'>"
    "Verit · Fourth Amendment Citation Corpus · Week 7 Scaffold"
    "</div>",
    unsafe_allow_html=True,
)