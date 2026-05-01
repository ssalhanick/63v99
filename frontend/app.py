"""
frontend/app.py
Week 9 — Verit Streamlit frontend with Haiku explanations and UMAP visualization.

Tabs:
    ⚖️ Citation Checker  — paste text, get verdicts + streaming Haiku explanations
    🗺️  Corpus Map        — UMAP of embedding space, with submitted citations overlaid

Usage:
    streamlit run frontend/app.py
    (FastAPI must be running on localhost:8000)
"""

import sys
from pathlib import Path
import urllib.parse

# Ensure the project root (Verit/) is on sys.path so that top-level packages
# like `visualization`, `detector`, `config`, etc. are importable when
# Streamlit is launched from any working directory.
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
REQUEST_TIMEOUT = 60

VERDICT_BADGE = {
    "REAL":         ("🟢", "#2e7d32", "Real"),
    "SUSPICIOUS":   ("🟡", "#e65100", "Suspicious"),
    "HALLUCINATED": ("🔴", "#c62828", "Hallucinated"),
}

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .verdict-real         { color: #2e7d32; font-weight: 700; }
    .verdict-suspicious   { color: #e65100; font-weight: 700; }
    .verdict-hallucinated { color: #c62828; font-weight: 700; }
    .citation-string      { font-family: monospace; font-size: 0.95rem; }
    .score-label          { color: #666; font-size: 0.85rem; }
    .llm-box {
        background: #f8f9fa;
        border-left: 3px solid #1976d2;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.92rem;
        margin-top: 6px;
        color: black;
    }
    .correction-box {
        background: #fff8e1;
        color: black;
        border-left: 3px solid #f9a825;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.92rem;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── UMAP cache ────────────────────────────────────────────────────────────────
# Loaded once per session — UMAP over 1,300 vectors takes ~30 seconds.
# Cached so switching tabs doesn't re-run it.

@st.cache_resource(show_spinner="Building corpus map (one-time, ~30s)...")
def _load_umap():
    """Load embeddings and run UMAP. Cached for the session."""
    try:
        from visualization.umap_viz import build_corpus_figure
        return build_corpus_figure(color_by="circuit")
    except Exception as e:
        return None, None, None, None, str(e)


# ── API helper ────────────────────────────────────────────────────────────────

def call_api(text: str, jurisdiction: str = None) -> dict:
    payload = {"text": text}
    if jurisdiction and jurisdiction != "All":
        payload["jurisdiction"] = jurisdiction
    response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


# ── Verdict rendering ─────────────────────────────────────────────────────────

def render_summary(citations: list[dict]) -> None:
    total        = len(citations)
    real         = sum(1 for c in citations if c.get("verdict") == "REAL")
    suspicious   = sum(1 for c in citations if c.get("verdict") == "SUSPICIOUS")
    hallucinated = sum(1 for c in citations if c.get("verdict") == "HALLUCINATED")

    st.subheader("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Citations", total)
    c2.metric("🟢 Real",         real)
    c3.metric("🟡 Suspicious",   suspicious)
    c4.metric("🔴 Hallucinated", hallucinated)
    st.divider()


def render_citation_result(citation: dict, show_llm: bool) -> None:
    """Render one citation card. If show_llm=True, stream Haiku explanation after."""
    verdict         = citation.get("verdict", "UNKNOWN")
    citation_string = citation.get("citation_string", "Unknown")
    semantic_score  = citation.get("semantic_score")
    density_score   = citation.get("density_score")
    existence       = citation.get("exists", False)
    top_matches     = citation.get("top_matches", [])
    context_text    = citation.get("context_text", "")
    llm_check       = citation.get("llm_check")

    emoji, color, label = VERDICT_BADGE.get(verdict, ("⚪", "gray", verdict))

    with st.container():
        col_v, col_c, col_s, col_d, col_docs = st.columns([1.5, 3, 1.2, 1.2, 1.5])

        with col_v:
            st.markdown(
                f'<span style="color:{color}; font-size:1.1rem; font-weight:700;">'
                f'{emoji} {label}</span>',
                unsafe_allow_html=True,
            )

        with col_c:
            st.markdown(
                f'<span class="citation-string">{citation_string}</span>',
                unsafe_allow_html=True,
            )

        with col_s:
            st.metric(
                label="Semantic",
                value=f"{semantic_score:.3f}" if semantic_score is not None else "—",
            )

        with col_d:
            st.metric(
                label="Density",
                value=int(density_score) if density_score is not None else "—",
            )
            
        with col_docs:
            signals = citation.get("confidence_signals") or {}
            has_docs = signals.get("has_doctrines")
            shared = signals.get("mean_shared_doctrines")
            val = f"{shared:.1f}" if shared is not None else ("Yes" if has_docs else "No")
            st.metric(
                label="Doctrines",
                value=val if has_docs is not None else "—",
            )

        st.caption("✅ Found in graph" if existence else "❌ Not found in graph")

        # ── Layer 2b LLM reason ───────────────────────────────────────────────
        if llm_check and not llm_check.get("skipped"):
            if verdict == "SUSPICIOUS":
                st.markdown(
                    f'<div class="correction-box">'
                    f'⚠️ <strong>Why suspicious:</strong> {llm_check["reason"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif verdict == "REAL":
                st.markdown(
                    f'<div class="llm-box">'
                    f'✅ <strong>Proposition verified:</strong> {llm_check["reason"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── Top corpus matches ────────────────────────────────────────────────
        if top_matches:
            expander_label = (
                f"Top corpus matches ({len(top_matches)})"
                if verdict != "HALLUCINATED"
                else f"📚 Closest real cases ({len(top_matches)})"
            )
            with st.expander(expander_label, expanded=(verdict == "HALLUCINATED")):
                h0, h1, h2, h3 = st.columns([4, 1.5, 1.5, 1.5])
                h0.markdown("**Case**")
                h1.markdown("**RRF**")
                h2.markdown("**Case ID**")
                h3.markdown("**Source**")
                st.divider()
                for m in top_matches:
                    case_id   = m.get("case_id")
                    case_name = m.get("case_name") or "Unknown"
                    rrf_score = m.get("rrf_score", 0)

                    r0, r1, r2, r3 = st.columns([4, 1.5, 1.5, 1.5])
                    r0.markdown(f"*{case_name}*")
                    r1.markdown(f"`{rrf_score:.4f}`")
                    r2.markdown(f"`{case_id or '—'}`")
                    if case_name and case_name != "Unknown":
                        search_query = urllib.parse.quote(case_name)
                        cl_url = f"https://www.courtlistener.com/?q={search_query}&type=o&order_by=score+desc&stat_Published=on"
                        r3.markdown(f"[CourtListener ↗]({cl_url})")
                    else:
                        r3.markdown("—")
        else:
            if verdict != "HALLUCINATED":
                with st.expander("Top corpus matches"):
                    st.caption("No matches returned.")

        # ── Haiku explanation ─────────────────────────────────────────────────
        if show_llm:
            from frontend.llm import stream_explanation, stream_correction

            st.markdown("**🤖 Explanation**")
            with st.container():
                st.write_stream(
                    stream_explanation(
                        citation_string = citation_string,
                        verdict         = verdict,
                        semantic_score  = semantic_score,
                        density_score   = density_score,
                        top_matches     = top_matches,
                    )
                )

            # Correction suggestion for hallucinated citations
            if verdict == "HALLUCINATED" and top_matches:
                st.markdown("**💡 Suggested Correction**")
                with st.container():
                    st.write_stream(
                        stream_correction(
                            citation_string = citation_string,
                            context_text    = context_text,
                            top_matches     = top_matches,
                        )
                    )

        st.divider()


# ── Tab layout ────────────────────────────────────────────────────────────────

st.title("⚖️ Verit")
st.caption("Legal Citation Hallucination Detector · Fourth Amendment Corpus")

tab_checker, tab_map, tab_graph = st.tabs(["⚖️ Citation Checker", "🗺️ Corpus Map", "🔗 Citation Graph"])

# ── Tab 1 — Citation Checker ──────────────────────────────────────────────────
with tab_checker:
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

    col_jur, col_btn, col_toggle = st.columns([1, 2, 1])
    with col_jur:
        jurisdiction = st.selectbox(
            "Jurisdiction Filter",
            options=["All", "ohioctapp", "indctapp", "tenncrimapp", "ncctapp"],
            index=0,
            label_visibility="collapsed"
        )
    with col_btn:
        check_button = st.button("🔍 Check Citations", type="primary", use_container_width=True)
    with col_toggle:
        show_llm = st.toggle("🤖 Haiku explanations", value=True)

    st.divider()

    if check_button:
        if not input_text.strip():
            st.warning("Please paste some legal text before checking.")
        else:
            with st.spinner("Running citation checks..."):
                try:
                    result    = call_api(input_text, jurisdiction)
                    citations = result.get("citations", [])
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the Verit API. Make sure FastAPI is running on localhost:8000.")
                    citations = []
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The text may be too long or the server is busy.")
                    citations = []
                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e.response.status_code} — {e.response.text}")
                    citations = []
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                    citations = []

            if citations:
                # Store in session state for UMAP overlay
                st.session_state["last_citations"] = citations

                render_summary(citations)
                st.subheader("Citation Details")

                # Column headers
                h1, h2, h3, h4, h5 = st.columns([1.5, 3, 1.2, 1.2, 1.5])
                h1.markdown("**Verdict**")
                h2.markdown("**Citation**")
                h3.markdown("**Semantic**")
                h4.markdown("**Density**")
                h5.markdown("**Doctrines**")
                st.divider()

                for citation in citations:
                    render_citation_result(citation, show_llm=show_llm)
            elif "citations" in result if 'result' in dir() else False:
                st.info("No citations were extracted from the provided text.")

    st.markdown(
        "<div style='text-align:center; color:#999; font-size:0.8rem; margin-top:2rem;'>"
        "Verit · Fourth Amendment Citation Corpus · Week 9"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Tab 2 — Corpus Map ────────────────────────────────────────────────────────
with tab_map:
    st.subheader("Corpus Embedding Space")
    st.caption(
        "768-dimensional Legal-BERT embeddings reduced to 2D via UMAP "
        "(StandardScaler → n_neighbors=15, min_dist=0.1, cosine metric). "
        "Each point is one case. Color = circuit. "
        "Run a citation check in the other tab to overlay your submitted citations."
    )

    color_by = st.radio(
        "Color by",
        options=["circuit", "year"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # Load UMAP (cached)
    result_tuple = _load_umap()

    # Handle both 4-tuple (success) and 5-tuple (error) returns
    if len(result_tuple) == 5:
        fig, coords, case_ids, df, err = result_tuple
    else:
        fig, coords, case_ids, df = result_tuple
        err = None

    if err or fig is None:
        st.error(
            f"Could not build corpus map: {err or 'Unknown error'}. "
            "Make sure embeddings.parquet exists and umap-learn is installed."
        )
    else:
        # Rebuild figure with selected color_by if changed
        # (coords are cached, only Plotly figure needs to be rebuilt)
        if color_by != "circuit":
            try:
                import plotly.express as px
                fig = px.scatter(
                    df,
                    x="x", y="y",
                    color=color_by,
                    hover_name="name",
                    hover_data={"x": False, "y": False, "court_id": True, "year": True, "cite_count": True},
                    title=f"Verit Corpus — UMAP Embedding Space (colored by {color_by})",
                    labels={"x": "UMAP-1", "y": "UMAP-2"},
                    template="plotly_white",
                    opacity=0.75,
                )
                fig.update_traces(marker=dict(size=5))
                fig.update_layout(height=600)
            except Exception as e:
                st.error(f"Could not rebuild figure: {e}")

        # Overlay submitted citations if available
        if "last_citations" in st.session_state and st.session_state["last_citations"]:
            from visualization.umap_viz import overlay_submitted_citations
            overlay_submitted_citations(fig, coords, case_ids, st.session_state["last_citations"])
            n = len(st.session_state["last_citations"])
            st.caption(
                f"★ Overlaid {n} submitted citation(s) — corpus dimmed for focus. "
                "⭐ Stars=REAL · ◆ Diamonds=SUSPICIOUS · ✕ X=HALLUCINATED. "
                "Hallucinated citations without a corpus match are placed at their "
                "closest semantic neighbors' centroid."
            )

        st.plotly_chart(fig, use_container_width=True)

# ── Tab 3 — Citation Graph ────────────────────────────────────────────────────
with tab_graph:
    st.subheader("Citation Graph Explorer")
    st.caption(
        "Neo4j subgraph for REAL and SUSPICIOUS citations. "
        "Node size = citation count. "
        "🔴 Submitted case · 🟡 Landmark · 🔵 Corpus · ⚫ Stub"
    )

    # Guard: need a prior check result
    if "last_citations" not in st.session_state or not st.session_state["last_citations"]:
        st.info("Run a citation check in the ⚖️ Citation Checker tab first.")
    else:
        # Filter to only REAL / SUSPICIOUS — HALLUCINATED are not in Neo4j
        graphable = [
            c for c in st.session_state["last_citations"]
            if c.get("verdict") in ("REAL", "SUSPICIOUS")
        ]

        if not graphable:
            st.info("No REAL or SUSPICIOUS citations in the last check result.")
        else:
            # ── Controls ──────────────────────────────────────────────────────
            col_drop, col_hops = st.columns([4, 1])

            with col_drop:
                VERDICT_ICON = {"REAL": "🟢", "SUSPICIOUS": "🟡"}

                options = {
                    f"{VERDICT_ICON.get(c['verdict'], '⚪')} "
                    f"{c.get('case_name') or c['citation_string']} "
                    f"({c['citation_string']})": c
                    for c in graphable
                }
                selected_label = st.selectbox(
                    "Select citation to explore",
                    options=list(options.keys()),
                )
                selected_citation = options[selected_label]

            with col_hops:
                hops = st.radio(
                    "Hops",
                    options=[1, 2],
                    index=1,
                    horizontal=True,
                )

            st.divider()

            # ── Graph render ──────────────────────────────────────────────────
            case_id = selected_citation["case_id"]

            try:
                from visualization.graph_viz import render_graph_html

                with st.spinner("Querying Neo4j and building graph..."):
                    graph_result = render_graph_html(case_id, hops=hops)

                if graph_result is None:
                    st.warning(
                        f"No citation relationships found in Neo4j for case ID {case_id}. "
                        "The case may exist as a node but have no outgoing CITES edges."
                    )
                else:
                    html, node_count, edge_count = graph_result
                    st.caption(f"**{node_count}** nodes · **{edge_count}** edges · {hops}-hop neighborhood")
                    st.components.v1.html(html, height=620, scrolling=False)

                    # ── Neo4j Browser button ──────────────────────────────────
                    import urllib.parse
                    cypher = (
                        f"MATCH path = (c:Case {{id: {case_id}}})-[:CITES*1..2]->(neighbor:Case)\n"
                        f"RETURN path\n"
                        f"LIMIT 100"
                    )
                    encoded = urllib.parse.quote(cypher)
                    neo4j_url = f"http://localhost:7474/browser/?cmd=edit&arg={encoded}"

                    st.link_button("🔍 Open in Neo4j Browser", neo4j_url)

            except Exception as e:
                st.error(f"Graph render failed: {e}")