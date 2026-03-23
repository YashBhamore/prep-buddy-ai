"""
app.py
======
Prep Buddy AI — Claude + LangGraph + ChromaDB + Streamlit

A document-based learning assistant with streaming responses,
multi-format ingestion, and persistent chat history.

Run: streamlit run app.py
"""

import streamlit as st

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, EMBEDDING_MODEL, SUPPORTED_EXTENSIONS
from agent import chat_stream
from vector_store import (
    ingest_documents,
    get_collection_stats,
    delete_source,
    clear_collection,
)
from chat_history import (
    create_conversation,
    save_message,
    get_conversations,
    get_messages,
    delete_conversation,
)


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Prep Buddy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ───────────────────────────────────────── */
    html, body, .main, .stApp {
        font-family: 'DM Sans', sans-serif !important;
    }
    .main .block-container {
        padding: 1rem 2rem 3rem;
        max-width: 960px;
    }

    /* ── Header ───────────────────────────────────────── */
    .app-header {
        background: linear-gradient(135deg, #0c0f1a 0%, #1a1f3a 50%, #0d1b2a 100%);
        border-radius: 16px;
        padding: 1.75rem 2.25rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(99, 102, 241, 0.15);
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .app-header h1 {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 0.35rem;
        position: relative;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.82rem;
        position: relative;
        line-height: 1.6;
    }
    .badge-row {
        margin-top: 0.85rem;
        position: relative;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .tech-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.68rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
        font-weight: 500;
    }

    /* ── Chat Messages ────────────────────────────────── */
    .stChatMessage {
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* ── Source Cards ──────────────────────────────────── */
    .src-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #6366f1;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .src-card:hover { border-left-color: #4f46e5; }
    .src-name {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .src-score {
        font-weight: 600;
        font-size: 0.75rem;
        padding: 1px 7px;
        border-radius: 4px;
        margin-left: 8px;
    }
    .score-high { background: #dcfce7; color: #15803d; }
    .score-mid  { background: #fef3c7; color: #b45309; }
    .score-low  { background: #fee2e2; color: #dc2626; }
    .src-text {
        color: #64748b;
        font-size: 0.78rem;
        font-style: italic;
        margin-top: 0.35rem;
        line-height: 1.55;
    }

    /* ── Sidebar ──────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
        margin-bottom: 0.5rem;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.8rem;
        transition: all 0.2s;
    }

    /* ── Stat Card ─────────────────────────────────────── */
    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stat-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #6366f1;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-label {
        font-size: 0.65rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    /* ── Conv History ──────────────────────────────────── */
    .conv-item {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.4rem;
        cursor: pointer;
        transition: all 0.15s;
    }
    .conv-item:hover {
        border-color: #6366f1;
        background: #f5f3ff;
    }
    .conv-item.active {
        border-color: #6366f1;
        background: #eef2ff;
        border-left: 3px solid #6366f1;
    }
    .conv-title {
        font-size: 0.8rem;
        font-weight: 500;
        color: #1e293b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .conv-time {
        font-size: 0.65rem;
        color: #94a3b8;
    }

    /* ── Empty State ───────────────────────────────────── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #94a3b8;
    }
    .empty-icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }
    .empty-state h3 {
        color: #475569;
        font-weight: 600;
        font-size: 1.15rem;
        margin-bottom: 0.5rem;
    }
    .empty-state p {
        font-size: 0.85rem;
        line-height: 1.65;
        max-width: 380px;
        margin: 0 auto;
    }

    /* ── File Chip ─────────────────────────────────────── */
    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 5px 10px;
        font-size: 0.78rem;
        color: #334155;
        margin: 2px;
    }
    .file-chip .ext {
        background: #6366f1;
        color: white;
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }

    /* ── Misc ──────────────────────────────────────────── */
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    </style>
    """, unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "chat_history": [],
        "last_sources": [],
        "current_conv_id": None,
        "_last_sources": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # ── New Chat ──────────────────────────────────────
        if st.button("✦  New Chat", use_container_width=True, type="primary"):
            st.session_state.chat_history = []
            st.session_state.last_sources = []
            st.session_state.current_conv_id = None
            st.rerun()

        st.markdown("---")

        # ── Document Management ───────────────────────────
        st.markdown("### Documents")

        exts = [e.replace(".", "") for e in SUPPORTED_EXTENSIONS]
        uploaded_files = st.file_uploader(
            "Upload files to the knowledge base",
            type=exts,
            accept_multiple_files=True,
            help=f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

        if uploaded_files:
            if st.button("⬆ Ingest", use_container_width=True):
                with st.spinner("Processing documents..."):
                    result = ingest_documents(uploaded_files)

                if result["errors"]:
                    for err in result["errors"]:
                        st.error(err)
                else:
                    st.success(
                        f"✓ {result['ingested']} chunks indexed "
                        f"({result['skipped']} duplicates skipped)"
                    )

        # ── Corpus Stats ─────────────────────────────────
        stats = get_collection_stats()

        if stats["total_chunks"] > 0:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-val">{stats["total_chunks"]}</div>'
                f'<div class="stat-label">Indexed Chunks</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("**Indexed Sources**")
            for source in stats["sources"]:
                ext = source.rsplit(".", 1)[-1] if "." in source else "?"
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(
                        f'<div class="file-chip">'
                        f'<span class="ext">{ext.upper()}</span> {source}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("✕", key=f"del_{source}", help=f"Remove {source}"):
                        deleted = delete_source(source)
                        st.toast(f"Removed {deleted} chunks from {source}")
                        st.rerun()

            if st.button("🗑 Clear All Documents", type="secondary", use_container_width=True):
                clear_collection()
                st.toast("Knowledge base cleared")
                st.rerun()
        else:
            st.info("No documents ingested yet.")

        st.markdown("---")

        # ── Conversation History ──────────────────────────
        st.markdown("### History")

        conversations = get_conversations()
        if conversations:
            for conv in conversations[:15]:
                is_active = st.session_state.current_conv_id == conv["id"]
                css_class = "conv-item active" if is_active else "conv-item"

                # Format time
                time_str = conv["updated_at"][:16].replace("T", " ")

                col1, col2 = st.columns([5, 1])
                with col1:
                    if st.button(
                        f"💬 {conv['title'][:35]}",
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                    ):
                        _load_conversation(conv["id"])
                        st.rerun()
                with col2:
                    if st.button("✕", key=f"delconv_{conv['id']}", help="Delete"):
                        delete_conversation(conv["id"])
                        if st.session_state.current_conv_id == conv["id"]:
                            st.session_state.chat_history = []
                            st.session_state.current_conv_id = None
                        st.rerun()
        else:
            st.caption("No conversation history yet.")

        # ── System Info ───────────────────────────────────
        st.markdown("---")
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;">'
            f'<span class="tech-badge">Claude {CLAUDE_MODEL.split("-")[1] if "-" in CLAUDE_MODEL else ""}</span>'
            f'<span class="tech-badge">{EMBEDDING_MODEL}</span>'
            f'<span class="tech-badge">ChromaDB</span>'
            f'<span class="tech-badge">LangGraph</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _load_conversation(conv_id: str):
    """Load a conversation from the database into session state."""
    messages = get_messages(conv_id)
    st.session_state.chat_history = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    # Load the last assistant's sources if available
    last_assistant = [m for m in messages if m["role"] == "assistant"]
    st.session_state.last_sources = (
        last_assistant[-1]["sources"] if last_assistant and last_assistant[-1]["sources"] else []
    )
    st.session_state.current_conv_id = conv_id


# ── Header ───────────────────────────────────────────────────────────────────

def render_header():
    st.markdown(
        '<div class="app-header">'
        "<h1>Prep Buddy AI</h1>"
        "<p>Upload notes and documents, study them, and ask grounded questions "
        "powered by Claude and retrieval-augmented generation.</p>"
        '<div class="badge-row">'
        '<span class="tech-badge">Anthropic Claude</span>'
        '<span class="tech-badge">LangGraph</span>'
        '<span class="tech-badge">ChromaDB</span>'
        '<span class="tech-badge">Streamlit</span>'
        '<span class="tech-badge">RAG</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Source Cards ─────────────────────────────────────────────────────────────

def _score_class(score: float) -> str:
    if score >= 0.7:
        return "score-high"
    if score >= 0.5:
        return "score-mid"
    return "score-low"


def render_sources(sources: list[dict]):
    """Render source citation cards in an expander."""
    if not sources:
        return

    with st.expander(f"📎 {len(sources)} source(s) referenced", expanded=False):
        for chunk in sources:
            sc = _score_class(chunk["score"])
            snippet = chunk["text"][:250].replace("\n", " ")
            st.markdown(
                f'<div class="src-card">'
                f'<span class="src-name">📄 {chunk["source"]}</span>'
                f'<span class="src-score {sc}">{chunk["score"]:.1%}</span>'
                f'<div class="src-text">{snippet}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Chat Interface ───────────────────────────────────────────────────────────

def render_chat():
    # API key check
    if not ANTHROPIC_API_KEY:
        st.warning(
            "⚠️ **ANTHROPIC_API_KEY** not set. "
            "Create a `.env` file with your key. "
            "[Get one here →](https://console.anthropic.com/)"
        )
        return

    # Empty state
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">⚡</div>'
            "<h3>What would you like to know?</h3>"
            "<p>Upload documents using the sidebar, then ask questions. "
            "Responses are grounded in your documents — no hallucination.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Show sources from last response
    render_sources(st.session_state.last_sources)

    # Chat input
    if query := st.chat_input("Ask about your documents..."):

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.chat_history.append({
            "role": "user",
            "content": query,
        })

        # Create conversation in DB if needed
        if st.session_state.current_conv_id is None:
            st.session_state.current_conv_id = create_conversation()

        # Save user message
        save_message(st.session_state.current_conv_id, "user", query)

        # Stream response
        with st.chat_message("assistant"):
            response = st.write_stream(
                chat_stream(query, st.session_state.chat_history[:-1])
            )

        # Get sources from the stream function
        sources = st.session_state.get("_last_sources", [])
        st.session_state.last_sources = sources

        # Save to history and DB
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
        })
        save_message(
            st.session_state.current_conv_id,
            "assistant",
            response,
            sources,
        )

        st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    init_state()
    render_header()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
