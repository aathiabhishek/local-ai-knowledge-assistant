import streamlit as st
import textwrap
from pathlib import Path
from datetime import datetime
import time

from src.config import (
    APP_TITLE,
    MODEL_PATH,
    EMBEDDING_MODEL,
    TOP_K,
    MAX_HISTORY_MESSAGES,
)

from src.llm import LocalLLM
from src.ingestion import load_uploaded_files
from src.retriever import HybridRetriever


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html: str) -> None:
    st.markdown(
        textwrap.dedent(html).strip(),
        unsafe_allow_html=True,
    )


# Typing indicator markup, defined flush-left at module level so
# it never needs dedenting and can be reused for both RAG search
# and generation status states.
TYPING_INDICATOR = """<div class="typing"><span></span><span></span><span></span></div>"""


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #0e1117;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 0.5rem;
        padding-bottom: 7rem;
    }

    header {
        visibility: hidden;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #11151d;
        border-right: 1px solid #252b36;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        padding: 8px 5px 20px 5px;
    }

    .sidebar-brand-title {
        font-size: 21px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .sidebar-brand-subtitle {
        font-size: 12px;
        color: #8b949e;
    }

    .sidebar-section {
        font-size: 12px;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .status-card {
        background: #171c25;
        border: 1px solid #272e3a;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }

    .status-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: #d7dde5;
        margin: 5px 0;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    .status-green { background: #3fb950; }
    .status-yellow { background: #d29922; }
    .status-red { background: #f85149; }


    /* ========================================================
       CHAT HEADER BAR (real-chatbot feel)
       ======================================================== */

    .chat-header {
        position: sticky;
        top: 0;
        z-index: 999;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 18px;
        margin: 0.5rem 0 0.5rem 0;
        background: #151a22;
        border: 1px solid #252b36;
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    }

    .chat-header-avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6e56cf, #8b5cf6);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
    }

    .chat-header-name {
        color: #ffffff;
        font-weight: 700;
        font-size: 15px;
    }

    .chat-header-status {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #8b949e;
        margin-top: 2px;
    }


    /* ========================================================
       CHAT MESSAGES — bubble styling
       ======================================================== */

    [data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 0 !important;
        margin: 4px 0;
        max-width: 100%;
    }

    /* Marker + :has() hook: a zero-height marker div is rendered
       immediately before each st.chat_message() call, tagged with
       the message role. This lets pure CSS reposition/recolor the
       *following* chat bubble without touching Python layout code,
       since Streamlit gives every top-level call its own wrapper
       sibling. Falls back gracefully to default styling on older
       browsers without :has() support. */

    div:has(> div > .row-marker-user) + div [data-testid="stChatMessage"] {
        flex-direction: row-reverse;
    }

    div:has(> div > .row-marker-user) + div [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    div:has(> div > .row-marker-user) + div [data-testid="stChatMessage"] > div:nth-child(2) {
        background: linear-gradient(135deg, #6e56cf, #7c5cdb);
        color: #ffffff;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 15px !important;
        max-width: 75%;
        margin-left: auto;
    }

    div:has(> div > .row-marker-bot) + div [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    div:has(> div > .row-marker-bot) + div [data-testid="stChatMessage"] > div:nth-child(2) {
        background: #1c222c;
        border: 1px solid #272e3a;
        color: #e6edf3;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 15px !important;
        max-width: 75%;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        font-size: 14.5px;
        line-height: 1.6;
    }

    [data-testid*="Avatar"] {
        width: 30px !important;
        height: 30px !important;
        font-size: 15px !important;
    }

    .msg-timestamp {
        font-size: 10.5px;
        color: #5c6472;
        margin: -2px 0 2px 44px;
    }

    div:has(> div > .row-marker-user) + div + div .msg-timestamp {
        text-align: right;
        margin: -2px 44px 2px 0;
    }


    /* ========================================================
       TYPING INDICATOR
       ======================================================== */

    .typing {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 10px 15px;
        background: #1c222c;
        border: 1px solid #272e3a;
        border-radius: 18px 18px 18px 4px;
    }

    .typing span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #8b949e;
        animation: typing-bounce 1.2s infinite ease-in-out;
    }

    .typing span:nth-child(2) { animation-delay: 0.15s; }
    .typing span:nth-child(3) { animation-delay: 0.3s; }

    @keyframes typing-bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-4px); opacity: 1; }
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {
        bottom: 20px;
    }

    [data-testid="stChatInput"] textarea {
        background: #171c25 !important;
        border: 1px solid #303846 !important;
        border-radius: 22px !important;
        color: #ffffff !important;
        padding: 14px 18px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #6e7681 !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #6e56cf !important;
        box-shadow: 0 0 0 1px #6e56cf !important;
    }


    /* ========================================================
       SOURCE CARDS
       ======================================================== */

    .source-card {
        background: #151a22;
        border: 1px solid #282f3a;
        border-radius: 10px;
        padding: 10px 12px;
        margin: 6px 0;
        font-size: 12px;
        color: #aeb7c4;
    }

    .source-title {
        color: #e6edf3;
        font-weight: 600;
    }


    /* ========================================================
       DOCUMENT CARDS
       ======================================================== */

    .document-card {
        background: #171c25;
        border: 1px solid #272e3a;
        border-radius: 10px;
        padding: 10px 12px;
        margin: 5px 0;
        font-size: 12px;
        color: #c9d1d9;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }


    /* ========================================================
       HERO / SUGGESTIONS
       ======================================================== */

    .hero {
        text-align: center;
        padding: 50px 20px 30px 20px;
    }

    .hero-icon {
        width: 64px;
        height: 64px;
        border-radius: 20px;
        background: linear-gradient(135deg, #6e56cf, #8b5cf6);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 18px auto;
        font-size: 30px;
        box-shadow: 0 12px 40px rgba(111, 86, 207, 0.25);
    }

    .hero-title {
        font-size: 28px;
        font-weight: 750;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #8b949e;
        font-size: 14px;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .suggestion-label {
        color: #8b949e;
        font-size: 12px;
        text-align: center;
        margin-top: 18px;
        margin-bottom: 8px;
    }


    /* ========================================================
       RESPONSE METADATA
       ======================================================== */

    .response-meta {
        color: #6e7681;
        font-size: 11px;
        margin: 2px 0 0 4px;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
        [data-testid="stChatMessage"] > div:nth-child(2) {
            max-width: 88% !important;
        }

        .hero-title { font-size: 24px; }
        .hero-subtitle { font-size: 13px; }
    }

    </style>
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "llm" not in st.session_state:
    st.session_state.llm = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# LOAD LOCAL LLM
# ------------------------------------------------------------
# Runs above the sidebar/header so st.session_state.llm is fully
# resolved before anything reads it for status display.
# ============================================================

if st.session_state.llm is None:

    if not Path(MODEL_PATH).exists():

        st.error(
            f"""
            **Local model not found**

            Expected model:

            `{MODEL_PATH}`

            Update `MODEL_PATH` in
            `src/config.py` or place your
            Mistral GGUF model at the configured location.
            """
        )

    else:

        with st.spinner("Loading local Mistral model..."):

            try:

                st.session_state.llm = LocalLLM(MODEL_PATH)

            except Exception as exc:

                st.error(
                    f"Could not load the local model: {exc}"
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                💬 Local AI Assistant
            </div>
            <div class="sidebar-brand-subtitle">
                Private local knowledge assistant
            </div>
        </div>
        """
    )

    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    render_html('<div class="sidebar-section">System</div>')

    model_exists = Path(MODEL_PATH).exists()
    model_loaded = st.session_state.llm is not None

    if model_loaded:
        model_status = """
        <div class="status-row">
            <span class="status-dot status-green"></span>
            <span>Model loaded</span>
        </div>
        """
    elif model_exists:
        model_status = """
        <div class="status-row">
            <span class="status-dot status-yellow"></span>
            <span>Model found, not loaded</span>
        </div>
        """
    else:
        model_status = """
        <div class="status-row">
            <span class="status-dot status-red"></span>
            <span>Model not found</span>
        </div>
        """

    if st.session_state.retriever is not None:
        rag_status = """
        <div class="status-row">
            <span class="status-dot status-green"></span>
            <span>Knowledge base active</span>
        </div>
        """
    else:
        rag_status = """
        <div class="status-row">
            <span class="status-dot status-yellow"></span>
            <span>General chat mode</span>
        </div>
        """

    render_html(
        f"""
        <div class="status-card">
            {model_status}
            {rag_status}
        </div>
        """
    )

    render_html('<div class="sidebar-section">Generation</div>')

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values produce more focused and deterministic answers.",
    )

    render_html('<div class="sidebar-section">Knowledge Base</div>')

    uploaded_files = st.file_uploader(
        "Add documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload documents to enable document-aware RAG.",
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=TOP_K,
        step=1,
        help="Number of relevant document chunks sent to the LLM.",
    )

    if uploaded_files:

        current_names = sorted([file.name for file in uploaded_files])

        if current_names != st.session_state.indexed_files:

            with st.spinner("Indexing documents..."):

                try:

                    documents = load_uploaded_files(uploaded_files)

                    if not documents:

                        st.warning(
                            "No readable text was found in the uploaded files."
                        )

                    else:

                        retriever = HybridRetriever(
                            embedding_model_name=EMBEDDING_MODEL
                        )
                        retriever.build(documents)

                        st.session_state.documents = documents
                        st.session_state.retriever = retriever
                        st.session_state.indexed_files = current_names

                        st.success(f"Indexed {len(documents)} chunks.")

                except Exception as exc:

                    st.error(f"Document indexing failed: {exc}")

    # If every file is removed from the uploader, `uploaded_files`
    # becomes empty/falsy, so the block above never runs. This
    # branch detects that and clears the stale knowledge base.
    elif st.session_state.indexed_files:

        st.session_state.documents = []
        st.session_state.retriever = None
        st.session_state.indexed_files = []
        st.rerun()

    if st.session_state.indexed_files:

        render_html('<div class="sidebar-section">Indexed Documents</div>')

        for name in st.session_state.indexed_files:
            render_html(f'<div class="document-card">📄 {name}</div>')

        if st.button("🗑️ Clear knowledge base", use_container_width=True):
            st.session_state.documents = []
            st.session_state.retriever = None
            st.session_state.indexed_files = []
            st.rerun()

    st.divider()
    st.caption("🔒 Your model and documents run locally.")
    st.caption("No uploaded documents are sent to an external API.")


# ============================================================
# CHAT HEADER BAR
# ============================================================

_header_dot = "status-green" if st.session_state.llm is not None else "status-red"
_header_label = "Online" if st.session_state.llm is not None else "Offline"

render_html(
    f"""
    <div class="chat-header">
        <div class="chat-header-avatar">💬</div>
        <div>
            <div class="chat-header-name">Local AI Assistant</div>
            <div class="chat-header-status">
                <span class="status-dot {_header_dot}"></span>
                <span>{_header_label}</span>
            </div>
        </div>
    </div>
    """
)


# ============================================================
# HELPERS
# ============================================================

def render_sources(sources):
    """Render a list of retrieved-chunk source cards under a message."""

    with st.expander(f"📚 Sources ({len(sources)})"):

        for source in sources:

            page_text = ""

            # `is not None` (rather than truthiness) preserves a
            # legitimate page number of 0.
            if source.get("page") is not None:
                page_text = f" · Page {source['page']}"

            render_html(
                f"""
                <div class="source-card">
                    <span class="source-title">📄 {source['source']}</span>
                    {page_text}
                    <br>
                    Relevance: <code>{source['score']:.3f}</code>
                </div>
                """
            )


def render_row_marker(role: str) -> None:
    """Zero-height marker used purely as a CSS hook (see the
    :has() rules in the stylesheet) so the *next* st.chat_message
    can be re-aligned/re-colored per role without custom HTML
    replacing st.chat_message's own (correct) markdown rendering."""

    css_class = "row-marker-user" if role == "user" else "row-marker-bot"
    render_html(f'<div class="{css_class}"></div>')


def render_timestamp(ts: str) -> None:
    if ts:
        render_html(f'<div class="msg-timestamp">{ts}</div>')


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    render_html(
        """
        <div class="hero">
            <div class="hero-icon">💬</div>
            <div class="hero-title">How can I help you?</div>
            <div class="hero-subtitle">
                Ask questions about AI, machine learning,
                programming, or upload your own documents
                for grounded answers using local RAG.
            </div>
        </div>
        """
    )

    render_html('<div class="suggestion-label">Try asking</div>')

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧠 What is an LLM?", use_container_width=True):
            st.session_state.pending_prompt = "What is an LLM?"
            st.rerun()

    with col2:
        if st.button("🔎 What is RAG?", use_container_width=True):
            st.session_state.pending_prompt = (
                "What is Retrieval-Augmented Generation (RAG)?"
            )
            st.rerun()

    with col3:
        if st.button("📚 How does RAG work?", use_container_width=True):
            st.session_state.pending_prompt = (
                "How does Retrieval-Augmented Generation work?"
            )
            st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message.get("role", "assistant")

    render_row_marker(role)

    with st.chat_message(
        role,
        avatar="🧑" if role == "user" else "💬",
    ):

        st.markdown(message.get("content", ""))

        if message.get("sources"):
            render_sources(message["sources"])

        if message.get("elapsed") is not None:
            render_html(
                f"""
                <div class="response-meta">
                    ⚡ {message['elapsed']:.2f}s
                </div>
                """
            )

    render_timestamp(message.get("timestamp", ""))


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input("Message your local AI assistant...")

if st.session_state.pending_prompt and not prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# ============================================================
# GENERATE RESPONSE
# ============================================================

if prompt:

    now = datetime.now().strftime("%I:%M %p")

    # --------------------------------------------------------
    # SAVE + DISPLAY USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": now}
    )

    render_row_marker("user")

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    render_timestamp(now)

    # --------------------------------------------------------
    # DISPLAY ASSISTANT
    # --------------------------------------------------------

    render_row_marker("bot")

    with st.chat_message("assistant", avatar="💬"):

        status = st.empty()
        start = time.perf_counter()

        # elapsed stays None unless a generation actually runs, so
        # "model not loaded" / error cases never get a fake timing
        # value persisted or displayed.
        elapsed = None

        try:

            if st.session_state.llm is None:

                answer = (
                    "The local Mistral model is not loaded. "
                    "Please check the model path and installation."
                )
                sources = []
                status.empty()
                st.error(answer)

            else:

                if st.session_state.retriever is not None:

                    status.markdown(TYPING_INDICATOR, unsafe_allow_html=True)

                    retrieved = st.session_state.retriever.search(
                        prompt, top_k=top_k
                    )

                    status.markdown(TYPING_INDICATOR, unsafe_allow_html=True)

                    answer = st.session_state.llm.generate_rag_answer(
                        question=prompt,
                        retrieved_chunks=retrieved,
                        history=st.session_state.messages[-MAX_HISTORY_MESSAGES:-1],
                        temperature=temperature,
                    )

                    sources = [
                        {
                            "source": item["source"],
                            "page": item.get("page"),
                            "score": item["score"],
                        }
                        for item in retrieved
                    ]

                    status.empty()

                else:

                    status.markdown(TYPING_INDICATOR, unsafe_allow_html=True)

                    answer = st.session_state.llm.generate_chat_answer(
                        question=prompt,
                        history=st.session_state.messages[-MAX_HISTORY_MESSAGES:-1],
                        temperature=temperature,
                    )

                    sources = []
                    status.empty()

                elapsed = time.perf_counter() - start

                st.markdown(answer)

                if sources:
                    render_sources(sources)

                render_html(
                    f"""
                    <div class="response-meta">
                        ⚡ {elapsed:.2f}s
                    </div>
                    """
                )

        except Exception as exc:

            answer = f"Sorry, an error occurred: `{exc}`"
            sources = []
            status.empty()
            st.error(answer)

    reply_time = datetime.now().strftime("%I:%M %p")
    render_timestamp(reply_time)

    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "elapsed": elapsed,
            "timestamp": reply_time,
        }
    )

    st.rerun()