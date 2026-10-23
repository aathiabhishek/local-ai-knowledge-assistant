import streamlit as st
import textwrap
from pathlib import Path
from datetime import datetime
import time
import os

from dotenv import load_dotenv

from src.config import (
    APP_TITLE,
    MODEL_PATH,
    EMBEDDING_MODEL,
    TOP_K,
    MAX_HISTORY_MESSAGES,
)

from src.llm import GroqLLM, LocalLLM, MAX_CHUNKS
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


# Load local environment variables and the external stylesheet.
load_dotenv(Path(__file__).resolve().parent / ".env")

CSS_PATH = Path(__file__).resolve().parent / "styles.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# Typing indicator markup, defined flush-left at module level so
# it never needs dedenting and can be reused for both RAG search
# and generation status states.
TYPING_INDICATOR = """<div class="typing"><span></span><span></span><span></span></div>"""


# ============================================================
# CUSTOM CSS
# ============================================================




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

if "model_backend" not in st.session_state:
    st.session_state.model_backend = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# LOAD MODEL
# ------------------------------------------------------------
# Prefer a local GGUF model when present. Otherwise, use Groq if
# GROQ_API_KEY is available in .env or Streamlit Secrets.
# ------------------------------------------------------------

try:
    if "GROQ_API_KEY" in st.secrets and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    # Streamlit Secrets may be unavailable during local development.
    pass


if st.session_state.llm is None:
    has_local_gguf = Path(MODEL_PATH).exists()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    local_load_error = None
    if has_local_gguf:
        with st.spinner("Loading local Llama model..."):
            try:
                st.session_state.llm = LocalLLM(MODEL_PATH)
                st.session_state.model_backend = "local"
            except Exception as exc:
                local_load_error = exc
                # If the local model exists but cannot initialize, try Groq below.

    if st.session_state.llm is None and groq_api_key:
        with st.spinner("Local model unavailable. Connecting to Groq..."):
            try:
                st.session_state.llm = GroqLLM(groq_api_key)
                st.session_state.model_backend = "groq"
                if local_load_error:
                    st.info("The local model could not be loaded, so the app switched to Groq.")
            except Exception as exc:
                st.error(f"Could not initialize Groq: {exc}")
                if local_load_error:
                    st.caption(f"Local model error: {local_load_error}")
    elif st.session_state.llm is None:
        if local_load_error:
            st.error(f"Could not load the local model: {local_load_error}")
        st.warning(
            "No usable model is configured. Add `GROQ_API_KEY=...` to the project "
            "`.env` file (Groq fallback uses `openai/gpt-oss-120b` by default), "
            "or place a working GGUF model at "
            f"`{MODEL_PATH}`."
        )


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

    model_exists = Path(MODEL_PATH).exists() or bool(os.getenv("GROQ_API_KEY"))

    model_loaded = st.session_state.llm is not None

    if model_loaded and st.session_state.model_backend == "groq":
        active_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        model_status = f"""
        <div class="status-row">
            <span class="status-dot status-green"></span>
            <span><strong>Groq API active</strong></span>
        </div>
        <div class="model-detail">Cloud model: {active_model}</div>
        """
    elif model_loaded and st.session_state.model_backend == "local":
        local_model_name = Path(MODEL_PATH).name
        model_status = f"""
        <div class="status-row">
            <span class="status-dot status-green"></span>
            <span><strong>Local model active</strong></span>
        </div>
        <div class="model-detail">On-device: {local_model_name}</div>
        """
    elif model_loaded:
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
    if getattr(st.session_state, "model_backend", None) == "groq":
        st.caption("☁️ Responses use Groq's hosted API.")
        st.caption("Relevant prompts and retrieved document excerpts are sent to Groq.")
    else:
        st.caption("🔒 The model runs locally on this machine.")
        st.caption("Documents are processed locally by the app.")


# ============================================================
# CHAT HEADER BAR
# ============================================================

_header_dot = "status-green" if st.session_state.llm is not None else "status-red"
if st.session_state.llm is None:
    _header_label = "Offline · No model connected"
elif st.session_state.model_backend == "groq":
    _header_label = f"Online · Groq API · {os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b')}"
elif st.session_state.model_backend == "local":
    _header_label = f"Online · Local model · {Path(MODEL_PATH).name}"
else:
    _header_label = "Online · Model connected"

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
                    "No language model is connected. Check your local model path or GROQ_API_KEY configuration."
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

                    # Both backends share the same RAG preparation rules and
                    # use the same top-N chunks as the shared prompt builder.
                    retrieved_for_answer = retrieved[:MAX_CHUNKS]
                    answer = st.session_state.llm.generate_rag_answer(
                        question=prompt,
                        retrieved_chunks=retrieved_for_answer,
                        history=st.session_state.messages[-MAX_HISTORY_MESSAGES:-1],
                        temperature=temperature,
                    )

                    sources = [
                        {
                            "source": item.get("source", "Unknown source"),
                            "page": item.get("page"),
                            "score": item.get("score"),
                        }
                        for item in retrieved_for_answer
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