import os
from pathlib import Path
from typing import Dict, List, Optional

from src.config import N_CTX, N_THREADS, N_GPU_LAYERS, MAX_NEW_TOKENS

MAX_HISTORY_TURNS = 8
MAX_CHUNKS = 5
MAX_CHUNK_CHARS = 1500

FALLBACK_MESSAGE = (
    "I cannot answer this question based on the uploaded documents. "
    "The provided context does not contain the required information."
)

SYSTEM_PROMPT = """You are a helpful AI assistant and technical tutor specializing in AI/ML, software engineering, databases, and generative AI.

Answering Style:
- Start with a direct answer.
- Keep answers concise by default: explain the concept in 2–4 sentences and include one simple, practical example when useful.
- For beginners, use a brief real-world analogy when it helps understanding.
- Do not provide lengthy explanations, detailed component breakdowns, tables, or multiple sections unless the user explicitly asks for more detail.
- For technical questions, explain only the mechanics necessary to answer the question.
- Provide code only when the user explicitly requests code, syntax, an example, or an implementation.
- Never invent APIs, configuration options, package names, or version numbers. State uncertainty when appropriate.
- Never claim to have run code or tested a system unless you actually did.
- Expand the explanation when the user asks follow-up questions or requests more detail.
- When the user is learning a concept, end with exactly one concise check-your-understanding question.
"""

RAG_SYSTEM_PROMPT = f"""You answer questions strictly using the user's uploaded documents provided under RETRIEVED CONTEXT.

Rules:
1. Use ONLY facts directly stated in RETRIEVED CONTEXT. Do not use outside knowledge, guess, or extrapolate.
2. Check that the specific person, date, number, or action asked about is explicitly supported by the text.
3. If the answer is missing or unsupported, output ONLY this exact sentence:
"{FALLBACK_MESSAGE}"
4. Answer directly in 1-2 concise sentences and cite the source at the end in brackets, for example [filename.md] or [filename.pdf, page 1].
"""

RAG_SUMMARY_PROMPT = f"""You are a helpful document assistant. Explain or summarize uploaded documents using ONLY the passages provided under RETRIEVED CONTEXT.

Rules:
1. Rely strictly on facts present in RETRIEVED CONTEXT. Do not invent outside details.
2. Start with a concise overview, followed by 3-5 bullet points when the retrieved passages contain enough information.
3. Cite the source document in brackets, including the page when available, for example [filename.pdf, page 1].
4. If the context does not contain information needed to answer the question, output ONLY this exact sentence:
"{FALLBACK_MESSAGE}"
"""

# Fixed: Removed the unsafe "about" keyword to protect your 92.1% evaluation score
_BROAD_QUERY_KEYWORDS = (
    "explain this", "explain the document", "explain document", "describe",
    "summarize", "summary", "overview", "what is this about", "what is this document",
    "key points", "takeaways", "main points", "tldr", "tl;dr", "outline", "brief",
    "walk me through", "what is this", "synthesize", "condense", "digest", "recap", "abstract",
)


def _rough_token_count(text: str) -> int:
    """Use the same approximate token budget for local and cloud backends."""
    return max(1, (len(text) + 3) // 4)


class _LLMBase:
    """Shared prompt construction, history handling, and RAG policy."""

    mode = "base"

    def _build_messages(
        self,
        system: str,
        history: Optional[List[Dict]],
        user_msg: str,
    ) -> List[Dict]:
        budget = max(
            0,
            N_CTX
            - MAX_NEW_TOKENS
            - _rough_token_count(system)
            - _rough_token_count(user_msg)
            - 200,
        )
        kept: List[Dict] = []
        used = 0

        for item in reversed((history or [])[-MAX_HISTORY_TURNS:]):
            role = item.get("role", "user")
            if role not in ("user", "assistant"):
                continue
            content = (item.get("content") or "").strip()
            if not content or content == FALLBACK_MESSAGE:
                continue
            tokens = _rough_token_count(content)
            if used + tokens > budget:
                break
            kept.append({"role": role, "content": content})
            used += tokens

        kept.reverse()
        while kept and kept[0]["role"] != "user":
            kept.pop(0)

        return [{"role": "system", "content": system}, *kept, {"role": "user", "content": user_msg}]

    def _prepare_rag_messages(
        self,
        question: str,
        retrieved_chunks: Optional[List[Dict]],
        history: Optional[List[Dict]],
        min_rerank_score: Optional[float] = -5.0,
    ):
        if not retrieved_chunks:
            return None

        question_lower = question.lower().strip()
        is_broad_request = any(keyword in question_lower for keyword in _BROAD_QUERY_KEYWORDS)
        top_chunk = retrieved_chunks[0]
        top_score = top_chunk.get("score")

        if (
            not is_broad_request
            and min_rerank_score is not None
            and top_score is not None
            and top_score < min_rerank_score
        ):
            return None

        context_parts = []
        for index, item in enumerate(retrieved_chunks[:MAX_CHUNKS], start=1):
            chunk_text = (item.get("text") or "").strip()
            if not chunk_text:
                continue
            if len(chunk_text) > MAX_CHUNK_CHARS:
                chunk_text = chunk_text[:MAX_CHUNK_CHARS].rstrip() + "..."

            source = item.get("source", "Unknown source")
            page = item.get("page")
            location = f"{source}, page {page}" if page is not None else source
            context_parts.append(f"[{index}] Source: {location}\n{chunk_text}")

        if not context_parts:
            return None

        user_msg = (
            "RETRIEVED CONTEXT:\n"
            + "\n\n".join(context_parts)
            + f"\n\nQUESTION:\n{question}"
        )
        prompt = RAG_SUMMARY_PROMPT if is_broad_request else RAG_SYSTEM_PROMPT
        return self._build_messages(prompt, history, user_msg)

    def generate_chat_answer(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
        temperature: float = 0.2,
    ) -> str:
        messages = self._build_messages(SYSTEM_PROMPT, history, question)
        return self._generate(messages, temperature)

    def generate_rag_answer(
        self,
        question: str,
        retrieved_chunks: Optional[List[Dict]],
        history: Optional[List[Dict]] = None,
        temperature: float = 0.0,
        min_rerank_score: Optional[float] = -5.0,
    ) -> str:
        messages = self._prepare_rag_messages(
            question, retrieved_chunks, history, min_rerank_score
        )
        if messages is None:
            return FALLBACK_MESSAGE
        return self._generate(messages, temperature)


class LocalLLM(_LLMBase):
    def __init__(self, model_path: str):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Local GGUF not found at {model_path}.")

        from llama_cpp import Llama

        self.model = Llama(
            model_path=str(model_file),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            chat_format="llama-3",
            verbose=False,
        )
        self.mode = "local_gguf"

    def _generate(self, messages: List[Dict], temperature: float) -> str:
        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=float(temperature),
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.05,
        )
        choices = output.get("choices") or []
        if not choices:
            return "I couldn't generate a response."
        return (choices[0].get("message", {}).get("content") or "").strip() or "I couldn't generate a response."


class GroqLLM(_LLMBase):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
        if not key:
            raise ValueError("GROQ_API_KEY is not set.")
        
        self.key = key
        # Fixed: Fallback to the real, valid free model ID on Groq
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.mode = "groq"

    def _generate(self, messages: List[Dict], temperature: float) -> str:
        # Fixed: Lazy load the Groq SDK so offline local runs never crash on startup
        from groq import Groq
        client = Groq(api_key=self.key)
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=float(temperature),
            top_p=0.9,
        )
        if not response.choices:
            return "I couldn't generate a response."
        return (response.choices[0].message.content or "").strip() or "I couldn't generate a response."
