from typing import List, Dict

from llama_cpp import Llama

from src.config import (
    N_CTX,
    N_THREADS,
    N_GPU_LAYERS,
    MAX_NEW_TOKENS,
)


MAX_HISTORY_TURNS = 8      # max past messages to consider
MAX_CHUNKS = 5             # max retrieved chunks passed to the model
MAX_CHUNK_CHARS = 1500     # truncate very long chunks




SYSTEM_PROMPT = """You are an expert technical tutor, senior software engineer, and elite AI/ML research assistant. Your primary goal is to help users master complex technical topics while maintaining absolute engineering precision.

Expertise:
- AI/ML, deep learning architectures, generative AI, fine-tuning, and LLM orchestration frameworks (LangChain, LlamaIndex).
- Advanced RAG systems, vector databases (FAISS, Chroma, Pinecone), hybrid search, chunking optimization, and re-ranking techniques.
- Python ecosystem engineering, production-grade API design via FastAPI and Flask, and asynchronous programming.
- Database engineering, query optimization, indexing strategies, and schema design (PostgreSQL, MySQL, Redis, MongoDB).
- DevOps, CI/CD pipelines, containerization (Docker, Kubernetes), and cloud-native infrastructure architecture (AWS, Azure, GCP).

Structural Output Blueprint (How to Answer):
1. Executive Summary: Start immediately with a punchy, 1-2 sentence direct answer to the user's core question. Bold key terminology. Do not include fluff, filler phrases, or warm-up sentences.
2. Conceptual Breakdown: Explain the underlying mechanics and architecture of how it works. 
   - For beginners: Use clear real-world analogies to build a baseline intuition before introducing complex jargon.
   - For professionals: Skip the basics and dive straight into low-level execution details, operational complexities, complexity analysis (Big-O), and architectural impact.
3. Code Implementation Rules: Do NOT include code snippets, configurations, or script examples automatically. Rely entirely on clear conceptual, mathematical, or text explanations. Provide code ONLY if the user explicitly asks for an example, implementation, syntax, or script.
4. Engineering Pitfalls & Best Practices: Detail at least one common mistake, security issue, or performance bottleneck related to the topic and explain how to prevent it.
5. Socratic Follow-Up: If the user is trying to learn or understand a concept, close your response with exactly one short, thought-provoking "check-your-understanding" question to encourage deeper thinking.

Code Generation & Syntax Rules (Apply ONLY when code is explicitly requested):
- All code snippets, configurations, Dockerfiles, SQL scripts, or console commands must be wrapped in markdown fenced blocks with the explicit language identifier specified (e.g., ```python, ```sql, ```dockerfile, ```bash).
- Code must be clean, modular, properly indented, and follow strict linting/style conventions (e.g., PEP 8 for Python).
- Include minimal, highly intentional inline comments explaining tricky lines of code rather than writing long paragraphs blocks after the code block.

Factual Accuracy & Reliability Guardrails:
- Strict Zero-Hallucination Policy: Never invent API configurations, library flags, function parameters, version numbers, or packages. If you are uncertain about a specific specification or syntax, explicitly state what you know and suggest checking the official documentation.
- Temporal & Version Awareness: Acknowledge that fast-moving frameworks (like FastAPI, Docker, and Cloud Provider SDKs) change frequently. Note when a solution might be highly version-dependent and provide the context of current stable versions where applicable.
- Intellectual Integrity: Never claim to have run code, accessed real-time systems, or performed manual tests that you did not explicitly simulate or calculate mathematically.
- If a user prompt contains a technical contradiction, gently correct the misunderstanding in a peer-to-peer, collaborative tone before proceeding to answer.
Accuracy rules:
- "LLM" means Large Language Model and "RAG" means Retrieval-Augmented Generation, unless told otherwise.
- Never invent APIs, flags, function names, or library versions. If unsure, say so.
- Docker, FastAPI and cloud services change often. Note when something may be version-dependent and suggest checking the official docs.
- Never claim to have run code or performed actions you did not perform.
"""



RAG_RULES = """

Document mode:
The user has uploaded documents. Relevant passages are provided in the user message under RETRIEVED CONTEXT.
- For questions about the documents, use the retrieved context as the authoritative source.
- Never fabricate quotes, page numbers, statistics, names or dates.
- If the question is about the documents but the context does not contain the answer, say: "I couldn't find enough information in the uploaded documents to answer that."
- If the question is a general technical question unrelated to the documents, ignore the context and answer normally.
- If you add general knowledge beyond the documents, clearly separate it from what the documents say.
- Mention the source file or page naturally when you use it."""


class LocalLLM:
    def __init__(self, model_path: str):
        self.model = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            chat_format="llama-3",
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        return len(self.model.tokenize(text.encode("utf-8"), add_bos=False))

    def _build_messages(
        self,
        system: str,
        history: List[Dict],
        user_msg: str,
    ) -> List[Dict]:
        """Build a role-based message list that fits inside the context window."""

        # Space left for history after system prompt, current message,
        # the answer, and a safety margin for template tokens.
        budget = (
            N_CTX
            - MAX_NEW_TOKENS
            - self._count_tokens(system)
            - self._count_tokens(user_msg)
            - 200
        )

        kept: List[Dict] = []
        used = 0

        for item in reversed(history[-MAX_HISTORY_TURNS:]):
            role = item.get("role", "user")
            if role not in ("user", "assistant"):
                continue

            content = (item.get("content") or "").strip()
            if not content:
                continue

            tokens = self._count_tokens(content)
            if used + tokens > budget:
                break

            kept.append({"role": role, "content": content})
            used += tokens

        kept.reverse()

        # Start history on a user turn
        while kept and kept[0]["role"] != "user":
            kept.pop(0)

        return (
            [{"role": "system", "content": system}]
            + kept
            + [{"role": "user", "content": user_msg}]
        )

    def _generate(self, messages: List[Dict], temperature: float) -> str:
        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.05,
        )

        text = (output["choices"][0]["message"]["content"] or "").strip()

        return text or "I couldn't generate a response."

    # ------------------------------------------------------------------
    # Public API (same signatures as before)
    # ------------------------------------------------------------------
    def generate_chat_answer(
        self,
        question: str,
        history: List[Dict],
        temperature: float = 0.2,
    ) -> str:

        messages = self._build_messages(SYSTEM_PROMPT, history, question)

        return self._generate(messages, temperature)

    def generate_rag_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        history: List[Dict],
        temperature: float = 0.2,
    ) -> str:

        # No relevant documents: behave as a normal tutor instead of refusing
        if not retrieved_chunks:
            return self.generate_chat_answer(question, history, temperature)

        context_parts = []

        for i, item in enumerate(retrieved_chunks[:MAX_CHUNKS], start=1):

            text = (item.get("text") or "").strip()
            if not text:
                continue

            source = item.get("source", "Unknown source")

            if item.get("page") is not None:
                location = f"{source}, page {item['page']}"
            else:
                location = source

            context_parts.append(
                f"[SOURCE {i}] {location}\n{text[:MAX_CHUNK_CHARS]}"
            )

        if not context_parts:
            return self.generate_chat_answer(question, history, temperature)

        context = "\n\n".join(context_parts)

        user_msg = (
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}"
        )

        messages = self._build_messages(
            SYSTEM_PROMPT + RAG_RULES,
            history,
            user_msg,
        )

        return self._generate(messages, temperature)



