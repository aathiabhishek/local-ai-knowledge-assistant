# 💬 Local AI Knowledge Assistant

### A Two-Stage, Object-Oriented RAG System & Technical Tutor

A privacy-focused, document-aware AI assistant and technical tutor designed to run local language models on consumer hardware.

The system uses an optimized, 4-bit quantized **Llama 3.1 8B Instruct (`Q4_K_M` GGUF)** model through `llama.cpp` for local inference. It implements a custom, framework-free, two-stage hybrid retrieval pipeline combining **dense vector search (FAISS)** and **sparse keyword search (BM25)**, followed by **Reciprocal Rank Fusion (RRF)** and optional cross-encoder reranking.

By grounding responses in retrieved document content, the assistant provides context-aware answers with source attribution while supporting document-based learning and technical Q&A.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/RAG-Hybrid%20Retrieval-6A5ACD?style=for-the-badge" alt="Hybrid RAG" />
  <img src="https://img.shields.io/badge/LLM-Llama%203.1-0467DF?style=for-the-badge" alt="Llama 3.1" />
  <img src="https://img.shields.io/badge/Inference-llama.cpp-2E8B57?style=for-the-badge" alt="llama.cpp" />
  <img src="https://img.shields.io/badge/Search-FAISS%20%2B%20BM25-orange?style=for-the-badge" alt="Hybrid Search" />
</p>

---

> **Privacy note:** Local inference can operate offline once the required model and dependencies are available. Optional Groq inference requires an internet connection and sends prompts to the external API. Deploying the application to Streamlit Community Cloud also means the app runs in a cloud environment rather than entirely on your own machine.

---

## 📌 Table of Contents
- [📊 Benchmark Results (100-Query SQuAD v2 Evaluation)](#-benchmark-results-100-query-squad-v2-evaluation)
- [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
- [✨ Core Engineering & Refactoring Highlights](#-core-engineering--refactoring-highlights)
- [🧰 Technology Stack](#-technology-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Virtual Environment & Installation](#-virtual-environment--installation)
- [🧠 Model Configuration & Cloud Fallback](#-model-configuration--cloud-fallback)
- [🔐 Environment Configuration](#-environment-configuration)
- [▶️ Running the Application](#%EF%B8%8F-running-the-application)
- [🛡️ Security Best Practices](#%EF%B8%8F-security-best-practices)
- [📄 License](#-license)

---

## 📊 Benchmark Results (100-Query SQuAD v2 Evaluation)

The system was evaluated using an automated test harness (`evaluation/evaluate.py`) across an independent validation corpus consisting of **80 technical documents (94 chunks)** and **100 test queries** (85 answerable and 15 adversarial unanswerable queries extracted from the Stanford SQuAD v2 dataset).

### 1. Final Pipeline Performance (`Top-K = 5`, `Relevance Gate = -5.0`)

| Pipeline Stage | Performance Metric | Score / Latency | Engineering Takeaway |
| :--- | :--- | :---: | :--- |
| **Stage 1 Retrieval** | **Context Recall (ROUGE-1)** | **98.8%** | Found ground-truth passages in 84 out of 85 valid queries. |
| **Stage 1 Retrieval** | **Hit Rate @ 5** | **98.8%** | At least one valid source chunk is captured in the Top-5 pool. |
| **Stage 2 Reranking** | **Mean Reciprocal Rank (MRR)** | **0.963** | Cross-Encoder isolates and ranks the correct chunk at **Rank #1**. |
| **Search Subsystem** | **Avg Retrieval Latency** | **50.8 ms** | Sub-51ms dense + sparse search, rank fusion, and neural reranking. |
| **LLM Inference** | **Answer Recall (ROUGE-1)** | **92.1%** | Grounded generation accurately captures 92.1% of reference facts. |
| **LLM Inference** | **Answer ROUGE-1 F1** | **25.3%** | Evaluated on full-sentence generation with regex citation-stripping. |
| **LLM Inference** | **Semantic Cosine Similarity** | **44.3%** | High vector alignment against reference target spans. |
| **Inference Subsystem**| **Avg Generation Latency** | **3,695 ms** | Local multi-sentence local decoding at `temperature=0.0`. |
| **Guardrails** | **Adversarial Rejection Rate** | **40.0%** | Rejects grammatically inverted, in-domain adversarial trap queries. |

### 2. Ablation & Prompt Optimization History

| System Version | Retrieval Engine Profile | Context Recall | Reranker MRR | Answer Recall | ROUGE-1 F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **v1 (Baseline)** | Naive space-split BM25 + RRF (27-query seed set) | — | — | 69.4% | — |
| **v2 (Two-Stage RAG)**| Word-boundary BM25 + Cross-Encoder (`gate = -4.5`) | **98.8%** | **0.963** | 90.4% | 21.3% |
| **v3 (Strict Gate)** | 1-Sentence restriction + strict relevance (`gate = -1.5`) | **98.8%** | **0.963** | 85.5% | 31.6% |
| **v4 (Calibrated)** | **Intent-routing prompt + calibrated gate (`gate = -5.0`)** | **98.8%** | **0.963** | **92.1%** | **25.3%** |

---

## 🏗️ System Architecture

```text
                                   User Query
                                       |
                                       v
                                 Streamlit UI
                                       |
                   +-------------------+-------------------+

                   |                                       |
           No Documents Uploaded                   Documents Uploaded

                   |                                       |
                   v                                       v
        General Technical Tutor                 Document Ingestion Engine
      (Llama 3.1 + SYSTEM_PROMPT)        (PyMuPDF sort=True / lxml DOCX Tables)
                                                           |
                                                           v
                                            Boundary-Aware Chunking (900 chars,
                                             150 overlap, word-safe boundaries)
                                                           |
                                           +---------------+---------------+

                                           |                               |
                                           v                               v
                                  Dense Vector Index              Sparse Lexical Index
                              (all-MiniLM-L6-v2, L2 Norm,        (BM25Okapi, regex token
                                   FAISS IndexFlatIP)             cleaning, >0.0 filter)

                                           |                               |
                                           +---------------+---------------+
                                                           |
                                                           v
                                       Stage 1: Reciprocal Rank Fusion (RRF, k=60)
                                              (Top 20 Candidate Oversampling)
                                                           |
                                                           v
                                       Stage 2: Cross-Encoder Neural Reranking
                                         (cross-encoder/ms-marco-MiniLM-L-6-v2)
                                                           |
                                                           v
                                       Intent Routing: Is Broad/Summary Request?
                                           +---------------+---------------+

                                           |                               |
                                       No (QA)                        Yes (Summary)

                                           |                               |
                                           v                               v
                               Calibrated Relevance Gate             Bypass Relevance Gate
                                 (Score < -5.0 Reject)                     |

                                   |               |                       |
                             Yes (Noise)       No (Valid)                  |

                                   |               |                       |
                                   v               +-----------+-----------+
                          Fast Fallback Refusal                |
                         (Short-circuits in 51ms)              v
                                                          Context Formatting
                                                       (Max 5 Chunks, Page Meta)
                                                               |
                                                               v
                                                   Dynamic BPE Token Budgeting
                                                  (N_CTX=8192, Slit-window History)
                                                               |
                                                               v
                                                   Polymorphic _LLMBase Router
                                                   (Local GGUF vs Groq Fallback)
                                                               |
                                                               v
                                                    Grounded Answer + Citations
```

---

## ✨ Core Engineering & Refactoring Highlights

- **Polymorphic Object-Oriented LLM Core (`src/llm.py`):** Refactored the core inference layer using the **Template Method Design Pattern**. Abstracted immutable operations—such as sliding-window conversation history truncation, rough character token-budgeting, and RAG context building—into an uninstantiated parent class `_LLMBase`. Concrete environments are encapsulated via polymorphic runtime sub-classes: `LocalLLM` for accelerated GGUF processing via `llama.cpp` and `GroqLLM` for serverless cloud execution.
- **Lazy Dependency Loading:** Wrapped external vendor imports (e.g., `from groq import Groq`) inside specific execution functions within sub-classes instead of the global scope. This completely decouples local runtime compilation from external infrastructure requirements, guaranteeing that the local offline benchmarking script (`evaluate.py`) can execute in an air-gapped environment without dependency collisions.
• Intent-Aware Adaptive Query Routing: Implemented an intent-classification mechanism (_BROAD_QUERY_KEYWORDS) inside _prepare_rag_messages. Factual questions execute with strict logical grounding rules (RAG_SYSTEM_PROMPT) and pass through the relevance gate. High-level requests (e.g., explain this document, summarize, takeaways) bypass the Cross-Encoder threshold check automatically and trigger a structured bulleted layout template (RAG_SUMMARY_PROMPT) to prevent false-negative rejection errors.
• Empirical Threshold Calibration: Calibrated the Cross-Encoder relevance filter threshold to -5.0 logits by analyzing the score distribution of the evaluation set. Profiling revealed that highly paraphrased, valid context queries scored as low as -4.65, whereas true out-of-domain queries scored below -8.0. Setting the bouncer gate to -5.0 achieves a 100% pass-through rate on valid documents while short-circuiting out-of-domain noise in 50.8ms.
• Custom Framework-Free Document Ingestion Portal (src/ingestion.py):
	• Spatial PDF Coordinate Sort: Parses text blocks utilizing PyMuPDF with sort=True to reconstruct the precise vertical and horizontal reading layout of complex multi-column documents.
	• Document-Order DOCX XML Parser: Iterates sequentially across low-level lxml nodes (<w:p> and <w:tbl>) to capture tables in exact document reading order, formatting text cells into pipe-delimited (|) structures and filtering horizontally merged block fragments.
	• Heuristic Token-Boundary Snapping: Splitting checks the second half (50%–100%) of each 900-character window to safely split on structural boundaries ($\text{Paragraph} \rightarrow \text{Newline} \rightarrow \text{Sentence} \rightarrow \text{Space}$), employing a whitespace regex lookup to shift overlaps forward so words are never cut in half.

🧰 Technology Stack

• Core Runtime: Python 3.11
• User Interface: Streamlit (Custom Dark-Theme UI)
• Local Inference Engine: llama-cpp-python (Metal GPU Accelerated on Apple Silicon)
• Quantized GGUF Model: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
• Cloud Fallback Driver: Groq Python SDK (llama-3.1-8b-instant or custom environment mapping)
• Dense Vector Search: Sentence Transformers (all-MiniLM-L6-v2) & FAISS (IndexFlatIP Inner Product)
• Sparse Keyword Search: Rank-BM25 (BM25Okapi)
• Neural Reranker: Cross-Encoder Transformer (ms-marco-MiniLM-L-6-v2)
• Document Extractors: PyMuPDF (fitz), python-docx (lxml)

## 📁 Project Structure

```text
local-ai-knowledge-assistant/
├── app.py                                  # Streamlit UI and application entry point
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation and portfolio overview
│
├── models/
│   └── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
│                                            # Local quantized GGUF model (excluded from Git)
│
├── src/
│   ├── __init__.py
│   ├── config.py                           # Environment variables and runtime configuration
│   ├── ingestion.py                        # PDF/DOCX ingestion and structural text chunking
│   ├── retriever.py                        # FAISS + BM25, RRF fusion, and cross-encoder reranking
│   └── llm.py                              # Object-oriented LLM abstraction and inference backends
│
└── evaluation/
    ├── build_large_benchmark.py             # Builds the SQuAD v2 benchmark and test harness
    ├── evaluate.py                          # Evaluates retrieval/answer quality and logs metrics
    ├── question.json                        # 100 benchmark queries: 85 answerable, 15 unanswerable
    └── results.json                         # Generated per-query evaluation results
```

### 🔍 Key Components

- **`app.py`** — Provides the Streamlit interface and connects user interactions to the assistant's functionality.
- **`src/ingestion.py`** — Processes supported documents and divides their content into chunks for retrieval.
- **`src/retriever.py`** — Implements dense retrieval with FAISS, sparse retrieval with BM25, Reciprocal Rank Fusion (RRF), and optional cross-encoder reranking.
- **`src/llm.py`** — Organizes model interaction through an object-oriented design, supporting local inference and an optional cloud inference backend.
- **`src/config.py`** — Centralizes configurable settings, including model paths, retrieval parameters, chunk sizes, and inference settings.
- **`evaluation/`** — Contains benchmark construction, evaluation scripts, test questions, and generated results.

**Note:** The GGUF model binary should remain outside version control because of its size. Add it to `.gitignore` and document how to download or configure the model locally.

🚀 Virtual Environment & Installation


1. Initialize Clean Virtual Environment

bash conda create -n rag-env python=3.11 -y conda activate rag-env 
(Or use venv: python3 -m venv .venv && source .venv/bin/activate)

2. Install Project Dependencies

bash pip install --upgrade pip pip install -r requirements.txt 
Note: Your requirements.txt contains clean, unbundled dependencies (faiss-cpu, sentence-transformers, openai, etc.) avoiding framework bloat.

🧠 Model Configuration & Cloud Fallback


Local Deployment

Obtain the model and place it directly into the git-ignored local models block:
text models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf 
On macOS architectures, llama-cpp-python will automatically compile with Metal support, offloading all layers directly to Apple Silicon unified memory (N_GPU_LAYERS=-1).

Cloud Fallback Integration

If the application is booted on hardware lacking a local GGUF file, the system automatically checks for a GROQ_API_KEY token. If configured, the parent router switches to GroqLLM mode, routing inference requests via the cloud configuration mapping.

🔐 Environment Configuration

Create a .env file within your root repository folder:
```dotenv

Model Configuration Paths

MODEL_PATH=models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
GROQ_API_KEY=your_production_groq_api_key_here
CLOUD_MODEL_NAME=openai/gpt-oss-120b

Retextualization & Ranking Models

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

Context Windows Constraints

TOP_K=5
MAX_HISTORY_MESSAGES=8
N_CTX=8192
N_THREADS=8
N_GPU_LAYERS=-1
MAX_NEW_TOKENS=2048

Chunking Splitting Boundaries

CHUNK_SIZE=900
CHUNK_OVERLAP=150
```

▶️ Running the Application


1. Launch UI Interface

bash streamlit run app.py 

2. Execute Automated Evaluation Suite

To regenerate the 100-question corpus and execute the evaluation metrics:
bash python evaluation/build_large_benchmark.py python evaluation/evaluate.py 
