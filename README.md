# 🤖 Local AI Knowledge Assistant

A privacy-focused local AI assistant that supports both:

1. **General AI chat** without uploading documents.
2. **Optional RAG-based document chat** using user-provided PDF, DOCX, TXT, or Markdown files.

Built to explore what a fully local RAG pipeline actually requires end-to-end — ingestion, hybrid retrieval, grounded generation, and a UI that doesn't feel like a debugging console — without depending on a hosted LLM API.

The application runs a local **Mistral 7B Instruct GGUF model** and uses semantic + keyword retrieval for document-grounded answers.


## Architecture

```text
                         User
                          |
                          v
                    Streamlit UI
                          |
              +-----------+-----------+
              |                       |
         No documents            Documents uploaded
              |                       |
              v                       v
       Direct Mistral          Document ingestion
                                      |
                                      v
                                  Chunking
                                      |
                           +----------+----------+
                           |                     |
                           v                     v
                     SentenceTransformer      BM25
                           |                     |
                           v                     |
                         FAISS                  |
                           |                     |
                           +----------+----------+
                                      |
                                Hybrid retrieval
                                      |
                                      v
                              Retrieved context
                                      |
                                      v
                                  Mistral 7B
                                      |
                                      v
                           Grounded final answer
                                      |
                                      v
                              Source references
```

## Features

- Local Mistral 7B inference
- General chat without document upload
- Optional PDF, DOCX, TXT, and Markdown ingestion
- Semantic search with FAISS
- Keyword search with BM25
- Hybrid retrieval using reciprocal rank fusion
- Source and page metadata
- Conversation history
- Configurable temperature
- Configurable retrieval Top-K
- Basic hallucination guardrails via system prompt
- No cloud LLM API required

## Project Structure

```text
local-ai-knowledge-assistant/
├── app.py
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── llm.py
│   └── retriever.py
├── models/
│   └── mistral-7b-instruct-v0.1-q4_k_m.gguf
├── data/
├── evaluation/
│   └── questions.json
├── requirements.txt
└── README.md
```

## 1. Create environment

### Conda

```bash
conda create -n local-ai python=3.11 -y
conda activate local-ai
```

Or with venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Apple Silicon note

`llama-cpp-python` can require a platform-specific build. If the standard installation does not provide the acceleration you want, install it using the official llama.cpp Python package instructions for your machine and then install the remaining requirements.

## 3. Add the Mistral model

Place your existing model here:

```text
models/mistral-7b-instruct-v0.1-q4_k_m.gguf
```

The model is intentionally ignored by Git because GGUF files can be several GB.

## 4. Run

```bash
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

A full 7B GGUF model is too large/slow for Streamlit Cloud's default resources, so a cloud deployment of this repo is meant as a **UI/UX and code-quality demo**, not a full-speed inference environment. Two practical options:

1. **Smaller quantized model** — swap `MODEL_PATH` to a smaller GGUF (e.g. a 1–3B instruct model) that fits within Streamlit Cloud's memory/CPU limits, at the cost of answer quality.
2. **Demo mode** — point `MODEL_PATH` at nothing and let the app show its "Model not found" state, or stub `LocalLLM` behind an environment flag so reviewers can still exercise the UI (upload, chat layout, source cards) without a real model loaded.

Steps:

1. Push this repo to GitHub (do **not** commit the GGUF file — keep it in `.gitignore`).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app.py`.
3. Add `requirements.txt` as-is; add any `packages.txt` entries `llama-cpp-python` needs for a CPU build on Streamlit's Linux runners.
4. If using a smaller model, either commit it via Git LFS or download it at startup from a URL in `src/config.py`.

## How to use

### General chat

Start the app and ask:

```text
What is retrieval augmented generation?
```

No documents are required.

### Document chat

Upload one or more:

```text
PDF
DOCX
TXT
MD
```

Then ask questions about the uploaded content.

Example:

```text
Summarize this document.

What are the main requirements?

What deadline is mentioned?

Explain section 3 in simple terms.
```

## Important behavior

If documents are uploaded, the application uses retrieved document context for document-aware answers.

If the answer cannot be found in the retrieved context, the system prompt instructs the model to say so rather than invent document-specific facts.

## Known limitations

- **Stale document context after removal.** Removing an uploaded document clears the active retriever/knowledge base immediately, so no further retrieval happens against it. However, if a prior answer generated *while the document was indexed* is still sitting in the conversation history, the model can still reference facts from that earlier answer in later turns of the same session (this is ordinary conversational memory, not retrieval). **Workaround:** start a new conversation after removing a document. **Planned fix:** prune document-grounded messages from history when the knowledge base is cleared.
- No persistent vector store — the index is rebuilt in memory each session and is lost on refresh/restart.
- No reranking step after hybrid retrieval; result ordering relies solely on reciprocal rank fusion.
- No streaming token output — responses render only once generation completes.
- Scanned/image-only PDFs are not OCR'd, so their text will not be retrievable.
- No automated tests yet for ingestion or retrieval correctness.

## Portfolio improvements

Potential future extensions:

- Cross-session persistent vector databases
- Better reranking models
- Streaming token output
- OCR for scanned PDFs
- Table extraction
- Web search tool
- More document formats
- Retrieval evaluation dashboard
- Faithfulness and answer-relevance evaluation
- User authentication
- Conversation export
- Multiple local models
- Quantization/performance comparison

## Security / privacy

The application is designed around local inference. Uploaded documents are processed by the application and are not intentionally sent to a hosted LLM API.

Do not place confidential files into the project repository.

## Evaluation (planned)

The project includes a starting point for retrieval/answer evaluation, not yet a completed benchmark:

```text
evaluation/questions.json
```

Planned metrics once a question set is populated:

- Retrieval hit rate
- Precision@K
- Recall@K
- Answer relevance
- Faithfulness
- Response latency
- Tokens per second

## License

MIT — see `LICENSE` for details.

