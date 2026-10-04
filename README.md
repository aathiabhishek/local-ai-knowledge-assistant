# Local AI Knowledge Assistant

A privacy-focused local AI assistant that supports both general chat and document-grounded retrieval using a local model.

This project is built to explore a fully local Retrieval-Augmented Generation (RAG) workflow end-to-end, including:
- document ingestion
- chunking
- semantic + keyword retrieval
- grounded answer generation
- a simple Streamlit-based interface

It runs locally and does not require cloud-based LLM APIs for the main experience.

## Features

- General chat without uploaded documents
- Optional document chat using PDF, DOCX, TXT, and Markdown files
- Hybrid retrieval using FAISS + BM25
- Source metadata and page references
- Conversation history
- Local Mistral inference via llama.cpp
- Privacy-first local processing

## Project Structure

```text
local-ai-knowledge-assistant/
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
├── run.sh
├── run.bat
├── .gitignore
├── evaluation/
│   └── questions.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── llm.py
│   └── retriever.py
├── models/
│   └── (your local GGUF model here)
├── data/
│   └── (local document storage, if used)
└── .streamlit/
    └── (local Streamlit config, if created)
```

## Architecture

```text
User
  |
  v
Streamlit UI
  |
  +-- No documents ------------------> Local Mistral chat
  |
  +-- Documents uploaded --> Ingestion --> Chunking --> FAISS + BM25 --> Hybrid retrieval --> Mistral --> Grounded answer
```

## Requirements

- Python 3.10+
- pip
- A compatible local GGUF model file
- System libraries required by llama-cpp-python for your platform

## Setup

### 1) Create a virtual environment

Using venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Or with Conda:

```bash
conda create -n local-ai python=3.11 -y
conda activate local-ai
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Add your local model

Place your GGUF model in the `models/` directory, for example:

```text
models/mistral-7b-instruct-v0.1-q4_k_m.gguf
```

The repo intentionally ignores `models/` and other large/generated files through `.gitignore`.

## Run the app

```bash
streamlit run app.py
```

You can also use the provided launcher scripts:

```bash
./run.sh
```

On Windows:

```bat
run.bat
```

## How to use

### General chat

Open the app and ask technical questions without uploading any documents.

### Document chat

Upload one or more of these file types:
- PDF
- DOCX
- TXT
- Markdown (.md)

Then ask questions about the uploaded material. The app will use the retrieved chunks as context before generating an answer.

## Important behavior

- If documents are uploaded, the app uses retrieved document context for document-aware answers.
- If the answer is not found in the retrieved passages, the model is instructed not to invent document-specific facts.
- The knowledge base is rebuilt in memory for the current session; it is not persisted across restarts.

## Known limitations

- No persistent vector store across sessions
- No reranking pipeline beyond the current hybrid retrieval approach
- No streaming token output
- OCR for scanned PDFs is not included
- No automated evaluation suite yet

## Security and privacy

This project is designed around local inference. Uploaded documents are processed locally and are not intentionally sent to a hosted LLM API.

Do not commit sensitive or confidential files to the repository.

## Maintenance notes

This repository keeps the original simple structure intentionally:
- core app logic in `app.py`
- reusable project logic in `src/`
- local model/data files ignored via `.gitignore`
- evaluation assets kept separate in `evaluation/`

This keeps the project easy to run, easy to review, and easy to maintain without adding unnecessary abstraction.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Contributing

If you want to improve the project:
1. keep the project structure simple
2. avoid adding generated files to git
3. test changes before submitting them
4. keep documentation aligned with the actual code
