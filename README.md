# Hybrid LLM-based IT Helpdesk Chatbot

An intelligent, hybrid AI-powered chatbot designed to automate internal IT support. This system ingests historical helpdesk logs to resolve repetitive user issues locally using open-source Large Language Models (LLMs). When facing unknown or out-of-context queries, it seamlessly falls back to real-time web search capabilities.

## 📌 Features
- **Local LLM Execution:** Uses quantized local models (LLaMA 7B / Mistral 7B) via `llama.cpp` for high performance, data privacy, and zero API costs.
- **Smart RAG (Retrieval-Augmented Generation):** Matches user inquiries against internal historical IT logs using a fast FAISS vector store.
- **Automated Search Fallback:** Integrates SerpAPI logic to fetch real-time web solutions if internal documentation cannot resolve the problem.
- **User-Friendly UI:** Simple, reactive web interface built with Streamlit featuring clear historical chat sessions.

---

## 🏗️ System Architecture

The chatbot operates via a robust pipeline organized as follows:
1. **Data Ingestion:** Loads internal IT helpdesk logs stored in CSV format.
2. **Text Processing & Embedding:** Documents are processed with LangChain's `TextLoader` and divided via `CharacterTextSplitter`. Chunks are converted to dense vectors using HuggingFace `sentence-transformers`.
3. **Vector Database:** High-efficiency vector indexing and similarity searches are performed locally using **FAISS**.
4. **Orchestration Chain:** Uses LangChain's `RetrievalQA` with a `stuff` document strategy to pass the top 3 most relevant data blocks to the model.
5. **Hybrid Response Engine:** Evaluates contexts to route queries either directly to the local LLaMA model or to the external **SerpAPI** engine.

---

## 🛠️ Tools & Technologies Used

- **Language:** Python
- **Data Analysis:** Pandas, Datetime
- **AI & Core NLP Stack:** LangChain, FAISS, HuggingFace Transformers
- **LLM Infrastructure:** LlamaCpp (`Llama 7B` / `Mistral 7B`)
- **API Integrations:** SerpAPI (Google Search API Engine)
- **Web Framework:** Streamlit

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com
cd Group_3_AI_Helpdesk_Chatbot
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed. Run:
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file in your root folder and add your SerpAPI token:
```env
SERPAPI_API_KEY=your_secret_serpapi_key_here
```

### 4. Prepare Your Data & Model
- Place your IT logs CSV file into the designated `data/` folder directory.
- Download your quantized model weights file (e.g., `.gguf` format) and configure its filepath in your configurations.

### 5. Launch the Chatbot
```bash
streamlit run app.py
```

---

## 📊 Model Evaluation
System accuracy and retrieval relevance are rigorously evaluated and benchmarked using the **ROUGE score** metrics framework against known technical documentation ground truths.
