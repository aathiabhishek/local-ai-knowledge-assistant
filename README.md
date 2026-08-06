# Hybrid LLM-based IT Helpdesk Chatbot

An intelligent, hybrid AI-powered chatbot designed to automate internal IT support. This system ingests historical helpdesk logs to resolve repetitive user issues locally using open-source Large Language Models (LLMs). When facing unknown or out-of-context queries, it seamlessly falls back to real-time web search capabilities.

## 📌 Features
- **Local LLM Execution:** Uses quantized local models (Mistral 7B) via `llama.cpp` for high performance, data privacy, and zero API costs.
- **Smart RAG (Retrieval-Augmented Generation):** Matches user inquiries against internal historical IT logs using a fast FAISS vector store and BM25 retrievers.
- **Automated Search Fallback:** Integrates web logic to fetch real-time solutions if internal documentation cannot resolve the problem.
- **User-Friendly UI:** Simple, reactive web interface built with Streamlit featuring clean chat elements.

---

## 🏗️ System Architecture

The chatbot operates via a robust pipeline organized as follows:
1. **Data Ingestion:** Loads internal IT helpdesk logs stored in CSV format.
2. **Text Processing & Embedding:** Documents are processed with LangChain's loaders and divided via text splitters. Chunks are converted to dense vectors using HuggingFace `sentence-transformers`.
3. **Vector Database:** High-efficiency vector indexing and similarity searches are performed locally using **FAISS** alongside **BM25** ranking metrics.
4. **Orchestration Chain:** Uses LangChain's orchestration handlers to pass the top relevant data blocks to the model.
5. **Hybrid Response Engine:** Evaluates contexts to route queries either directly to the local model or to the external search fallback.

---

## 🛠️ Tools & Technologies Used

- **Language:** Python
- **Data Analysis & Visualization:** Pandas, Matplotlib, Seaborn, Wordcloud
- **AI & Core NLP Stack:** LangChain, FAISS, HuggingFace Transformers, NLTK
- **LLM Infrastructure:** LlamaCpp (`Mistral 7B GGUF`)
- **Web Framework:** Streamlit, Streamlit-Chat

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com
cd Group_3_AI_Helpdesk_Chatbot
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed and your virtual environment activated, then run:
```bash
pip install -r requirements.txt
```

### 3. Model Setup & Download (Choose Option A or Option B)
The chatbot relies on the **Mistral-7B-Instruct-v0.1-GGUF (Q4_K_M)** 4.37 GB model file. Because of GitHub's strict file limits, this file is excluded via `.gitignore` and must be obtained using one of these methods:

#### 🔹 Option A: Automatic Script Download (Recommended)
You do not need to download anything manually. If you use the `huggingface_hub` integration in your python script, the model will download completely for free on your first run:
```python
from huggingface_hub import hf_hub_download

# This automatically downloads and caches the 4GB file locally
local_model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf"
)
```

#### 🔹 Option B: Manual Browser Download
If you prefer downloading via your web browser to store the file inside your project directory manually:
1. Navigate to the model page on Hugging Face: [TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co)
2. Go to the **Files and versions** tab and click download on `mistral-7b-instruct-v0.1.Q4_K_M.gguf`.
3. Create a folder named `models/` inside your project root directory.
4. Move your downloaded `mistral-7b-instruct-v0.1.Q4_K_M.gguf` file into that `models/` folder.
5. In your main interface code file, update your model path link variable:
   ```python
   local_model_path = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
   ```
*(Note: Your `.gitignore` file is fully configured to ignore the `models/` folder so it will not bloat your GitHub uploads).*

### 4. Prepare Your Data
Place your IT logs CSV file inside a designated `data/` folder directory named `IT_helpdesk.csv`.

### 5. Launch the Chatbot
```bash
streamlit run IT_Chatbot_Inteface.py
```

---

## 📊 Model Evaluation
System accuracy and retrieval relevance are rigorously evaluated and benchmarked using the **ROUGE score** metrics framework against known technical documentation ground truths.
