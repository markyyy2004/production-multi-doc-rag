# ⚡ Nexus RAG — Enterprise Multi-Document Knowledge Engine

A production-grade, full-stack **Retrieval-Augmented Generation (RAG)** copilot designed for precise technical document search, OCR ingestion, and conversational question answering. 

Powered by a **Hybrid Search Pipeline (FAISS Dense Vectors + BM25 Lexical Matching)** fused via **Reciprocal Rank Fusion (RRF)**, high-throughput LLM streaming using **Groq (`openai/gpt-oss-20b`)**, and persistent user session storage.

---

## 🌟 Key Features

* **Hybrid Retrieval Architecture (FAISS + BM25):** Combines dense semantic understanding (`sentence-transformers/all-MiniLM-L6-v2`) with exact technical keyword matching (BM25) using Reciprocal Rank Fusion (RRF) to eliminate zero-overlap search failures.
* **Multi-Threaded Document Ingestion:** Uses Python's `ThreadPoolExecutor` (4 worker threads) for parallel page parsing, cutting multi-page document indexing latency by ~65%.
* **RapidOCR Vision Fallback:** Automatically extracts text from scanned pages, diagrams, and low-resolution images with a $\ge 0.50$ confidence threshold filter.
* **Full-Stack Security & Persistence:** SQLite relational storage (`database.py`) with SHA-256 password hashing for multi-user authentication and per-user conversation history.
* **Zero-Hallucination Guardrails:** Input clipping (1,500-char max) to prevent token overflow and deterministic static fallbacks when no relevant context is indexed.
* **Split-Screen Document Inspector:** Interactive UI allowing users to click and inspect source documents, page numbers, and exact text chunks alongside live chat.

---

## 🏗️ Architecture Pipeline


```

[ User Query / Document Upload ]
│
▼
┌──────────────────────────────────────────────┐
│  Ingestion & Parallel Extraction Pipeline    │
│  - PyMuPDF (PDFs) / docx / pptx / raw code   │
│  - RapidOCR (Image diagrams & scans)         │
│  - RecursiveCharacterTextSplitter (500/80)   │
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│           Hybrid Indexing Engine             │
│   ├── Dense Index: FAISS (MiniLM-L6-v2)      │
│   └── Sparse Index: BM25 (Exact terms)       │
│                      │                       │
│     Reciprocal Rank Fusion (RRF Algorithm)   │
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│    Context Injection & Generation Engine     │
│   - Groq API (openai/gpt-oss-20b)            │
│   - Token-by-Token Streaming                 │
│   - Page-Level Source Verification           │
└──────────────────────┬───────────────────────┘
│
▼
┌──────────────────────────────────────────────┐
│    Multi-User SQLite Persistence & UI        │
│   - SHA-256 Authenticated Session Isolation │
│   - Streamlit Glassmorphism Interface        │
└──────────────────────────────────────────────┘

```

---

## 📊 Performance Benchmarks

Profiled with `profile_pipeline.py` on 500 technical chunks:

| Metric | Result |
| :--- | :--- |
| **Embedding Batch Size** | `64` |
| **Indexing Throughput** | **~327 chunks / sec** |
| **Top-4 Vector Search Latency** | **8.78 ms** |
| **Automated Test Coverage** | **7 / 7 Passing (`pytest`)** |

---

## 📂 Project Structure

```text
nexus-rag-engine/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI/CD test runner
├── tests/
│   ├── __init__.py
│   ├── test_extractors.py       # Unit tests for PyMuPDF, DOCX, PPTX & OCR
│   ├── test_retriever.py        # Integration tests for FAISS + BM25 RRF
│   ├── test_pipeline.py         # Mocked end-to-end LLM streaming tests
│   └── test_database.py         # Tests for user authentication and chat logs
├── app.py                       # Core Streamlit application & hybrid RAG engine
├── database.py                  # SQLite schema, user auth, and message history
├── profile_pipeline.py          # Latency & throughput benchmarking script
├── conftest.py                  # Pytest configuration
├── requirements.txt             # Locked production dependencies
├── .gitignore                   # Ignores venv, database, and local indexes
└── README.md

```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/nexus-rag-engine.git
cd nexus-rag-engine

```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here

```

### 5. Run the Automated Test Suite

```bash
python -m pytest -v

```

### 6. Launch the Application

```bash
streamlit run app.py

```

---

## 🧪 Testing & CI/CD

The project includes unit and integration tests covering corrupt file handling, OCR fallbacks, exact keyword retrieval, semantic similarity, and database persistence.

To trigger the test suite manually:

```bash
python -m pytest -v

```

Continuous integration is automated via **GitHub Actions** on every push to `main`.

---

## 🛠️ Tech Stack

* **Frontend / Framework:** Streamlit (Custom Glassmorphism UI)
* **LLM Orchestration:** LangChain (LCEL) & Groq API (`openai/gpt-oss-20b`)
* **Vector Store & Indexing:** FAISS (`faiss-cpu`), BM25 (`rank_bm25`)
* **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
* **Parsing & OCR:** PyMuPDF (`fitz`), `python-docx`, `python-pptx`, `rapidocr-onnxruntime`
* **Database & Security:** SQLite3, SHA-256 Hashing
* **Testing & CI:** Pytest, GitHub Actions

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
