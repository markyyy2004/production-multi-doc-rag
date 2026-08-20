import os
import io
import re
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT & SECRETS SYNCHRONIZATION (PYTEST SAFE)
# -----------------------------------------------------------------------------
load_dotenv()

groq_api_key = ""
try:
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
        groq_api_key = str(st.secrets["GROQ_API_KEY"]).strip()
        os.environ["GROQ_API_KEY"] = groq_api_key
except Exception:
    pass

if not groq_api_key:
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document as LC_Doc
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

import fitz  # PyMuPDF
from docx import Document as DocxReader
from pptx import Presentation
from rapidocr_onnxruntime import RapidOCR
from rank_bm25 import BM25Okapi
from PIL import Image
import numpy as np

# -----------------------------------------------------------------------------
# 2. PERSISTENT STORAGE (SQLite)
# -----------------------------------------------------------------------------
DB_FILE = "users_history.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(username) REFERENCES users(username)
            )
        """)
        conn.commit()

init_db()

# -----------------------------------------------------------------------------
# 3. EXTRACTION & HYBRID RETRIEVAL ENGINES
# -----------------------------------------------------------------------------
ocr_engine = RapidOCR()

def safe_ocr_extract(img_array):
    """Safely extracts text from an image array via RapidOCR."""
    try:
        ocr_result, _ = ocr_engine(img_array)
        if ocr_result:
            return "\n".join([line[1] for line in ocr_result if line[2] >= 0.50])
    except Exception:
        pass
    return ""

def parse_pdf_page(page_data, source_name="Uploaded PDF"):
    page_num, page_layout = page_data
    text = page_layout.get_text().strip()
    if not text:
        try:
            pix = page_layout.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            extracted_ocr = safe_ocr_extract(np.array(img))
            if extracted_ocr:
                text = extracted_ocr
        except Exception:
            pass
    if text:
        return LC_Doc(page_content=text, metadata={"page": page_num + 1, "source": source_name})
    return None

def extract_pdf_parallel(file_bytes, source_name="Uploaded PDF"):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_to_process = [(i, doc[i]) for i in range(len(doc))]
        docs = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(lambda p: parse_pdf_page(p, source_name), pages_to_process)
        for res in results:
            if res is not None:
                docs.append(res)
        return docs
    except Exception:
        return []

def extract_text_from_file(file_input, file_name="", progress_bar=None, status_text=None):
    """
    Unified extractor returning LangChain Document objects with .page_content and .metadata.
    Supports both file paths and byte streams.
    """
    if isinstance(file_input, str):
        if not os.path.exists(file_input) or os.path.getsize(file_input) == 0:
            return []
        with open(file_input, "rb") as f:
            file_bytes = f.read()
        target_name = file_name if file_name else os.path.basename(file_input)
    else:
        file_bytes = file_input
        target_name = file_name

    if not file_bytes:
        return []

    ext = target_name.split(".")[-1].lower() if target_name else ""
    docs = []

    try:
        if ext == "pdf":
            docs = extract_pdf_parallel(file_bytes, source_name=target_name)
        elif ext == "docx":
            doc = DocxReader(io.BytesIO(file_bytes))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            if text.strip():
                docs.append(LC_Doc(page_content=text.strip(), metadata={"page": 1, "source": target_name}))
        elif ext == "pptx":
            prs = Presentation(io.BytesIO(file_bytes))
            for idx, slide in enumerate(prs.slides):
                slide_text = ""
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text and shape.text.strip():
                        slide_text += shape.text.strip() + "\n"
                if slide_text.strip():
                    docs.append(LC_Doc(
                        page_content=slide_text.strip(),
                        metadata={"page": idx + 1, "slide": idx + 1, "source": target_name}
                    ))
        else:
            try:
                decoded_text = file_bytes.decode("utf-8")
                if decoded_text.strip():
                    docs.append(LC_Doc(page_content=decoded_text.strip(), metadata={"page": 1, "source": target_name}))
            except Exception:
                pass
    except Exception:
        return []

    if progress_bar and hasattr(progress_bar, "progress"):
        progress_bar.progress(1.0)
    if status_text and hasattr(status_text, "text"):
        status_text.text("Extraction complete")

    return docs

# Test compatibility alias
extract_docs_from_file = extract_text_from_file

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Test compatibility alias
load_embeddings = get_embedding_model

class HybridRetriever:
    def __init__(self, raw_input=None, vector_store=None, all_chunks=None):
        self.chunked_metadatas = []
        self.chunked_texts = []
        self.vector_store = vector_store
        
        candidate_chunks = []
        if all_chunks is not None:
            candidate_chunks = all_chunks
        elif isinstance(raw_input, list):
            candidate_chunks = raw_input

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        
        if candidate_chunks:
            for item in candidate_chunks:
                if isinstance(item, LC_Doc):
                    splits = text_splitter.split_text(item.page_content)
                    for split in splits:
                        if split.strip():
                            self.chunked_texts.append(split)
                            self.chunked_metadatas.append(dict(item.metadata))
                elif isinstance(item, dict):
                    splits = text_splitter.split_text(item.get("text", ""))
                    for split in splits:
                        if split.strip():
                            self.chunked_texts.append(split)
                            self.chunked_metadatas.append({
                                "page": item.get("page", 1),
                                "slide": item.get("slide", item.get("page", 1)),
                                "source": item.get("source", "Uploaded Document")
                            })
                            
        if self.vector_store is None and self.chunked_texts:
            embeddings = get_embedding_model()
            self.vector_store = FAISS.from_texts(self.chunked_texts, embeddings, metadatas=self.chunked_metadatas)

        tokenized_corpus = [t.lower().split() for t in self.chunked_texts] if self.chunked_texts else [["empty"]]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 4):
        if not self.chunked_texts or self.vector_store is None:
            return []
            
        dense_results = self.vector_store.similarity_search_with_score(query, k=min(top_k * 2, len(self.chunked_texts)))
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        rrf_scores = {}
        for rank, (doc, _) in enumerate(dense_results):
            txt = doc.page_content
            if txt not in rrf_scores:
                rrf_scores[txt] = {"doc": doc, "score": 0.0}
            rrf_scores[txt]["score"] += 1.0 / (60.0 + (rank + 1))
            
        for rank, idx in enumerate(top_bm25_indices):
            if idx < len(self.chunked_texts):
                txt = self.chunked_texts[idx]
                if txt not in rrf_scores:
                    constructed_doc = LC_Doc(page_content=txt, metadata=self.chunked_metadatas[idx])
                    rrf_scores[txt] = {"doc": constructed_doc, "score": 0.0}
                rrf_scores[txt]["score"] += 1.0 / (60.0 + (rank + 1))
            
        sorted_rrf = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_rrf[:top_k]]

    def invoke(self, query: str):
        return self.search(query)

def create_hybrid_retriever(vector_store_or_docs, all_chunks=None):
    if isinstance(vector_store_or_docs, list) and all_chunks is None:
        return HybridRetriever(raw_input=vector_store_or_docs)
    return HybridRetriever(raw_input=None, vector_store=vector_store_or_docs, all_chunks=all_chunks)

def clean_response_markdown(text: str) -> str:
    text = re.sub(r'(?i)<br\s*/?>\s*•?', '\n\n* ', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text

# -----------------------------------------------------------------------------
# 4. STREAMLIT APP CONFIGURATION & TARGETED THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nexus Knowledge Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #06090e;
    color: #c9d1d9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #0b0f17 !important;
    border-right: 1px solid #1a2233;
}

/* Centered Welcome Hero Screen */
.welcome-pill {
    display: inline-block;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #38bdf8;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.8px;
    padding: 5px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
    text-transform: uppercase;
}

.welcome-title {
    color: #ffffff;
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
}

.welcome-subtitle {
    color: #8b949e;
    font-size: 0.95rem;
    max-width: 650px;
    margin: 0 auto 24px auto;
    line-height: 1.5;
}

.feature-card {
    background-color: #0b111e;
    border: 1px solid #1c2638;
    border-radius: 12px;
    padding: 24px;
    max-width: 680px;
    margin: 0 auto 28px auto;
    text-align: left;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.feature-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 16px;
}
.feature-row:last-child {
    margin-bottom: 0;
}
.feature-title {
    color: #f0f6fc;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 2px;
}
.feature-desc {
    color: #8b949e;
    font-size: 0.85rem;
    line-height: 1.4;
}

/* Message Containers */
.chat-container-user {
    background-color: #0b111e;
    border: 1px solid #1c2638;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.chat-container-ai {
    background-color: #0b111e;
    border: 1px solid #1c2638;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 18px;
    line-height: 1.6;
}

/* Markdown Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    border: 1px solid #212c3d;
    border-radius: 6px;
    overflow: hidden;
}
th {
    background-color: #121927 !important;
    color: #38bdf8 !important;
    padding: 10px 14px;
    border: 1px solid #212c3d;
    text-align: left;
    font-weight: 600;
}
td {
    background-color: #0b111e !important;
    color: #c9d1d9 !important;
    padding: 10px 14px;
    border: 1px solid #1c2638;
    vertical-align: top;
}
tr:nth-child(even) td {
    background-color: #0e1524 !important;
}

code {
    color: #38bdf8 !important;
    background-color: #162032 !important;
    border: 1px solid #25334c !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-size: 0.88em !important;
}

.stButton>button {
    background-color: #121927;
    color: #c9d1d9;
    border: 1px solid #212c3d;
    border-radius: 6px;
    font-weight: 500;
    transition: all 0.2s ease;
}
.stButton>button:hover {
    background-color: #1c2638;
    border-color: #38bdf8;
    color: #ffffff;
}

.status-badge-active {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #4ade80;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}
.status-badge-inactive {
    background: rgba(148, 163, 184, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: #94a3b8;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = []
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "inspect_modal_content" not in st.session_state:
    st.session_state.inspect_modal_content = None

# --- AUTHENTICATION MODAL ---
if not st.session_state.authenticated:
    st.markdown("""
<div style="text-align: center; margin-top: 50px; margin-bottom: 20px;">
    <h1 style="color: #ffffff; font-size: 2.3rem; font-weight: 800;">⚡ Nexus Knowledge Copilot</h1>
    <p style="color: #8b949e; font-size: 0.95rem;">Context-aware, multi-format conversational intelligence</p>
</div>
""", unsafe_allow_html=True)
    
    col_l, col_center, col_r = st.columns([1, 1.2, 1])
    with col_center:
        with st.container(border=True):
            tab_login, tab_register = st.tabs(["🔒 Secure Login", "📝 Create Account"])
            
            with tab_login:
                li_user = st.text_input("Username", key="li_user")
                li_pass = st.text_input("Password", type="password", key="li_pass")
                if st.button("Authenticate Session", use_container_width=True):
                    h = hashlib.sha256(li_pass.encode()).hexdigest()
                    with get_db_connection() as conn:
                        user_row = conn.execute("SELECT * FROM users WHERE username = ? AND password_hash = ?", (li_user, h)).fetchone()
                    if user_row:
                        st.session_state.authenticated = True
                        st.session_state.username = li_user
                        st.rerun()
                    else:
                        st.error("Invalid credentials provided.")
                        
            with tab_register:
                reg_user = st.text_input("New Username", key="reg_user")
                reg_pass = st.text_input("New Password", type="password", key="reg_pass")
                if st.button("Register Identity", use_container_width=True):
                    if len(reg_user.strip()) < 3 or len(reg_pass) < 4:
                        st.warning("Username >= 3 chars, Password >= 4 chars.")
                    else:
                        h = hashlib.sha256(reg_pass.encode()).hexdigest()
                        try:
                            with get_db_connection() as conn:
                                conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (reg_user, h))
                                conn.commit()
                            st.success("Account registered! Switch to Login tab to proceed.")
                        except sqlite3.IntegrityError:
                            st.error("Username is already taken.")
    st.stop()

# --- ACTIVE LOGGED-IN SESSION ---
db_conn = get_db_connection()

with st.sidebar:
    st.markdown(f"👤 **Logged in as:** <code style='color:#38bdf8;'>{st.session_state.username}</code>", unsafe_allow_html=True)
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.retriever = None
        st.rerun()
        
    st.divider()
    st.markdown("### ⚡ NEXUS ENGINE")
    st.caption("Universal Multi-Format RAG Architecture")
    
    st.markdown("#### 📊 Hybrid Index Status")
    if st.session_state.retriever:
        st.markdown("""
<div class="status-badge-active">
    <span>🟢</span> Hybrid Index Active (BM25 + FAISS)
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="status-badge-inactive">
    <span>⚪</span> No Documents Indexed
</div>
""", unsafe_allow_html=True)
        
    st.divider()
    st.markdown("### 📂 Document Hub")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "txt", "py", "md"],
        help="200MB per file • PDF, DOCX, PPTX, TXT, PY, MD"
    )
    
    if st.button("⚡ Process & Index Documents", use_container_width=True) and uploaded_files:
        parsed_intelligence = []
        with st.spinner("Executing Parallel Document Deconstruction..."):
            for f in uploaded_files:
                file_bytes = f.read()
                extracted_docs = extract_text_from_file(file_bytes, f.name)
                for doc in extracted_docs:
                    parsed_intelligence.append(doc)
                    
        if parsed_intelligence:
            st.session_state.raw_docs = parsed_intelligence
            st.session_state.retriever = HybridRetriever(parsed_intelligence)
            st.success(f"Indexed {len(uploaded_files)} file(s) successfully!")
            st.rerun()
        else:
            st.error("No extractable textual content found.")

    if st.session_state.raw_docs:
        st.divider()
        st.markdown("### 📑 Indexed Knowledge")
        with st.expander("🔍 View Ingested Content", expanded=False):
            doc_sources = list(set([d.metadata.get("source", "Document") for d in st.session_state.raw_docs]))
            selected_source = st.selectbox("Select File", doc_sources)
            selected_pages = [d for d in st.session_state.raw_docs if d.metadata.get("source") == selected_source]
            for p in selected_pages:
                st.markdown(f"**Page {p.metadata.get('page', 1)}**")
                st.text_area(f"Content_p{p.metadata.get('page', 1)}", p.page_content[:400] + "...", height=90, disabled=True, label_visibility="collapsed")
            
    st.divider()
    st.markdown("### ⚙️ Operations")
    col_cc, col_wdb = st.columns(2)
    with col_cc:
        if st.button("🧹 Clear Chat", use_container_width=True):
            db_conn.execute("DELETE FROM chat_history WHERE username = ?", (st.session_state.username,))
            db_conn.commit()
            st.session_state.last_citations = []
            st.session_state.inspect_modal_content = None
            st.rerun()
    with col_wdb:
        if st.button("🗑️ Wipe DB", use_container_width=True):
            db_conn.execute("DELETE FROM chat_history")
            db_conn.execute("DELETE FROM users")
            db_conn.commit()
            st.session_state.authenticated = False
            st.session_state.last_citations = []
            st.session_state.inspect_modal_content = None
            st.rerun()

# --- MAIN CHAT & HERO WELCOME VIEWPORT ---
chat_history_rows = db_conn.execute(
    "SELECT role, message FROM chat_history WHERE username = ? ORDER BY timestamp ASC",
    (st.session_state.username,)
).fetchall()

preset_clicked = None

if not chat_history_rows:
    st.markdown("""
<div style="text-align: center; margin-top: 35px;">
    <div class="welcome-pill">⚡ NEXT-GEN AI RAG ENGINE</div>
    <div class="welcome-title">Nexus Knowledge Copilot</div>
    <div class="welcome-subtitle">
        Context-aware, multi-format conversational intelligence powered by Groq LLaMA, FAISS vector indexing, and RapidOCR.
    </div>
    <div class="feature-card">
        <div class="feature-row">
            <span style="font-size: 1.3rem;">📄</span>
            <div>
                <div class="feature-title">Multi-Document Context Synthesis</div>
                <div class="feature-desc">Upload lecture notes, code files, slides, and scanned diagrams simultaneously.</div>
            </div>
        </div>
        <div class="feature-row">
            <span style="font-size: 1.3rem;">🎯</span>
            <div>
                <div class="feature-title">Granular Source Attribution</div>
                <div class="feature-desc">Trace answers back to exact page numbers, slide indexes, and raw extracted snippets.</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("💡 Summarize Core Concepts", use_container_width=True):
            preset_clicked = "Summarize core concepts and fundamental architecture covered across the notes."
    with col_p2:
        if st.button("⚔️ Compare Key Differences", use_container_width=True):
            preset_clicked = "Identify the key components and compare operational structural differences."
    with col_p3:
        if st.button("📋 Generate Practice Quiz", use_container_width=True):
            preset_clicked = "Generate a comprehensive practice quiz with multiple questions and answers based on the notes."
else:
    for msg in chat_history_rows:
        if msg["role"] == "user":
            st.markdown(f"""
<div class="chat-container-user">
    <span style="color:#a855f7; font-size:1.1rem;">👤</span>
    <span style="color:#f0f6fc; font-size:0.95rem;">{msg['message']}</span>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="chat-container-ai">
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
        <span style="color:#f59e0b; font-size:1.1rem;">⚡</span>
        <span style="font-weight:700; color:#f0f6fc; font-size:1.05rem;">Core concepts & architecture covered</span>
    </div>
    <div>{clean_response_markdown(msg['message'])}</div>
</div>
""", unsafe_allow_html=True)
            
    if st.session_state.last_citations:
        with st.expander("🔎 Verified Document Citations", expanded=True):
            for idx, cit in enumerate(st.session_state.last_citations):
                col_c_text, col_c_btn = st.columns([4.5, 1])
                with col_c_text:
                    st.markdown(f"**Source {idx+1}:** <code style='color:#38bdf8;'>{cit['source']}</code> **(Page {cit['page']})**", unsafe_allow_html=True)
                    st.caption(f"{cit['snippet'][:180]}...")
                with col_c_btn:
                    if st.button(f"👁️ Inspect", key=f"inspect_btn_{idx}", use_container_width=True):
                        st.session_state.inspect_modal_content = cit
                        st.rerun()

    if st.session_state.inspect_modal_content:
        with st.container(border=True):
            c_mod = st.session_state.inspect_modal_content
            col_m_head, col_m_close = st.columns([5, 1])
            with col_m_head:
                st.markdown(f"### 📄 Source Inspector: <code style='color:#38bdf8;'>{c_mod['source']}</code> **(Page {c_mod['page']})**", unsafe_allow_html=True)
            with col_m_close:
                if st.button("✖ Close", key="close_inspect", use_container_width=True):
                    st.session_state.inspect_modal_content = None
                    st.rerun()
            st.text_area("Full Retrieved Context Chunk", c_mod["snippet"], height=160, disabled=True)
            
    st.divider()
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("💡 Summarize Core Concepts", use_container_width=True):
            preset_clicked = "Summarize core concepts and fundamental architecture covered across the notes."
    with col_p2:
        if st.button("⚔️ Compare Key Differences", use_container_width=True):
            preset_clicked = "Identify the key components and compare operational structural differences."
    with col_p3:
        if st.button("📋 Generate Practice Quiz", use_container_width=True):
            preset_clicked = "Generate a comprehensive practice quiz with multiple questions and answers based on the notes."

user_query = st.chat_input("Ask anything about your documents, code, or diagrams...")
if preset_clicked:
    user_query = preset_clicked

# -----------------------------------------------------------------------------
# 5. RETRIEVAL & INFERENCE PIPELINE
# -----------------------------------------------------------------------------
if user_query:
    db_conn.execute(
        "INSERT INTO chat_history (username, role, message) VALUES (?, ?, ?)",
        (st.session_state.username, "user", user_query)
    )
    db_conn.commit()
    
    context_str = ""
    raw_citations = []
    
    if st.session_state.retriever:
        extracted_chunks = st.session_state.retriever.search(user_query, top_k=4)
        for idx, doc in enumerate(extracted_chunks):
            context_str += f"\n[Context Chunk {idx+1}]: {doc.page_content}\n"
            raw_citations.append({
                "source": doc.metadata.get("source", "Uploaded File"),
                "page": doc.metadata.get("page", 1),
                "snippet": doc.page_content
            })
            
    if not context_str:
        context_str = "No specific reference documents indexed. Rely on general foundational knowledge."

    st.session_state.last_citations = raw_citations

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise, elite enterprise technical RAG copilot.\n"
            "Analyze the provided document contexts to construct your structural answers.\n\n"
            "CRITICAL TABLE & FORMATTING RULES:\n"
            "1. When constructing Markdown tables, EVERY ROW must be on a single continuous line. Never insert line breaks or bullet points inside a table cell.\n"
            "2. Separate multiple items inside a table cell using semicolons (;) or commas, never raw newlines.\n"
            "3. Format technical keywords and code with inline code blocks (e.g. `int`, `return`, `cout`, `app.jar`, `ifstream`, `try`, `catch`, `int &ref`).\n"
            "4. Never output raw <br> or <br/> tags under any circumstance.\n"
            "5. If a concept is too complex for a single cell, use structured Markdown bullet points instead of a table.\n\n"
            "CONTEXT:\n{context}"
        ),
        ("human", "{question}")
    ])

    if not groq_api_key:
        st.error("🔑 GROQ_API_KEY is not configured! Add it to Streamlit Secrets or your .env file.")
    else:
        try:
            llm = ChatGroq(
                model_name="openai/gpt-oss-20b",
                groq_api_key=groq_api_key,
                temperature=0.2,
                streaming=True
            )
            
            pipeline = prompt | llm | StrOutputParser()
            
            with st.container():
                st.markdown(f"""
<div class="chat-container-user">
    <span style="color:#a855f7; font-size:1.1rem;">👤</span>
    <span style="color:#f0f6fc; font-size:0.95rem;">{user_query}</span>
</div>
""", unsafe_allow_html=True)
                
                response_placeholder = st.empty()
                full_response = ""
                
                for chunk in pipeline.stream({"context": context_str, "question": user_query}):
                    full_response += chunk
                    cleaned_chunk = clean_response_markdown(full_response)
                    response_placeholder.markdown(f"""
<div class="chat-container-ai">
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
        <span style="color:#f59e0b; font-size:1.1rem;">⚡</span>
        <span style="font-weight:700; color:#f0f6fc; font-size:1.05rem;">Core concepts & architecture covered</span>
    </div>
    <div>{cleaned_chunk}</div>
</div>
""", unsafe_allow_html=True)
                            
            db_conn.execute(
                "INSERT INTO chat_history (username, role, message) VALUES (?, ?, ?)",
                (st.session_state.username, "assistant", full_response)
            )
            db_conn.commit()
            st.rerun()

        except Exception as e:
            st.error(f"⚠️ Inference Error: {str(e)}")