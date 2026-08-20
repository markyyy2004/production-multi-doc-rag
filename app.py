import os
import io
import re
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. ENVIRONMENT & SECRETS SYNCHRONIZATION
# -----------------------------------------------------------------------------
load_dotenv()

groq_api_key = ""
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"].strip()
    os.environ["GROQ_API_KEY"] = groq_api_key
elif os.getenv("GROQ_API_KEY"):
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

import fitz  # PyMuPDF
from docx import Document
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

def parse_pdf_page(page_data):
    page_num, page_layout = page_data
    text = page_layout.get_text().strip()
    if not text:
        try:
            pix = page_layout.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            ocr_result, _ = ocr_engine(np.array(img))
            if ocr_result:
                text = "\n".join([line[1] for line in ocr_result if line[2] >= 0.50])
        except Exception:
            pass
    return {"page": page_num + 1, "text": text if text else ""}

def extract_pdf_parallel(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_to_process = [(i, doc[i]) for i in range(len(doc))]
    chunks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(parse_pdf_page, pages_to_process)
    for res in results:
        if res["text"]:
            chunks.append(res)
    return chunks

def extract_text_from_file(file_name, file_bytes):
    ext = file_name.split(".")[-1].lower()
    chunks = []
    if ext == "pdf":
        return extract_pdf_parallel(file_bytes)
    elif ext == "docx":
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if text:
            chunks.append({"page": 1, "text": text})
    elif ext == "pptx":
        prs = Presentation(io.BytesIO(file_bytes))
        for idx, slide in enumerate(prs.slides):
            slide_text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text += shape.text + "\n"
            if slide_text.strip():
                chunks.append({"page": idx + 1, "text": slide_text.strip()})
    else:
        try:
            decoded_text = file_bytes.decode("utf-8")
            if decoded_text.strip():
                chunks.append({"page": 1, "text": decoded_text})
        except Exception:
            pass
    return chunks

@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class HybridRetriever:
    def __init__(self, processed_docs):
        self.docs = processed_docs
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        self.chunked_metadatas = []
        self.chunked_texts = []
        
        for doc in self.docs:
            splits = self.text_splitter.split_text(doc["text"])
            for split in splits:
                if split.strip():
                    self.chunked_texts.append(split)
                    self.chunked_metadatas.append({
                        "page": doc.get("page", 1),
                        "source": doc.get("source", "Uploaded Document")
                    })
                    
        embeddings = get_embedding_model()
        self.vector_store = FAISS.from_texts(self.chunked_texts, embeddings, metadatas=self.chunked_metadatas)
        tokenized_corpus = [text.lower().split(" ") for text in self.chunked_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=4):
        dense_results = self.vector_store.similarity_search_with_score(query, k=top_k * 2)
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        rrf_scores = {}
        for rank, (doc, _) in enumerate(dense_results):
            txt = doc.page_content
            if txt not in rrf_scores:
                rrf_scores[txt] = {"doc": doc, "score": 0.0}
            rrf_scores[txt]["score"] += 1.0 / (60.0 + (rank + 1))
            
        for rank, idx in enumerate(top_bm25_indices):
            txt = self.chunked_texts[idx]
            if txt not in rrf_scores:
                from langchain_core.documents import Document as LC_Doc
                constructed_doc = LC_Doc(page_content=txt, metadata=self.chunked_metadatas[idx])
                rrf_scores[txt] = {"doc": constructed_doc, "score": 0.0}
            rrf_scores[txt]["score"] += 1.0 / (60.0 + (rank + 1))
            
        sorted_rrf = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_rrf[:top_k]]

# -----------------------------------------------------------------------------
# 4. STREAMLIT APP CONFIGURATION & EXACT UI STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nexus Knowledge Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Background Palette */
    .stApp {
        background-color: #05070c;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0b0f17 !important;
        border-right: 1px solid #1a2233;
    }
    
    /* Centered Hero Welcome Box */
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
        margin: 0 auto 28px auto;
        line-height: 1.5;
    }
    
    .feature-card {
        background-color: #0b111e;
        border: 1px solid #1c2638;
        border-radius: 12px;
        padding: 24px;
        max-width: 650px;
        margin: 0 auto;
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

    /* Message Boxes */
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
    }
    
    /* Clean Markdown Tables */
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
        line-height: 1.6;
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
    }
    .stButton>button:hover {
        background-color: #1c2638;
        border-color: #38bdf8;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- AUTHENTICATION DIALOG ---
if not st.session_state.authenticated:
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; margin-bottom: 20px;">
        <h1 style="color: #ffffff; font-weight: 800;">⚡ Nexus Knowledge Copilot</h1>
        <p style="color: #8b949e;">Context-aware, multi-format conversational intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_l, col_center, col_r = st.columns([1, 1.4, 1])
    with col_center:
        tab_login, tab_register = st.tabs(["🔒 Secure Login", "📝 Create Account"])
        
        with tab_login:
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Authenticate Session"):
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
            if st.button("Register Identity"):
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
    st.markdown("### ⚡ NEXUS ENGINE")
    st.caption("Universal Multi-Format RAG Architecture")
    
    st.divider()
    st.markdown("#### 💾 Vector Storage Status")
    if st.session_state.retriever:
        st.success("🟢 Persistent Index Online")
    else:
        st.info("⚪ No Documents Indexed")
        
    st.divider()
    st.markdown("### 📂 Ingestion Hub")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "txt", "py", "md"],
        help="200MB per file • PDF, DOCX, PPTX, TXT, PY, MD"
    )
    
    if st.button("⚡ Process & Index Documents") and uploaded_files:
        parsed_intelligence = []
        with st.spinner("Executing Parallel Document Deconstruction..."):
            for f in uploaded_files:
                file_bytes = f.read()
                extracted_pages = extract_text_from_file(f.name, file_bytes)
                for chunk in extracted_pages:
                    chunk["source"] = f.name
                    parsed_intelligence.append(chunk)
                    
        if parsed_intelligence:
            st.session_state.retriever = HybridRetriever(parsed_intelligence)
            st.success(f"Indexed {len(uploaded_files)} file(s) successfully!")
            st.rerun()
        else:
            st.error("No extractable textual content found.")
            
    st.divider()
    st.markdown("### ⚙️ Operations")
    col_cc, col_wdb = st.columns(2)
    with col_cc:
        if st.button("🧹 Clear Chat"):
            db_conn.execute("DELETE FROM chat_history WHERE username = ?", (st.session_state.username,))
            db_conn.commit()
            st.rerun()
    with col_wdb:
        if st.button("🗑️ Wipe DB"):
            db_conn.execute("DELETE FROM chat_history")
            db_conn.execute("DELETE FROM users")
            db_conn.commit()
            st.session_state.authenticated = False
            st.rerun()

# --- MAIN CHAT & WELCOME SCREEN ---
chat_history_rows = db_conn.execute(
    "SELECT role, message FROM chat_history WHERE username = ? ORDER BY timestamp ASC",
    (st.session_state.username,)
).fetchall()

# If no messages exist yet, show the exact Centered Hero Card Box
if not chat_history_rows:
    st.markdown("""
    <div style="text-align: center; margin-top: 45px;">
        <div class="welcome-pill">⚡ NEXT-GEN AI RAG ENGINE</div>
        <div class="welcome-title">Nexus Knowledge Copilot</div>
        <div class="welcome-subtitle">
            Context-aware, multi-format conversational intelligence powered by LLaMA 3.3, FAISS vector indexing, and RapidOCR.
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
                    <span style="font-weight:700; color:#f0f6fc; font-size:1.05rem;">Core Concepts & Architecture</span>
                </div>
                <div>{msg['message']}</div>
            </div>
            """, unsafe_allow_html=True)

user_query = st.chat_input("Ask anything about your documents, code, or diagrams...")

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
    source_citations = []
    
    if st.session_state.retriever:
        extracted_chunks = st.session_state.retriever.search(user_query, top_k=4)
        for idx, doc in enumerate(extracted_chunks):
            context_str += f"\n[Context Chunk {idx+1}]: {doc.page_content}\n"
            source_citations.append(f"📄 File: `{doc.metadata.get('source')}` | Page: **{doc.metadata.get('page')}**")
            
    if not context_str:
        context_str = "No specific reference documents indexed. Rely on general foundational knowledge."

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a precise, elite enterprise technical RAG copilot.\n"
            "Analyze the provided document contexts to construct your structural answers.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Structure all concepts into clean Markdown tables (| Topic | Key Points |) or bullet lists.\n"
            "- Format tokens, keywords, and code with inline code blocks (`int`, `return`, `cout`).\n"
            "- Never generate raw <br> tags.\n"
            "- Base your answers strictly on the context if relevant.\n\n"
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
                    response_placeholder.markdown(f"""
                    <div class="chat-container-ai">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                            <span style="color:#f59e0b; font-size:1.1rem;">⚡</span>
                            <span style="font-weight:700; color:#f0f6fc; font-size:1.05rem;">Core Concepts & Architecture</span>
                        </div>
                        <div>{full_response}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if source_citations:
                    with st.expander("🔎 Verified Document Citations", expanded=False):
                        for citation in sorted(list(set(source_citations))):
                            st.markdown(citation)
                            
            db_conn.execute(
                "INSERT INTO chat_history (username, role, message) VALUES (?, ?, ?)",
                (st.session_state.username, "assistant", full_response)
            )
            db_conn.commit()
            st.rerun()

        except Exception as e:
            st.error(f"⚠️ Inference Error: {str(e)}")