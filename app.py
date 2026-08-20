import os
import io
import sqlite3
import hashlib
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. SECURE API KEY & ENVIRONMENT SYNCHRONIZATION
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
# 4. STREAMLIT ULTRA-AESTHETIC DARK UI & CYBER GLASSMORPHISM
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NEXUS // Hybrid RAG Copilot",
    page_icon="🦇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep aesthetic CSS injection: Neon accents, frosted glass, smooth borders, and glow states
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    /* Global Root Styling */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d111a 0%, #07090e 100%);
        color: #e2e8f0;
    }

    /* Ambient Header Glow */
    .hero-container {
        padding: 24px 28px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.75), rgba(8, 12, 22, 0.85));
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        backdrop-filter: blur(14px);
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Sidebar Aesthetics */
    [data-testid="stSidebar"] {
        background-color: #080c14 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .sidebar-user-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Modern Dark Chat Bubbles */
    .user-bubble {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-left: 3px solid #818cf8;
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
    }

    .assistant-bubble {
        background: linear-gradient(135deg, rgba(13, 18, 30, 0.95), rgba(7, 10, 18, 0.9));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-left: 3px solid #38bdf8;
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(10px);
    }

    .bubble-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Structured Markdown Tables */
    table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 16px 0;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: #080c14;
    }
    th {
        background: linear-gradient(90deg, #131b2e 0%, #17223b 100%) !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.2) !important;
    }
    td {
        background-color: transparent !important;
        color: #cbd5e1 !important;
        font-size: 0.92rem;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        line-height: 1.6;
    }
    tr:last-child td {
        border-bottom: none;
    }
    tr:hover td {
        background-color: rgba(56, 189, 248, 0.03) !important;
    }

    /* Code Tags */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(56, 189, 248, 0.08) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        padding: 2px 7px !important;
        border-radius: 5px !important;
        font-size: 0.85em !important;
    }

    /* Custom Streamlit Form Elements & Inputs */
    .stButton>button {
        background: linear-gradient(135deg, #131b2e, #0e1626);
        color: #f1f5f9;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        padding: 8px 16px;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        border-color: #38bdf8;
        color: #38bdf8;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
        transform: translateY(-1px);
    }
    .stButton>button:active {
        transform: translateY(0);
    }

    /* Expander Styling */
    div[data-testid="stExpander"] {
        background: rgba(13, 18, 30, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(8px);
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #080c14;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e293b;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
</style>
""", unsafe_allowed_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- AUTHENTICATION DIALOG ---
if not st.session_state.authenticated:
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; margin-bottom: 25px;">
        <h1 style="font-size: 2.4rem; font-weight: 800; background: linear-gradient(90deg, #f8fafc, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚡ NEXUS INTELLIGENCE</h1>
        <p style="color: #64748b; font-size: 0.95rem; font-family: 'JetBrains Mono', monospace;">ENTERPRISE MULTI-DOC HYBRID RAG CORE</p>
    </div>
    """, unsafe_allowed_html=True)
    
    _, col_center, _ = st.columns([1, 1.4, 1])
    with col_center:
        tab_login, tab_register = st.tabs(["🔒 Secure Login", "📝 Create Account"])
        
        with tab_login:
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("AUTHENTICATE SESSION"):
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
            if st.button("REGISTER IDENTITY"):
                if len(reg_user.strip()) < 3 or len(reg_pass) < 4:
                    st.warning("Username must be $\ge 3$ characters, Password $\ge 4$ characters.")
                else:
                    h = hashlib.sha256(reg_pass.encode()).hexdigest()
                    try:
                        with get_db_connection() as conn:
                            conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (reg_user, h))
                            conn.commit()
                        st.success("Account created successfully! Switch to Login to proceed.")
                    except sqlite3.IntegrityError:
                        st.error("Username is already taken.")
    st.stop()

# --- ACTIVE LOGGED-IN SESSION ---
db_conn = get_db_connection()

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-user-card">
        <div>
            <div style="font-size: 0.7rem; color: #64748b; font-family: 'JetBrains Mono', monospace;">ACTIVE USER</div>
            <div style="font-weight: 700; color: #f8fafc;">{st.session_state.username}</div>
        </div>
        <span style="font-size: 1.1rem;">🦇</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Log Out"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.retriever = None
        st.rerun()
        
    st.divider()
    st.markdown("<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.75rem; color: #64748b; letter-spacing: 0.5px;'>SYSTEM ARCHITECTURE</div>", unsafe_allow_html=True)
    
    if st.session_state.retriever:
        st.markdown("""
        <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 8px; padding: 8px 12px; margin-top: 6px;">
            <div style="color: #4ade80; font-size: 0.8rem; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                <span style="height: 6px; width: 6px; background-color: #4ade80; border-radius: 50%; display: inline-block;"></span>
                HYBRID INDEX ONLINE
            </div>
            <div style="font-size: 0.7rem; color: #86efac; margin-top: 2px;">Dense FAISS + Sparse BM25 (RRF)</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 8px 12px; margin-top: 6px;">
            <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                <span style="height: 6px; width: 6px; background-color: #94a3b8; border-radius: 50%; display: inline-block;"></span>
                INDEX STANDBY
            </div>
            <div style="font-size: 0.7rem; color: #64748b; margin-top: 2px;">Awaiting file ingestion...</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    st.markdown("<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.75rem; color: #64748b; letter-spacing: 0.5px;'>DOCUMENT INGESTION</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "txt", "py", "md"],
        label_visibility="collapsed"
    )
    
    if st.button("⚡ Process & Ingest Files") and uploaded_files:
        parsed_intelligence = []
        with st.spinner("Deconstructing & Embedding Intelligence..."):
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
    st.markdown("<div style='font-family: \"JetBrains Mono\", monospace; font-size: 0.75rem; color: #64748b; letter-spacing: 0.5px;'>OPERATIONS</div>", unsafe_allow_html=True)
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

# --- MAIN HERO HEADER ---
st.markdown("""
<div class="hero-container">
    <div>
        <h1 class="hero-title">⚡ Nexus Hybrid RAG</h1>
        <p style="color: #94a3b8; font-size: 0.88rem; margin: 4px 0 0 0;">Next-Gen Enterprise Multi-Document Context Ingestion & Neural Search</p>
    </div>
    <div class="hero-badge">Groq LLaMA 3.3 Engine</div>
</div>
""", unsafe_allow_html=True)

# Fetch History
chat_history_rows = db_conn.execute(
    "SELECT role, message FROM chat_history WHERE username = ? ORDER BY timestamp ASC",
    (st.session_state.username,)
).fetchall()

# Display Chat History with Aesthetic Bubbles
for msg in chat_history_rows:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-bubble">
            <div class="bubble-meta" style="color: #a5b4fc;">
                <span>👤</span> USER QUERY
            </div>
            <div style="color: #f1f5f9; font-size: 0.95rem; line-height: 1.5;">{msg['message']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-bubble">
            <div class="bubble-meta" style="color: #38bdf8;">
                <span>⚡</span> NEXUS INTELLIGENCE SYNTHESIS
            </div>
            <div style="color: #e2e8f0; font-size: 0.94rem;">{msg['message']}</div>
        </div>
        """, unsafe_allow_html=True)

# Quick Action Pill Buttons
st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
col_p1, col_p2, col_p3 = st.columns(3)
preset_clicked = None
with col_p1:
    if st.button("💡 Summarize Core Concepts"):
        preset_clicked = "Summarize the core concepts and fundamental architecture covered across the documents in a clean Markdown table."
with col_p2:
    if st.button("⚔️ Compare Key Differences"):
        preset_clicked = "Identify the key components and compare operational structural differences in a clear table."
with col_p3:
    if st.button("📋 Generate Practice Quiz"):
        preset_clicked = "Generate a comprehensive technical practice evaluation quiz with questions and answer explanations."

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
    source_citations = []
    
    if st.session_state.retriever:
        extracted_chunks = st.session_state.retriever.search(user_query, top_k=4)
        for idx, doc in enumerate(extracted_chunks):
            context_str += f"\n[Context Chunk {idx+1}]: {doc.page_content}\n"
            source_citations.append(f"📄 File: `{doc.metadata.get('source')}` | Reference Page: **{doc.metadata.get('page')}**")
            
    if not context_str:
        context_str = "No specific reference documents indexed. Rely on foundational knowledge."

    system_prompt = f"""You are an elite, highly intelligent technical RAG copilot.
Use the retrieved document context below to answer the user request accurately and thoroughly.

CRITICAL FORMATTING RULES:
- Use clean Markdown tables (| Header 1 | Header 2 |) with bold headers for all structured comparisons.
- Never output raw `<br>` HTML tags.
- Deliver rich, well-organized technical explanations.

CONTEXT:
{context_str}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    if not groq_api_key:
        st.error("🔑 GROQ_API_KEY is not configured! Add it to Streamlit Secrets or your .env file.")
    else:
        try:
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                groq_api_key=groq_api_key,
                temperature=0.2,
                streaming=True
            )
            
            pipeline = prompt | llm | StrOutputParser()
            
            with st.chat_message("assistant"):
                def stream_gen():
                    for chunk in pipeline.stream({"question": user_query}):
                        yield chunk
                        
                full_response = st.write_stream(stream_gen)
                
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