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

# Synchronize Streamlit Cloud Secrets with OS Environment
groq_api_key = ""
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"].strip()
    os.environ["GROQ_API_KEY"] = groq_api_key
elif os.getenv("GROQ_API_KEY"):
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

# LangChain & Engine Dependencies
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
# 2. PERSISTENT STORAGE & AUTHENTICATION (SQLite)
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
# 3. DOCUMENT INGESTION & PARALLEL OCR EXTRACTION
# -----------------------------------------------------------------------------
ocr_engine = RapidOCR()

def parse_pdf_page(page_data):
    page_num, page_layout = page_data
    text = page_layout.get_text().strip()
    
    # RapidOCR fallback if no text layer found
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
                
    else:  # Raw Text / Code Files
        try:
            decoded_text = file_bytes.decode("utf-8")
            if decoded_text.strip():
                chunks.append({"page": 1, "text": decoded_text})
        except Exception:
            pass
            
    return chunks

# -----------------------------------------------------------------------------
# 4. HYBRID RETRIEVAL PIPELINE (FAISS + BM25 via RRF)
# -----------------------------------------------------------------------------
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
                    
        # FAISS Dense Index
        embeddings = get_embedding_model()
        self.vector_store = FAISS.from_texts(self.chunked_texts, embeddings, metadatas=self.chunked_metadatas)
        
        # BM25 Sparse Index
        tokenized_corpus = [text.lower().split(" ") for text in self.chunked_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=4):
        dense_results = self.vector_store.similarity_search_with_score(query, k=top_k * 2)
        
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Reciprocal Rank Fusion (RRF)
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
# 5. STREAMLIT UI CONFIGURATION & USER SESSION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Nexus RAG Engine", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Dark Glassmorphism Styling
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: #e6edf3; }
.stButton>button { width: 100%; background-color: #21262d; color: #f0f6fc; border: 1px solid #30363d; border-radius: 6px; }
.stButton>button:hover { background-color: #30363d; border-color: #8b949e; }
div[data-testid="stExpander"] { background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; }
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
    st.title("⚡ Nexus Production Multi-Doc RAG")
    st.caption("Secure multi-user hybrid document intelligence platform.")
    
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
                st.error("Invalid username or password.")
                
    with tab_register:
        reg_user = st.text_input("New Username", key="reg_user")
        reg_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register Account"):
            if len(reg_user.strip()) < 3 or len(reg_pass) < 4:
                st.warning("Username must be >= 3 characters, Password >= 4 characters.")
            else:
                h = hashlib.sha256(reg_pass.encode()).hexdigest()
                try:
                    with get_db_connection() as conn:
                        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (reg_user, h))
                        conn.commit()
                    st.success("Account created! Switch to Login tab to proceed.")
                except sqlite3.IntegrityError:
                    st.error("Username is already taken.")
    st.stop()

# --- ACTIVE LOGGED-IN SESSION ---
db_conn = get_db_connection()

with st.sidebar:
    st.markdown(f"👤 **Logged in as:** `{st.session_state.username}`")
    if st.button("🚪 Log Out"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.retriever = None
        st.rerun()
        
    st.divider()
    st.subheader("⚡ NEXUS ENGINE")
    if st.session_state.retriever:
        st.success("🟢 Hybrid Index Active (BM25 + FAISS)")
    else:
        st.info("⚪ No Documents Indexed")
        
    st.divider()
    st.subheader("📂 Document Hub")
    uploaded_files = st.file_uploader(
        "Upload Intelligence Files",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "txt", "py", "md"]
    )
    
    if st.button("⚡ Process & Index Documents") and uploaded_files:
        parsed_intelligence = []
        with st.spinner("Extracting & Indexing Multi-Format Documents..."):
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
            st.error("No valid text or OCR readable content found.")
            
    st.divider()
    st.subheader("⚙️ Operations")
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

# --- MAIN INTERACTIVE INTERFACE ---
st.title("⚡ Nexus Enterprise Multi-Doc Hybrid RAG")

# Render Chat History
chat_history_rows = db_conn.execute(
    "SELECT role, message FROM chat_history WHERE username = ? ORDER BY timestamp ASC",
    (st.session_state.username,)
).fetchall()

for msg in chat_history_rows:
    with st.chat_message(msg["role"]):
        st.markdown(msg["message"])

# Quick Suggested Prompts
st.divider()
col_p1, col_p2, col_p3 = st.columns(3)
preset_clicked = None
with col_p1:
    if st.button("💡 Summarize Core Concepts"):
        preset_clicked = "Summarize core concepts and fundamental architecture covered across the documents."
with col_p2:
    if st.button("⚔️ Compare Key Differences"):
        preset_clicked = "Identify the key components and compare operational structural differences."
with col_p3:
    if st.button("📋 Generate Practice Quiz"):
        preset_clicked = "Generate a short evaluation practice quiz with answers based on the document content."

user_query = st.chat_input("Ask anything about your documents, code, or diagrams...")
if preset_clicked:
    user_query = preset_clicked

# -----------------------------------------------------------------------------
# 6. RETRIEVAL & INFERENCE PIPELINE
# -----------------------------------------------------------------------------
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
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

    system_prompt = f"""You are a precise, professional enterprise technical RAG copilot.
Use the retrieved document context below to answer the user request accurately and thoroughly.

CRITICAL INSTRUCTIONS:
- Use clean Markdown tables, lists, and bold headers to format explanations.
- Never output raw `<br>` HTML tags.
- Base your answers strictly on the context if relevant.

CONTEXT:
{context_str}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    if not groq_api_key:
        st.error("🔑 GROQ_API_KEY is not configured! Please provide it in Streamlit Secrets or your .env file.")
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