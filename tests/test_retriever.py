import pytest
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app import load_embeddings, create_hybrid_retriever

@pytest.fixture(scope="module")
def sample_documents():
    return [
        Document(page_content="Git stash temporarily shelves changes in your working copy and repository.", metadata={"source": "git.pdf"}),
        Document(page_content="The Java Virtual Machine JVM executes compiled Java bytecode on any operating system.", metadata={"source": "java.pdf"}),
        Document(page_content="Python lists are mutable ordered sequence collections of elements.", metadata={"source": "python.pdf"}),
        Document(page_content="C++ pointers store raw memory addresses of variables directly in hardware RAM.", metadata={"source": "cpp.pdf"})
    ]

def test_hybrid_retriever_exact_keyword(sample_documents):
    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(sample_documents, embeddings)
    hybrid_retriever = create_hybrid_retriever(vector_store, all_chunks=sample_documents)
    
    results = hybrid_retriever.invoke("JVM bytecode")
    assert len(results) > 0
    assert "Java Virtual Machine" in results[0].page_content

def test_hybrid_retriever_semantic_similarity(sample_documents):
    embeddings = load_embeddings()
    vector_store = FAISS.from_documents(sample_documents, embeddings)
    hybrid_retriever = create_hybrid_retriever(vector_store, all_chunks=sample_documents)
    
    results = hybrid_retriever.invoke("shelving modified repository files temporarily")
    assert len(results) > 0
    assert "stash" in results[0].page_content.lower()