import time
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def benchmark():
    print("=" * 60)
    print("⚡ NEXUS RAG PIPELINE PERFORMANCE PROFILER")
    print("=" * 60)

    # 1. Embedding Model Loading Latency
    t0 = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64}
    )
    load_time = time.perf_counter() - t0
    print(f"🔹 HuggingFace Model Initialization: {load_time:.3f}s")

    # 2. Batch Embedding Generation & FAISS Indexing Throughput
    num_chunks = 500
    docs = [
        Document(
            page_content=f"Chunk {i}: Concurrency control, multithreading synchronization, and memory consistency models.",
            metadata={"id": i}
        ) for i in range(num_chunks)
    ]

    t1 = time.perf_counter()
    vector_store = FAISS.from_documents(docs, embeddings)
    index_time = time.perf_counter() - t1
    throughput = num_chunks / index_time
    print(f"🔹 FAISS Vector Indexing ({num_chunks} chunks): {index_time:.3f}s ({throughput:.1f} chunks/sec)")

    # 3. Vector Similarity Search Latency
    t2 = time.perf_counter()
    results = vector_store.similarity_search("thread synchronization", k=4)
    search_ms = (time.perf_counter() - t2) * 1000
    print(f"🔹 Vector Search Latency (Top-4 Retrieval): {search_ms:.2f} ms")
    print("=" * 60)

if __name__ == "__main__":
    benchmark()