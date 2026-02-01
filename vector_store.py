import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.join("data", "chroma")
os.makedirs(DB_DIR, exist_ok=True)

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
    )
)

COLLECTION_NAME = "personal_knowledge"
collection = client.get_or_create_collection(COLLECTION_NAME)

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True).tolist()

def add_documents(docs, metadatas=None, ids=None):
    if ids is None:
        ids = [f"doc-{i}" for i in range(len(docs))]
    embeddings = embed_texts(docs)
    collection.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)
    client.persist()

def query_documents(query: str, n_results: int = 5):
    embedding = embed_texts([query])[0]
    res = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))
